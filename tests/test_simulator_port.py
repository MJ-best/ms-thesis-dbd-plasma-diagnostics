"""The browser simulator app/circuit_simulator.html runs a JavaScript port of examples/vrms_simulation.py.
Both must give the same numbers for the same circuit (skipped without Node.js)."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dbd_diagnostics import features, lissajous

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import cell_count_check as cc  # noqa: E402
import vrms_simulation as vs  # noqa: E402

NODE = shutil.which("node")
P = cc.M_CIRCUIT  # 10.6 fit to the healthy M1-M3 captures


def browser(**kw):
    """DBD.simulate of the page's own script for the 10.6 circuit with the page's controls changed by kw."""
    html = (ROOT / "app" / "circuit_simulator.html").read_text(encoding="utf-8")
    sim = html.split('<script id="sim">', 1)[1].split("</script>", 1)[0]
    p = dict(Cd=P["C_d"], Cg=P["C_g"], Ub=P["U_b"], spread=P["spread"], E=P["E"], R=P["R"], f=vs.F0,
             nExposed=0, gamma=0.1, nThinned=0, theta=0.5, nDischarging=126) | kw
    code = sim + f"\nconst s = DBD.simulate({json.dumps(p)});\n" \
                 "process.stdout.write(JSON.stringify({truth: s.truth, S: s.S, liss: {...s.liss, loop: undefined}}));\n"
    out = subprocess.run([NODE, "-"], input=code, capture_output=True, text=True, check=True, timeout=120)
    return json.loads(out.stdout)


def python(dev):
    """The same quantities from vrms_simulation: truth, S (3 % heavier load) and the Lissajous fit per 10 ns period."""
    t, v, Q = vs.run(dev, E=P["E"], R=P["R"])
    h = vs.truth(t, v, Q)
    e = vs.truth(*vs.run(dict(dev, Cd=dev["Cd"] * 1.03, Cg=dev["Cg"] * 1.03), E=P["E"], R=P["R"]))
    S = -np.log(e["v_rms"] / h["v_rms"]) / np.log(e["q_per_v"] / h["q_per_v"])
    t, v, Q = t[::5], v[::5], Q[::5]
    rows = []
    for sl in features._periods(t, v):
        q = Q[sl] - 0.5 * (Q[sl].max() + Q[sl].min())
        L = lissajous.analyse(t[sl], v[sl] - v[sl].mean(), q)
        rows.append(dict(c_total=L.C_total, c_d=L.C_dielectric, dq=L.dQ, v_on=L.V_on, u_b=L.U_b_onset, energy=L.E_cycle))
    return h, S, {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


@pytest.mark.skipif(NODE is None, reason="needs Node.js")
def test_exposed_electrode_matches_python():
    dev = vs.exposed(vs.device(P["C_d"], P["C_g"], P["U_b"], P["spread"]), 48)
    h, S, L = python(dev)
    js = browser(nExposed=48)
    for k in h:
        assert js["truth"][k] == pytest.approx(h[k], rel=1e-7)
    assert js["S"] == pytest.approx(S, rel=1e-6)
    for k in L:
        assert js["liss"][k] == pytest.approx(L[k], rel=1e-7)


@pytest.mark.skipif(NODE is None, reason="needs Node.js")
def test_one_cell_matches_python():
    h, S, _ = python(cc.few_cells(vs.device(P["C_d"], P["C_g"], P["U_b"], P["spread"]), 1))
    js = browser(nDischarging=1)
    for k in h:
        assert js["truth"][k] == pytest.approx(h[k], rel=1e-7)
    assert js["S"] == pytest.approx(S, rel=1e-6)
