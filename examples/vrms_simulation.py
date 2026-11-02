"""Equivalent-circuit simulation of the V_rms hypothesis: does damage make the cells start discharging earlier
and lower the voltage of the driving power supply?

    python examples/vrms_simulation.py [docs/img]

No raw data are needed. The model is the 126-pair device as a set of per-cell Manley circuits (dielectric C_d,i
in series with the air gap C_g,i; while a cell discharges its air-gap voltage stays at its burning voltage
U_b,i), with burning voltages spread over the cells, driven by its own driving power supply: an open-circuit
voltage E behind an output resistance R (a Thevenin equivalent). The simulated waveforms pass through the scope chain of
dbd_diagnostics.simulate.cells (60 MHz, 10 ns, 8 bit) and are analysed with dbd_diagnostics.features,
exactly like the measured captures.

1. The model is fitted to the HEALTHY data only: the medians of the 5-10 h captures of all devices and the
   load sensitivity S = 0.52 of the healthy M3 (docs/raw-data-verification.md, section 10.5).
2. Damage is then predicted, without any fit, for two kinds of damage: thinned dielectric in some cells,
   and exposed electrode in some cells (burning voltage lowered by the Townsend factor for gamma 0.01 -> 0.1).
3. Random discharge initiation (burning-voltage jitter per discharge) and supply fluctuation give the effect-to-noise ranking.
4. A supply sweep checks the trends of the measured sweep.
Prints the tables of section 10.6 of docs/raw-data-verification.md and writes vrms_simulation.png and
vrms_simulation_model.png (circuit, mechanism, simulated air-gap voltages and Lissajous graphs).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from dbd_diagnostics import features
from dbd_diagnostics.io import Waveform

F0 = 46.8e3  # drive frequency of the healthy captures [Hz]
# healthy 5-10 h captures of M1-M3 (10 ns sampling), medians; S from the healthy M3's sessions
HEALTHY = dict(v_rms=1494.0, v_max=2280.0, c_total=12.15e-12, c_dielectric=32.4e-12, q_per_v=25.3e-12, von_frac=0.199,
               u_b=855.0, energy=0.126e-3)
S_HEALTHY = 0.52
# damaged minus M3 in the same sessions, session medians, % (section 10.5): M1, M2
DAMAGE = dict(v_rms=(-2.85, -2.31), q_per_v=(6.56, 5.08), von_frac=(-33.3, -31.0), u_b=(-13.6, -1.3),
              c_total=(2.36, -1.37), c_dielectric=(-4.05, 2.55), energy=(-15.6, 7.45))
# effect / within-session noise in the data (section 10.5, 0 where M1 and M2 disagree in sign)
DATA_EN = dict(v_rms=4.3, von_frac=1.3, i_rms=1.2, q_per_v=0.7, q_max=0.3, u_b=0.2, c_total=0.0, energy=0.0, c_dielectric=0.0)
LAB = {"v_rms": "V_rms", "v_max": "V_max", "c_total": "C_total", "c_dielectric": "C_d", "q_per_v": "ΔQ/V_max",
       "von_frac": "V_on/V_max", "u_b": "U_b", "energy": "E", "i_rms": "I_rms", "q_max": "ΔQ", "S": "S"}


def paschen(gamma, pd_torr_cm=760 * 0.01, A=15.0, B=365.0):
    """Townsend breakdown voltage of a 100 um air gap, gamma(exp(alpha d) - 1) = 1."""
    return B * pd_torr_cm / np.log(A * pd_torr_cm / np.log(1 + 1 / gamma))


EXPOSED = paschen(0.1) / paschen(0.01)  # metal cathode (exposed Ag) vs dielectric: 718 / 865 V


def device(C_d, C_g, U_b, spread, n=126, seed=0):
    z = np.random.default_rng(seed).standard_normal(n)
    return dict(Cd=np.full(n, C_d / n), Cg=np.full(n, C_g / n), Ub=U_b * np.exp(spread * z - spread ** 2 / 2))


def pick(dev, k, seed=0):
    return np.random.default_rng(seed).choice(len(dev["Cd"]), k, replace=False)


def thinned(dev, k, theta):
    """Dielectric of k cells thinned by the fraction theta: C_d,i / (1 - theta); gap and U_b unchanged."""
    Cd = dev["Cd"].copy()
    Cd[pick(dev, k)] /= 1 - theta
    return dict(dev, Cd=Cd)


def exposed(dev, k, factor=EXPOSED):
    """Electrode exposed in k cells: their burning voltage times the Townsend factor."""
    Ub = dev["Ub"].copy()
    Ub[pick(dev, k)] *= factor
    return dict(dev, Ub=Ub)


def run(dev, E, R, f=F0, settle=4, periods=6, dt=2e-9, jitter=0.0, seed=1, trace=None):
    """Time-domain solution: power-supply current (e - v)/R charges the node; a discharging cell holds its
    air-gap voltage at its burning voltage.

    With ``trace`` (cell indices) the air-gap voltages of those cells are returned as a fourth array."""
    rng = np.random.default_rng(seed)
    Cd, Cg, Ub = dev["Cd"], dev["Cg"], dev["Ub"]
    Cc = Cd * Cg / (Cd + Cg)
    kf, Cnode, w = Cc / Cg, Cc.sum(), 2 * np.pi * f
    n = int(round((settle + periods) / f / dt))
    thr = Ub * (1 + jitter * rng.standard_normal(len(Ub))) if jitter else Ub.copy()
    qc = np.zeros(len(Ub))  # charge carried through each air gap
    v = 0.0
    V, Q = np.empty(n), np.empty(n)
    U = np.empty((n, len(trace))) if trace is not None else None
    for k in range(n):
        v += dt * (E * np.sin(w * k * dt) - v) / R / Cnode
        ug = v - Cc * (v + qc / Cg) / Cd
        fire = np.abs(ug) > thr
        if fire.any():
            # hold u_g at +/-U_b,i: qc = a v + b with a = C_d; the node charge is conserved through the step
            a = Cd[fire]
            b = -np.sign(ug[fire]) * thr[fire] * Cd[fire] * Cg[fire] / Cc[fire]
            v = (v * Cnode - np.sum(kf[fire] * (b - qc[fire]))) / (Cnode + np.sum(kf[fire] * a))
            qc[fire] = a * v + b
            if jitter:  # statistical spread of the next discharge initiation
                thr[fire] = Ub[fire] * (1 + jitter * rng.standard_normal(fire.sum()))
        V[k], Q[k] = v, np.sum(Cc * v + kf * qc)
        if U is not None:
            U[k] = v - Cc[trace] * (v + qc[trace] / Cg[trace]) / Cd[trace]
    t = np.arange(n) * dt
    keep = t >= settle / f
    out = (t[keep] - t[keep][0], V[keep], Q[keep])
    return out + (U[keep],) if U is not None else out


def truth(t, v, Q):
    return dict(v_rms=float(np.sqrt(np.mean(v ** 2))), v_max=float(np.max(np.abs(v))),
                q_per_v=float((Q.max() - Q.min()) / 2 / np.max(np.abs(v))))


def measure(t, v, Q, seed=2, bandwidth=60e6, adc_v=40.0, adc_i=8e-3, noise_i=2e-3, cap_len=5000):
    """Scope chain of simulate.cells, 5000-point captures, then features.extract as for the measured data."""
    rng = np.random.default_rng(seed)
    dt = t[1] - t[0]
    i = np.gradient(Q, t)
    a, acc = np.exp(-dt * 2 * np.pi * bandwidth), i[0]
    for k in range(len(i)):
        acc = a * acc + (1 - a) * i[k]
        i[k] = acc
    step = int(round(1e-8 / dt))
    t, v, i = t[::step], v[::step], i[::step]
    i = np.round((i + noise_i * rng.standard_normal(len(i))) / adc_i) * adc_i
    v = np.round(v / adc_v) * adc_v
    rows = []
    for s in range(0, len(t) - cap_len + 1, cap_len // 2):
        sl = slice(s, s + cap_len)
        try:
            r = features.extract(Waveform(t=t[sl] - t[sl][0], v=v[sl], i=i[sl], meta={}))
        except ValueError:  # no full period in the window
            continue
        r = {k: x for k, x in r.items() if isinstance(x, (int, float))}
        r["q_per_v"], r["von_frac"] = r["q_max"] / r["v_max"], r["v_on_lissajous"] / r["v_max"]
        rows.append(r)
    return rows


def medians(dev, src, seeds=(1, 2), **kw):
    rows = []
    for s in seeds:
        rows += measure(*run(dev, **src, seed=s, **kw), seed=s + 50)
    return pd.DataFrame(rows).median(numeric_only=True)


def fit():
    """Device (C_d, C_g, U_b, spread) and driving power supply (R, E) from the healthy targets and S."""
    def unpack(x):
        return dict(C_d=np.exp(x[0]) * 1e-12, C_g=np.exp(x[1]) * 1e-12, U_b=x[2], spread=abs(x[3]), R=np.exp(x[4]) * 1e3, E=x[5])

    def model(p):
        dev = device(p["C_d"], p["C_g"], p["U_b"], p["spread"])
        t, v, Q = run(dev, E=p["E"], R=p["R"])
        o = pd.DataFrame(measure(t, v, Q)).median(numeric_only=True)
        env = truth(*run(dict(dev, Cd=dev["Cd"] * 1.03, Cg=dev["Cg"] * 1.03), E=p["E"], R=p["R"]))  # environment-like load
        h = truth(t, v, Q)
        o["S"] = -np.log(env["v_rms"] / h["v_rms"]) / np.log(env["q_per_v"] / h["q_per_v"])
        return o

    def loss(x):
        p = unpack(x)
        if not (300 < p["U_b"] < 2000 and 1e3 < p["R"] < 1e7 and p["E"] > 500):
            return 1e6
        o = model(p)
        if not np.isfinite(o[list(HEALTHY) + ["S"]]).all():
            return 1e6
        w = {"v_rms": 3, "v_max": 3}
        return sum(w.get(k, 1) * ((o[k] - v) / v) ** 2 for k, v in HEALTHY.items()) + 3 * ((o["S"] - S_HEALTHY) / S_HEALTHY) ** 2

    res = minimize(loss, [np.log(34), np.log(20), 850, 0.15, np.log(200), 4200], method="Nelder-Mead",
                   options=dict(maxfev=240, xatol=1e-3, fatol=1e-4))
    p = unpack(res.x)
    return p, model(p)


def main(out="."):
    pd.set_option("display.width", 200)
    print("1. fit to the healthy data only")
    p, o = fit()
    print("  parameters: " + ", ".join(f"{k} {v * 1e12:.1f} pF" if k in ("C_d", "C_g") else f"{k} {v:.4g}" for k, v in p.items()))
    calib = pd.DataFrame({"data": pd.Series(HEALTHY | {"S": S_HEALTHY}), "model": o[list(HEALTHY) + ["S"]]})
    calib["model/data"] = calib.model / calib.data
    pf = [k for k in calib.index if k.startswith(("c_", "q_"))]
    show = calib.copy()
    show.loc[pf, ["data", "model"]] *= 1e12  # pF
    show.loc["energy", ["data", "model"]] *= 1e3  # mJ
    print("  (capacitances and charge per volt in pF, energy in mJ)")
    print(show.round(3).to_string())
    dev, src = device(p["C_d"], p["C_g"], p["U_b"], p["spread"]), dict(E=p["E"], R=p["R"])

    K = ["v_rms", "q_per_v", "von_frac", "u_b", "c_total", "c_dielectric", "energy"]
    base = medians(dev, src, seeds=(1, 2, 3))
    print(f"\n2. damage, predicted (no fit), change against the healthy model [%]; Townsend factor {EXPOSED:.3f} "
          f"({paschen(0.01):.0f} -> {paschen(0.1):.0f} V)")
    cases = {"thinned dielectric, 8 cells x0.5": thinned(dev, 8, .5), "thinned dielectric, all cells -5 %": thinned(dev, 126, .05)}
    cases |= {f"exposed electrode, {k} cells": exposed(dev, k) for k in (16, 32, 48, 64, 96)}
    dmg = pd.DataFrame({name: 100 * np.log(medians(d, src, seeds=(1, 2, 3))[K] / base[K]) for name, d in cases.items()}).T
    dmg.loc["data M1"] = [DAMAGE[k][0] for k in K]
    dmg.loc["data M2"] = [DAMAGE[k][1] for k in K]
    print(dmg.rename(columns=LAB).round(1).to_string())

    print("\n3. effect / noise with random discharge initiation (jitter 5 % per discharge) and 0.5 % supply fluctuation per run")
    feat = ["v_rms", "i_rms", "energy", "c_total", "c_dielectric", "q_max", "q_per_v", "von_frac", "u_b"]
    rng = np.random.default_rng(7)
    runs = {"healthy": [], "damaged": []}
    for s in range(8):
        for name, d in (("healthy", dev), ("damaged", exposed(dev, 48))):
            runs[name] += measure(*run(d, E=p["E"] * (1 + .005 * rng.standard_normal()), R=p["R"], jitter=.05, seed=20 + s),
                                  seed=70 + s)
    lh, lx = (np.log(pd.DataFrame(runs[k])[feat].abs()) for k in ("healthy", "damaged"))
    en = ((lx.median() - lh.median()) / lh.std()).abs().sort_values(ascending=False)
    print("  " + ", ".join(f"{LAB[k]} {v:.1f} (data {DATA_EN[k]:.1f})" for k, v in en.items()))

    print("\n4. supply sweep (prediction): V_max 2240 / 2330 / 2390 / 2570 V as measured")
    h = truth(*run(dev, **src))
    sw = []
    for vmax in (2240, 2330, 2390, 2570):
        E = p["E"] * vmax / h["v_max"]
        for _ in range(4):
            E *= vmax / truth(*run(dev, E=E, R=p["R"]))["v_max"]
        sw.append(medians(dev, dict(E=E, R=p["R"]), seeds=(1,))[["v_max", "v_rms", "q_per_v", "c_dielectric", "von_frac", "u_b"]])
    sw = pd.DataFrame(sw)
    slope = lambda a, b: np.polyfit(np.log(a), np.log(b), 1)[0]  # noqa: E731
    print(sw.assign(q_per_v=sw.q_per_v * 1e12, c_dielectric=sw.c_dielectric * 1e12).round(3).to_string(index=False))
    print(f"  dln(ΔQ/V)/dlnV {slope(sw.v_rms, sw.q_per_v):+.2f} (data +2.7..+8.2); dlnC_d/dlnV_max {slope(sw.v_max, sw.c_dielectric):+.2f} "
          f"(data +2.5); V_on/V_max {sw.von_frac.iloc[0]:.3f} -> {sw.von_frac.iloc[-1]:.3f} (data 0.165 -> 0.130)")
    figure(calib, dmg, en, out)
    figure_model(p, out)


def figure(calib, dmg, en, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, AQUA, YELLOW, INK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.8, "axes.edgecolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": GREY,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "legend.frameon": False})
    fig, ax = plt.subplots(2, 2, figsize=(8, 7))

    a = ax[0, 0]  # (a) calibration on healthy data
    y = np.arange(len(calib))[::-1]
    a.axvspan(.9, 1.1, color=GREY, lw=0)
    a.axvline(1, color=INK, lw=.8)
    a.plot(calib["model/data"], y, "o", color=BLUE, ms=6, mec="white", mew=.7)
    a.set_yticks(y, [LAB[k] for k in calib.index])
    a.set(xlim=(.8, 1.2), xlabel="model / data (shade ±10 %)", title="(a) Fitted to healthy data only")
    a.grid(axis="y", visible=False)

    a = ax[0, 1]  # (b) damage signatures: thinned vs exposed vs data
    keys = ["v_rms", "q_per_v", "von_frac", "u_b"]
    x = np.arange(len(keys))
    for row, c, m, lab, dx in (("data M1", INK, "o", "data M1", -.24), ("data M2", INK, "s", "data M2", -.12),
                               ("thinned dielectric, 8 cells x0.5", YELLOW, "v", "thinned (8 cells x0.5)", .06),
                               ("exposed electrode, 48 cells", BLUE, "D", "exposed (48 cells)", .2)):
        filled = "data" not in row
        a.plot(x + dx, dmg.loc[row, keys], m, color=c, mfc=c if filled else "white", ms=6, mew=1.1, label=lab)
    a.axhline(0, color=INK, lw=.8)
    a.set_yscale("symlog", linthresh=5)
    a.set_ylim(-70, 20)
    a.set_yticks([-50, -30, -10, -5, -2, 0, 2, 5, 10], ["-50", "-30", "-10", "-5", "-2", "0", "2", "5", "10"])
    a.set_xticks(x, [LAB[k] for k in keys])
    a.set(ylabel="change against healthy [%] (symlog)", title="(b) Only the exposed electrode reproduces all four")
    a.legend(loc="lower left")
    a.grid(axis="x", visible=False)

    a = ax[1, 0]  # (c) share of exposed cells
    rows = [r for r in dmg.index if r.startswith("exposed")]
    share = [int(r.split(", ")[1].split()[0]) / 1.26 for r in rows]
    for k, c in (("v_rms", BLUE), ("q_per_v", ORANGE), ("von_frac", AQUA)):
        lo, hi = sorted(DAMAGE[k])
        a.axhspan(lo, hi, color=c, alpha=.15, lw=0)
        a.plot(share, dmg.loc[rows, k], "o-", color=c, ms=5, mec="white", mew=.7, lw=1.2, label=LAB[k])
    a.axhline(0, color=INK, lw=.8)
    a.set_yscale("symlog", linthresh=5)
    a.set_yticks([-80, -40, -20, -10, -5, -2, 0, 2, 5, 10], ["-80", "-40", "-20", "-10", "-5", "-2", "0", "2", "5", "10"])
    a.set(xlabel="share of cells with an exposed electrode [%]", ylabel="change against healthy [%] (symlog)",
          title="(c) About a third of the cells matches all three (bands: data)")
    a.legend(loc="upper right", bbox_to_anchor=(1, .57))

    a = ax[1, 1]  # (d) effect-to-noise
    r = en[::-1]
    yy = np.arange(len(r))
    a.barh(yy, r, color=[BLUE if k == "v_rms" else "#9ec3ee" for k in r.index], height=.6, label="simulation")
    a.plot([max(DATA_EN[k], .1) for k in r.index], yy, "D", color=INK, mfc="white", ms=5, mew=1.1, label="data (section 10.5)")
    a.set_xscale("log")
    a.set_yticks(yy, [LAB[k] for k in r.index])
    a.set(xlabel="|effect| / noise (log)", title="(d) Random initiation and supply noise: V_rms stays first")
    a.legend(loc="lower right")
    a.grid(axis="y", visible=False)
    fig.text(.01, .004, "Per-cell Manley model of the 126-pair device driven by its own driving power supply (E behind R), measured through the "
             "scope chain and features.extract. (b), (c) no damage data were used to fit;\n(c) bands: M1 and M2 against M3. "
             "(d) data values below 0.1 drawn at 0.1 (the devices disagree in sign).", fontsize=6.2, color=INK)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(Path(out) / "vrms_simulation.png", bbox_inches="tight")


def figure_model(p, out, n_exposed=48):
    """Schematic of the simulated equivalent circuit and of the mechanism, with simulated air-gap voltages and Lissajous graphs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

    BLUE, ORANGE, INK, GREY, PALE = "#2a78d6", "#eb6834", "#52514e", "#e4e3df", "#f4f3ef"
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.8, "axes.edgecolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": GREY,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "legend.frameon": False})
    dev = device(p["C_d"], p["C_g"], p["U_b"], p["spread"])
    dmg = exposed(dev, n_exposed)
    hit = pick(dev, n_exposed)
    ok = next(i for i in np.argsort(np.abs(np.log(dev["Ub"][:] / np.median(dev["Ub"])))) if i not in hit)
    ex = hit[np.argmin(np.abs(np.log(dev["Ub"][hit] / dev["Ub"][ok])))]  # an exposed cell that was like the healthy one
    th, vh, qh, uh = run(dev, p["E"], p["R"], trace=[ok, ex])
    td, vd, qd, ud = run(dmg, p["E"], p["R"], trace=[ok, ex])

    fig = plt.figure(figsize=(8, 9.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, .8, 1.05])
    ax_c, ax_f = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])
    ax_w, ax_q = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])

    # (a) circuit
    a = ax_c
    a.set(xlim=(0, 10), ylim=(0, 4.4), title="(a) Simulated equivalent circuit: the device's driving power supply and 126 discharge cells in parallel")
    a.axis("off")
    line = lambda xs, ys, **k: a.plot(xs, ys, color=k.pop("color", INK), lw=k.pop("lw", 1.2), solid_capstyle="round", **k)  # noqa: E731

    def cap(x, y, c=INK, label=None):
        line([x, x], [y + .32, y + .08], color=c); line([x - .22, x + .22], [y + .08, y + .08], color=c, lw=1.8)
        line([x - .22, x + .22], [y - .08, y - .08], color=c, lw=1.8); line([x, x], [y - .08, y - .32], color=c)
        if label:
            a.text(x - .3, y, label, ha="right", va="center", fontsize=6.6, color=c)

    def cell(x, c, ub, tag):
        line([x, x], [3.7, 3.35], color=c)
        cap(x, 3.03, c, "C_d,i")
        line([x, x], [2.71, 2.45], color=c)
        line([x - .45, x + .45], [2.45, 2.45], color=c); line([x - .45, x + .45], [1.25, 1.25], color=c)
        line([x - .45, x - .45], [2.45, 2.17], color=c); line([x - .45, x - .45], [1.53, 1.25], color=c)
        cap(x - .45, 1.85, c, "C_g,i")
        a.add_patch(Rectangle((x + .2, 1.55), .5, .6, fc="white", ec=c, lw=1.2))
        a.text(x + .45, 1.85, ub, ha="center", va="center", fontsize=6.2, color=c)
        line([x + .45, x + .45], [2.45, 2.15], color=c); line([x + .45, x + .45], [1.55, 1.25], color=c)
        line([x, x], [1.25, .7], color=c)
        a.text(x, .38, tag, ha="center", va="center", fontsize=6.6, color=c)

    a.add_patch(FancyBboxPatch((.2, .55), 2.75, 3.55, boxstyle="round,pad=.05", fc=PALE, ec=GREY, lw=.8))
    a.text(1.57, 4.18, "driving power supply (Thévenin)", ha="center", fontsize=6.8, color=INK)
    a.add_patch(Circle((.85, 2.2), .38, fc="white", ec=INK, lw=1.2))
    a.text(.85, 2.2, "~", ha="center", va="center", fontsize=13, color=INK)
    a.text(1.32, 2.2, f"E sin ωt\n{p['E'] / 1e3:.1f} kV", ha="left", va="center", fontsize=6.6, color=INK)
    line([.85, .85], [2.58, 3.7]); line([.85, 1.35], [3.7, 3.7])
    zx = np.linspace(1.35, 2.45, 13)
    line(zx, 3.7 + .13 * np.array([0, 1, -1] * 4 + [0]))
    a.text(1.9, 3.38, f"R ≈ {p['R'] / 1e3:.0f} kΩ", ha="center", fontsize=6.6, color=INK)
    line([2.45, 9.4], [3.7, 3.7]); line([.85, .85], [1.82, .7]); line([.85, 9.4], [.7, .7])
    a.plot(3.25, 3.7, "o", color=INK, ms=4)
    a.text(3.25, 4.0, "V (HV probe)", ha="center", fontsize=6.6, color=INK)
    a.add_patch(FancyArrowPatch((3.5, 3.55), (4.05, 3.55), arrowstyle="-|>", mutation_scale=8, color=INK, lw=.8))
    a.text(3.78, 3.3, "I", ha="center", fontsize=6.6, color=INK)
    cell(4.75, INK, "±U_b,i", "healthy cell")
    cell(6.85, INK, "±U_b,i", "healthy cell")
    a.text(7.95, 3.0, "· · ·", ha="center", va="center", fontsize=12, color=INK)
    cell(9.1, ORANGE, "±0.83\nU_b,i", "exposed electrode")
    a.text(7.0, 4.05, "126 discharge cells in parallel (Manley); U_b,i spread over the cells\n"
           "a discharging cell holds its air-gap voltage at ±U_b,i", ha="center", va="bottom", fontsize=6.6, color=INK)

    # (b) mechanism
    a = ax_f
    a.set(xlim=(0, 10), ylim=(0, 2), title="(b) Mechanism tested by the simulation, step by step")
    a.axis("off")
    steps = [("Exposed Ag\nelectrode", "γ 0.01 → 0.1", "data: EDS Ag\n1.3–13.9 at.% at damage"),
             ("Lower burning\nvoltage", "U_b × 0.83\n(Townsend, eq. 3b)", "865 → 718 V\n(100 µm air)"),
             ("Earlier discharge\ninitiation", "V_on/V_max (eq. 3c)", "sim −36 %\ndata −31 … −33 %"),
             ("More discharge\ncharge", "ΔQ/V_max", "sim +7.9 %\ndata +5.1 … +6.6 %"),
             ("Power-supply\nvoltage drops", "δlnV = −S δlnY\n(eq. 5)", "S: sim 0.51\ndata 0.52 (M3)"),
             ("V_rms falls", "most visible\nindicator", "sim −2.4 %\ndata −2.3 … −2.8 %")]
    w = 1.45
    for k, (head, sub, num) in enumerate(steps):
        x = .1 + k * 1.65
        a.add_patch(FancyBboxPatch((x, .95), w, .95, boxstyle="round,pad=.03", fc=BLUE if k == 5 else PALE,
                                   ec=BLUE if k == 5 else GREY, lw=.9))
        a.text(x + w / 2, 1.62, head, ha="center", va="center", fontsize=6.9, weight="bold", color="white" if k == 5 else INK)
        a.text(x + w / 2, 1.17, sub, ha="center", va="center", fontsize=6.2, color="white" if k == 5 else INK)
        if num:
            a.text(x + w / 2, .55, num, ha="center", va="center", fontsize=6.3, color=INK)
        if k < 5:
            a.add_patch(FancyArrowPatch((x + w + .02, 1.42), (x + 1.63, 1.42), arrowstyle="-|>", mutation_scale=9, color=INK, lw=1))

    # (c) air-gap voltages over one period
    a = ax_w
    one = th < 1 / F0
    ph = 360 * th[one] * F0
    a.plot(ph, vh[one] / 1e3, color=GREY, lw=2.2, label="applied V, healthy device")
    a.plot(ph, vd[one] / 1e3, color=INK, lw=.8, ls="--", label="applied V, damaged device")
    a.plot(ph, uh[one, 1] / 1e3, color=INK, lw=1.2, label="air-gap voltage, healthy cell")
    a.plot(ph, ud[one, 1] / 1e3, color=ORANGE, lw=1.4, label="same cell, electrode exposed")
    for u, c in ((uh[one, 1], INK), (ud[one, 1], ORANGE)):
        k = np.argmax(u >= .999 * u.max())  # the cell reaches its burning voltage: start of its discharge
        a.axvline(ph[k], color=c, lw=.8, ls=":")
    a.set(xlim=(0, 360), xticks=range(0, 361, 90), xlabel="phase of the power supply [°]", ylabel="voltage [kV]",
          title="(c) Exposed cell: lower burning voltage, earlier discharge")
    a.legend(loc="upper right", fontsize=6.2)

    # (d) Lissajous graphs
    a = ax_q
    for v_, q_, c, lab in ((vh, qh, INK, "healthy device"), (vd, qd, BLUE, f"{n_exposed} cells exposed")):
        a.plot(v_[one] / 1e3, q_[one] * 1e9, color=c, lw=1.2, label=lab)
    a.annotate("discharge starts earlier", xy=(-.05, 21), xytext=(-2.3, 29), fontsize=6.6, color=BLUE,
               arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=.8))
    a.annotate("more charge per cycle", xy=(2.15, 47), xytext=(.1, 45), fontsize=6.6, color=BLUE,
               arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=.8))
    a.set(xlabel="applied voltage [kV]", ylabel="charge [nC]", title="(d) Lissajous graph: earlier discharge, more charge")
    a.legend(loc="upper left")
    fig.text(.01, .004, f"Model fitted to healthy data (section 10.6): C_d {p['C_d'] * 1e12:.1f} pF, C_g {p['C_g'] * 1e12:.1f} pF, "
             f"U_b {p['U_b']:.0f} V, spread {p['spread']:.2f}. (c) one cell traced in both devices; dotted: start of its discharge.",
             fontsize=6.2, color=INK)
    fig.tight_layout(rect=(0, .02, 1, 1))
    fig.savefig(Path(out) / "vrms_simulation_model.png", bbox_inches="tight")


if __name__ == "__main__":
    main(*sys.argv[1:2])
