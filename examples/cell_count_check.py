"""Addendum, not part of the thesis: fewer discharging cells on source 4 (raw archive, 2020-08-08 and 08-13)

    python examples/cell_count_check.py "<archive>/전류전압 측정/4번소스" [docs/img]

Source 4 (`전류전압 측정/4번소스`) is a separate source of the thesis's structure (126 cells) on its own driving power
supply. On 2020-08-08 it was measured discharging in all cells (13:20) and again with its surface covered by Kapton
tape except one cell (14:33). On 2020-08-13 it was measured with the clip on the electrode. The experiment changes
the load on purpose, so it tests step 3 of docs/raw-data-verification.md 10.5 (a smaller load raises the output
voltage of the driving power supply) without the time trend and the shared external effects of the thesis sources,
and it checks the size the 10.6 equivalent circuit gives for that change.

Prints the tables of docs/cell-count-check.md:
  1. captures per condition and their scope settings
  2. feature medians per condition (10 ns captures)
  3. the 10.6 circuit with the discharge stopped in all but one or two cells, against the data
  4. the output resistance R of the circuit that reproduces the measured change, and that circuit's S
Writes cell_count_check.png.
"""
import sys
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vrms_simulation as vs  # noqa: E402
from dbd_diagnostics import electrical, features, io, waveform  # noqa: E402

COND = {"0808M 셀 120개 방전": "all cells", "0808M 셀하나 방전": "one cell",
        "0813M 오리주둥이 클립으로 전극접촉후 방전": "clip on the electrode (08-13)"}
M_CIRCUIT = dict(C_d=34.9e-12, C_g=16.7e-12, U_b=898.3, spread=0.153, R=2.126e5, E=4194.0)  # 10.6, fitted to M1-M3
S_M3 = 0.52  # docs/raw-data-verification.md 10.5, step 4
C_HEALTHY = 12.15e-12  # C_total of the healthy M1-M3 captures (10.6)
FEAT = ["v_rms", "v_1", "v_max", "frequency", "c_total", "c_dielectric", "q_per_v", "von_frac", "u_b", "energy", "i_rms"]
LAB = {"v_rms": "V_rms [V]", "v_1": "V_1, fundamental amplitude [V]", "v_max": "V_max [V]", "frequency": "f [kHz]",
       "c_total": "C_total [pF]", "c_dielectric": "C_d, discharge-phase slope [pF]", "q_per_v": "ΔQ/V_max [pF]",
       "von_frac": "V_on/V_max", "u_b": "U_b [V]", "energy": "E per cycle [µJ]", "i_rms": "I_rms [mA]"}
SCALE = dict(frequency=1e-3, c_total=1e12, c_dielectric=1e12, q_per_v=1e12, energy=1e6, i_rms=1e3)
R_GRID = np.geomspace(1e4, 3e5, 12)


def nfc(s):
    """macOS stores Korean file names decomposed (NFD); compare them in NFC."""
    return unicodedata.normalize("NFC", s)


def fundamental(t, v):
    """Amplitude of the drive-frequency component over the longest whole number of periods."""
    w = waveform.integer_period_window(t, v)
    x, ph = v[w.slice], 2 * np.pi * (t[w.slice] - t[w.start]) / w.period
    return 2 * abs(np.mean((x - x.mean()) * np.exp(-1j * ph)))


def find_captures(root):
    return [(f, COND[nfc(d.name)]) for d in sorted(Path(root).iterdir()) if nfc(d.name) in COND
            for f in sorted(d.glob("ALL*.CSV"))]


def extract(job):
    f, cond = job
    out = dict(cond=cond, file=f.name)
    try:
        wf = io.load_csv(f)
    except ValueError:  # ALL0031 of the one-cell folder: not a two-channel capture
        return out | dict(skip="not a two-channel capture")
    out |= dict(time=wf.meta["Time"][0], dt_ns=round(wf.dt * 1e9), i_scale=float(wf.meta["Vertical Scale"][1]))
    if out["dt_ns"] != 10:  # 20 ns captures and the 0.2 ms overview: other settings (raw-data verification, section 5)
        return out | dict(skip=f"{out['dt_ns']} ns sampling")
    r = features.extract(wf)
    out |= {k: r[k] for k in ("v_rms", "v_max", "frequency", "c_total", "c_dielectric", "u_b", "energy", "i_rms")}
    return out | dict(v_1=fundamental(wf.t, wf.v), q_per_v=r["q_max"] / r["v_max"], von_frac=r["v_on_lissajous"] / r["v_max"])


def few_cells(dev, k):
    """Discharge stopped in all but the k cells nearest the median burning voltage; capacitances unchanged."""
    keep = np.argsort(np.abs(np.log(dev["Ub"] / np.median(dev["Ub"]))))[:k]
    Ub = np.full(len(dev["Ub"]), np.inf)
    Ub[keep] = dev["Ub"][keep]
    return dict(dev, Ub=Ub)


def supply_for(dev, R, f, vmax):
    """Open-circuit voltage E that gives the device the measured V_max with output resistance R."""
    E = M_CIRCUIT["E"]
    for _ in range(5):
        E *= vmax / vs.truth(*vs.run(dev, E=E, R=R, f=f))["v_max"]
    return E


def amplitudes(t, v):
    return dict(v_rms=float(np.sqrt(np.mean(v ** 2))), v_1=fundamental(t, v), v_max=float(np.max(np.abs(v))))


def response(job):
    """One output resistance R: the one-cell change of V_rms, V_1 and V_max, and the circuit's S (3 % heavier load,
    as in vrms_simulation.fit), with E set to the measured V_max of the source with all cells discharging."""
    R, f, vmax = job
    p = M_CIRCUIT
    dev = vs.device(p["C_d"], p["C_g"], p["U_b"], p["spread"])
    E = supply_for(dev, R, f, vmax)
    t, v, Q = vs.run(dev, E=E, R=R, f=f)
    h, base = vs.truth(t, v, Q), amplitudes(t, v)
    one = amplitudes(*vs.run(few_cells(dev, 1), E=E, R=R, f=f)[:2])
    e = vs.truth(*vs.run(dict(dev, Cd=dev["Cd"] * 1.03, Cg=dev["Cg"] * 1.03), E=E, R=R, f=f))
    S = -np.log(e["v_rms"] / h["v_rms"]) / np.log(e["q_per_v"] / h["q_per_v"])
    return dict(R=R, E=E, S=S) | {k: 100 * (one[k] / base[k] - 1) for k in base}


def main(root, out="."):
    pd.set_option("display.width", 200)
    with Pool() as pool:
        d = pd.DataFrame(pool.map(extract, find_captures(root)))

    print("1. captures (10 ns captures are used; the others have other scope settings)")
    for c, g in d.groupby("cond", sort=False):
        u = g[g.skip.isna()]
        print(f"  {c}: {len(g)} files, {len(u)} used, {u.time.min()} - {u.time.max()[-8:]}, current {1e3 * u.i_scale.iloc[0]:.0f} mA/div;"
              f" skipped: {', '.join(f'{n} {k}' for k, n in g.skip.value_counts().items())}")
    u = d[d.skip.isna()]
    off = (u.frequency / u.groupby("cond").frequency.transform("median") - 1).abs() > .2  # period detection failed
    for _, r in u[off].iterrows():
        print(f"  dropped {r.cond} {r.file}: f {r.frequency / 1e3:.1f} kHz, every value per half period")
    u = u[~off]
    med = u.groupby("cond", sort=False)[FEAT].median()
    show = med.T.copy()
    for k, s in SCALE.items():
        show.loc[k] *= s
    a, o = med.loc["all cells"], med.loc["one cell"]
    show["one cell / all cells"] = [f"{100 * (o[k] / a[k] - 1):+.1f} %" for k in FEAT]
    show.index = [LAB[k] for k in FEAT]
    print("\n2. medians (I_rms depends on the current scale, section 5 of the raw-data verification)")
    print(show.to_string(float_format=lambda x: f"{x:.3f}"))
    sd = u.groupby("cond", sort=False)[["v_rms", "frequency"]].std() / med[["v_rms", "frequency"]]
    print("  within-condition SD: " + "; ".join(f"{c} V_rms {100 * r.v_rms:.2f} %, f {100 * r.frequency:.2f} %" for c, r in sd.iterrows()))
    print(f"  V_rms/V_max: all cells {a.v_rms / a.v_max:.3f}, one cell {o.v_rms / o.v_max:.3f} (a sine 0.707); "
          f"V_rms/V_1: {a.v_rms / a.v_1:.3f}, {o.v_rms / o.v_1:.3f}")

    f, vmax = a.frequency, a.v_max
    p = M_CIRCUIT
    dev = vs.device(p["C_d"], p["C_g"], p["U_b"], p["spread"])
    E = supply_for(dev, p["R"], f, vmax)
    t, v, _ = vs.run(dev, E=E, R=p["R"], f=f)
    base = amplitudes(t, v)
    k_day, k_m = o.c_total / a.c_total, o.c_total / C_HEALTHY
    n_left = int(round(126 * o.energy / a.energy))
    cases = {"one cell": (few_cells(dev, 1), f), "two cells": (few_cells(dev, 2), f),
             f"{n_left} cells (energy per cycle {100 * o.energy / a.energy:.0f} % of all cells, as measured)": (few_cells(dev, n_left), f),
             f"one cell, capacitance ×{k_day:.2f} (C_total one cell / all cells)": (dict(few_cells(dev, 1), Cd=dev["Cd"] * k_day, Cg=dev["Cg"] * k_day), f),
             f"one cell, capacitance ×{k_m:.2f} (C_total one cell / M1-M3)": (dict(few_cells(dev, 1), Cd=dev["Cd"] * k_m, Cg=dev["Cg"] * k_m), f),
             f"one cell, at the measured one-cell frequency {o.frequency / 1e3:.2f} kHz": (few_cells(dev, 1), o.frequency)}
    rows = []
    for name, (dv, fc) in cases.items():
        x = amplitudes(*vs.run(dv, E=E, R=p["R"], f=fc)[:2])
        rows.append(dict(case=name) | {k: 100 * (x[k] / base[k] - 1) for k in base})
    rows.append(dict(case="data (one cell / all cells)") | {k: 100 * (o[k] / a[k] - 1) for k in base})
    print(f"\n3. the 10.6 circuit (R {p['R'] / 1e3:.1f} kΩ; E {E:.0f} V for V_max {vmax:.0f} V at {f / 1e3:.2f} kHz), noise-free, change [%]")
    print(pd.DataFrame(rows).set_index("case").round(1).to_string())

    with Pool() as pool:
        scan = pd.DataFrame(pool.map(response, [(R, f, vmax) for R in R_GRID]))
    meas = {k: 100 * (o[k] / a[k] - 1) for k in ("v_rms", "v_1")}
    fits = {}
    for k, y in meas.items():  # the one-cell rise grows monotonically with R
        R = float(np.exp(np.interp(y, scan[k], np.log(scan.R))))
        fits[k] = response((R, f, vmax))
    ref = response((p["R"], f, vmax))
    print("\n4. output resistance R: one-cell change [%] and the circuit's S")
    print(scan.assign(R=scan.R / 1e3, E=scan.E.round()).round(3).rename(columns={"R": "R [kΩ]"}).to_string(index=False))
    for k, r in fits.items():
        print(f"  R reproducing the measured {k} rise ({meas[k]:+.1f} %): {r['R'] / 1e3:.1f} kΩ, S {r['S']:.3f} "
              f"(V_rms {r['v_rms']:+.1f} %, V_1 {r['v_1']:+.1f} %, V_max {r['v_max']:+.1f} %)")
    print(f"  10.6 circuit (R {p['R'] / 1e3:.1f} kΩ): S {ref['S']:.3f}; M3 data: S {S_M3}")
    print(f"  load path for M1, M2 (ΔQ/V_max +6.6, +5.1 %, 10.5 step 5) with these S: "
          + "; ".join(f"S {r['S']:.2f}: {-r['S'] * 6.6:.1f}, {-r['S'] * 5.1:.1f} %" for r in fits.values()) + " (observed -2.9, -2.3 %)")
    reps = [u[u.cond == c].iloc[(u[u.cond == c].v_rms - med.loc[c, "v_rms"]).abs().argsort().iloc[0]] for c in ("all cells", "one cell")]
    figure(root, reps, scan, fits, ref, meas, out)


def figure(root, reps, scan, fits, ref, meas, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, INK, GREY = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.8, "axes.edgecolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": GREY,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "legend.frameon": False})
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    folder = {c: d for d in Path(root).iterdir() for k, c in COND.items() if nfc(d.name) == k}
    for r, col, lab in zip(reps, (BLUE, ORANGE), ("all 126 cells", "one cell (rest under Kapton tape)")):
        wf = io.load_csv(folder[r["cond"]] / r["file"])
        w = waveform.one_period_window(wf.t, wf.v)
        t, v = wf.t[w.slice], wf.v[w.slice] - wf.v[w.slice].mean()
        i = electrical.remove_offset(t, wf.i[w.slice])  # as in features.extract
        ax[0].plot(360 * (t - t[0]) / w.period, v / 1e3, color=col, lw=1.2,
                   label=f"{lab}:\nV_rms {r['v_rms']:.0f} V, f {r['frequency'] / 1e3:.1f} kHz")
        ax[1].plot(v / 1e3, electrical.charge(t, i, center=True) * 1e9, color=col, lw=1.2,
                   label=f"{lab}:\nC_d {r['c_dielectric'] * 1e12:.1f}, C_total {r['c_total'] * 1e12:.1f} pF")
    ax[0].set(xlabel="phase [°]", ylabel="applied voltage [kV]", xlim=(0, 360), xticks=range(0, 361, 90),
              title="(a) Applied voltage, one period (measured)")
    ax[0].legend(loc="lower left")
    ax[1].set(xlabel="applied voltage [kV]", ylabel="charge [nC]", title="(b) Lissajous graph (measured)")
    ax[1].legend(loc="upper left")

    a = ax[2]
    for k, ls, lab in (("v_rms", "-", "V_rms"), ("v_1", "--", "V_1 (fundamental)")):
        a.plot(scan.S, scan[k], ls, color=INK, lw=1.2, label=f"circuit: one-cell rise of {lab}")
        a.axhline(meas[k], color=ORANGE, ls=ls, lw=1, label=f"measured {lab}: {meas[k]:+.1f} %")
        a.plot(fits[k]["S"], fits[k][k], "o", color=ORANGE, ms=6, mec="white", mew=.7)
    a.plot(ref["S"], ref["v_rms"], "s", color=INK, ms=6, mec="white", mew=.7)
    a.annotate(f"10.6 circuit (R {ref['R'] / 1e3:.0f} kΩ):\nS {ref['S']:.2f}, V_rms {ref['v_rms']:+.0f} %", (ref["S"], ref["v_rms"]),
               xytext=(-8, 4), textcoords="offset points", ha="right", va="bottom", fontsize=6.8, color=INK)
    lo, hi = sorted(r["S"] for r in fits.values())
    a.annotate(f"S {lo:.2f}–{hi:.2f}\nreproduces the data", (hi, fits["v_rms"]["v_rms"]), xytext=(10, -22),
               textcoords="offset points", fontsize=6.8, color=ORANGE)
    a.set(xlabel="S of the circuit (output resistance R varied)", ylabel="one cell against all cells [%]", xlim=(0, None),
          title="(c) Equivalent circuit against the data")
    a.set_ylim(0, 1.25 * max(scan.v_rms.max(), scan.v_1.max()))
    a.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(Path(out) / "cell_count_check.png", bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ".")
