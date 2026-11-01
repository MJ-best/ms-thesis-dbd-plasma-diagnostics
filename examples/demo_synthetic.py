"""Synthetic end-to-end demo (no measurement data needed).

    python examples/demo_synthetic.py            # prints results, writes docs/img/*.png

Generates a 126-cell surface-DBD capture with the thesis operating point
(46.5 kHz, ~2.2 kV peak, C_d ~ 34 pF, U_b = 0.9 kV) through an emulated
100 MS/s, 8-bit oscilloscope, then runs the full analysis and compares the
thesis breakdown-voltage formula with the burning voltage.
"""
from pathlib import Path

import numpy as np

from dbd_diagnostics import electrical, features, lissajous, simulate, waveform

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"


def main():
    wf = simulate.cells(seed=0, periods=3.3)
    r = features.extract(wf)
    print("--- features (averaged over full periods) ---")
    for k in ("frequency", "v_rms", "i_rms", "energy", "power", "c_total", "c_dielectric", "c_gas",
              "q_max", "v_max", "v_min", "v_breakdown", "u_b", "u_b_energy", "en_td", "te", "te_corrected",
              "n_discharges"):
        print(f"{k:14s} {r[k]:.4g}")
    print(f"\nthesis V_breakdown = {r['v_breakdown']:.0f} V  vs  V_max = {r['v_max']:.0f} V"
          f"  vs  burning voltage U_b = {r['u_b']:.0f} V (model input 900 V, cells fire 0.4-1.0 U_b)")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    # figure: ideal Manley model (exact parallelogram, true U_b = 900 V)
    wf = simulate.ideal()
    w = waveform.one_period_window(wf.t, wf.v)
    t, v, i = wf.t[w.slice], wf.v[w.slice], wf.i[w.slice]
    v = v - v.mean()
    i = electrical.remove_offset(t, i)
    q = electrical.charge(t, i)
    L = lissajous.analyse(t, v, q)
    ug = lissajous.gap_voltage(v, q, L.C_dielectric)

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a = ax[0]
    a.plot((t - t[0]) * 1e6, v / 1e3, color="0.2", lw=1, label="applied voltage v(t)")
    a.plot((t - t[0]) * 1e6, ug / 1e3, color="tab:blue", lw=1, label="gap voltage u_g = v - Q/C_d")
    a2 = a.twinx()
    a2.plot((t - t[0]) * 1e6, i * 1e3, color="tab:red", lw=0.6, alpha=0.7)
    a2.set_ylabel("current [mA]", color="tab:red")
    a.axhline(L.U_b / 1e3, color="tab:blue", ls=":", lw=1)
    a.axhline(-L.U_b / 1e3, color="tab:blue", ls=":", lw=1)
    a.set_xlabel("time [us]")
    a.set_ylabel("voltage [kV]")
    a.legend(loc="lower left", fontsize=8)
    a.set_title("Ideal equivalent circuit: v, u_g and i (true U_b = 900 V)")
    b = ax[1]
    b.plot(v / 1e3, q * 1e9, color="0.3", lw=0.8)
    vv = np.linspace(-L.V_max, L.V_max, 50)
    b.plot(vv / 1e3, (L.C_dielectric * (vv - L.U_b)) * 1e9, color="tab:orange", ls="--", lw=1,
           label=f"discharge line, slope C_d = {L.C_dielectric * 1e12:.1f} pF")
    b.axvline(L.U_b / 1e3, color="tab:blue", ls=":", lw=1.2, label=f"burning voltage U_b = {L.U_b:.0f} V")
    b.axvline(L.V_breakdown_thesis / 1e3, color="tab:red", ls=":", lw=1.2,
              label=f"thesis Eq.15 V_min+Q/C_d = {L.V_breakdown_thesis:.0f} V (~V_max)")
    b.set_xlabel("voltage [kV]")
    b.set_ylabel("charge [nC]")
    b.set_ylim(-1.3 * abs(q).max() * 1e9, 1.3 * abs(q).max() * 1e9)
    b.legend(loc="upper left", fontsize=8)
    b.set_title("Lissajous (Q-V) figure")
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_waveform_lissajous.png", dpi=130)
    print("wrote", OUT / "synthetic_waveform_lissajous.png")


if __name__ == "__main__":
    main()
