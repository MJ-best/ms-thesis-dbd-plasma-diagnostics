"""Matplotlib figures shared by the demo app, the notebook and the examples.

matplotlib is an optional dependency (``pip install dbd-diagnostics[plot]``).
"""
from __future__ import annotations

import numpy as np

from . import electrical, lissajous, waveform
from .io import Waveform


def one_period(wf: Waveform):
    """(t, v, i_offset_free, q_centred, LissajousResult) for the first full period."""
    w = waveform.one_period_window(wf.t, wf.v)
    t, v, i = wf.t[w.slice], wf.v[w.slice], wf.i[w.slice]
    v = v - np.mean(v)
    i = electrical.remove_offset(t, i)
    q = electrical.charge(t, i)
    return t, v, i, q, lissajous.analyse(t, v, q)


def waveform_figure(wf: Waveform, show_gap_voltage: bool = True):
    import matplotlib.pyplot as plt

    t, v, i, q, L = one_period(wf)
    tt = (t - t[0]) * 1e6
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(tt, v / 1e3, color="0.15", lw=1, label="applied voltage v(t)")
    if show_gap_voltage:
        ax.plot(tt, lissajous.gap_voltage(v, q, L.C_dielectric) / 1e3, color="tab:blue", lw=1,
                label="gap voltage u_g = v - Q/C_d")
        for s in (1, -1):
            ax.axhline(s * L.U_b / 1e3, color="tab:blue", ls=":", lw=1)
    ax.set_xlabel("time [µs]")
    ax.set_ylabel("voltage [kV]")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(tt, i * 1e3, color="tab:red", lw=0.6, alpha=0.75)
    ax2.set_ylabel("current [mA]", color="tab:red")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("One period of the capture")
    fig.tight_layout()
    return fig


def lissajous_figure(wf: Waveform):
    import matplotlib.pyplot as plt

    t, v, i, q, L = one_period(wf)
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.plot(v / 1e3, q * 1e9, color="0.3", lw=0.8, label="Q–V")
    # fitted segments: rising branch in (V, Q); falling branch was fitted mirrored (-V, -Q)
    for k, (ft, fd) in enumerate(zip(L.fit_total, L.fit_dielectric)):
        sgn = 1 if k == 0 else -1
        v_on = sgn * L.V_on
        seg_t = np.linspace(-sgn * L.V_max, v_on, 20)
        seg_d = np.linspace(v_on, sgn * L.V_max, 20)
        ax.plot(seg_t / 1e3, (ft.slope * seg_t + sgn * ft.intercept) * 1e9, "--", color="tab:green", lw=1.2,
                label=f"no discharge: C_total = {L.C_total * 1e12:.1f} pF" if k == 0 else None)
        ax.plot(seg_d / 1e3, (fd.slope * seg_d + sgn * fd.intercept) * 1e9, "--", color="tab:orange", lw=1.2,
                label=f"discharge: C_dielectric = {L.C_dielectric * 1e12:.1f} pF" if k == 0 else None)
    ax.axvline(L.U_b / 1e3, color="tab:blue", ls=":", label=f"burning voltage U_b = {L.U_b:.0f} V")
    ax.axvline(L.V_breakdown_thesis / 1e3, color="tab:red", ls=":",
               label=f"thesis Eq.15 = {L.V_breakdown_thesis:.0f} V (≈ V_max)")
    lim = 1.4 * np.max(np.abs(q)) * 1e9
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("voltage [kV]")
    ax.set_ylabel("charge [nC]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("Lissajous figure")
    fig.tight_layout()
    return fig
