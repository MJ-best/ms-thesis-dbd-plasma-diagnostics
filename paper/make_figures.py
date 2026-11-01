"""Figures of the manuscript, regenerated from the raw-data archive (not distributed).

    python paper/make_figures.py "<archive root>" paper/figures

<archive root> is the folder that contains `전류전압 측정`, `전압에 따른 QV 변화` and `현미경사진`.
"""
import sys
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.base import clone  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import verify_raw_archive as vra  # noqa: E402

from dbd_diagnostics import ml  # noqa: E402

DEV = {"m1": ("M1", "#2a78d6", "o"), "m2": ("M2", "#eb6834", "s"), "m3": ("M3 (control)", "#1baf7a", "^")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 5, "savefig.dpi": 300,
    "legend.frameon": False,
})


def nfc(s):
    return unicodedata.normalize("NFC", s)


def load(root):
    jobs = vra.find_captures(root / "전류전압 측정", root / "전압에 따른 QV 변화" / "I 기준으로 계산")
    with Pool() as p:
        rows = [r for r in p.map(vra.extract, jobs, chunksize=4) if r]
    d = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    d["f_kHz"] = d.frequency / 1e3
    d["seq"] = d.file.str[3:7].astype(int)
    return d[d.f_kHz.between(42, 50)]  # two captures with a failed period detection


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_micrographs(root, out):
    micro = root / "현미경사진"
    picks = [("M2, 10 h", "2번", "10h", "Snapshot000005.jpg"), ("M2, 15 h", "2번", "15h", "Snapshot000005.jpg"),
             ("M2, 40 h", "2번", "40시간", "Snapshot000000.jpg"), ("M1, \"5 h\" (before 10 h run)", "1번", "5h", "Snapshot000008.jpg"),
             ("M1, 25 h", "1번", "25시간", "오른쪽 끝 결국 터짐.jpg"), ("M3 (control), 40 h", "3번", "40시간", "Snapshot000013.jpg")]
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 3.9))
    for ax, (title, dev, tag, fname) in zip(axs.flat, picks):
        folder = next(p for p in (micro / dev).iterdir() if nfc(p.name).endswith("_" + tag))
        f = next(p for p in folder.iterdir() if nfc(p.name) == fname)
        ax.imshow(Image.open(f))
        ax.set_title(title, color=INK)
        ax.axis("off")
    fig.tight_layout()
    save(fig, out, "fig1_micrographs")


def fig_session_effect(d, out):
    e = d[(d.kind == "endurance") & (d.hours >= 5)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    for dev, (lab, col, mk) in DEV.items():
        g = e[e.device == dev].groupby("hours").v_rms.agg(["mean", "std"])
        a.errorbar(g.index, g["mean"], yerr=g["std"], color=col, marker=mk, capsize=2, label=lab)
        s = e[(e.device == dev) & (e.hours == 15)].sort_values("seq")
        b.plot(s.seq, s.v_rms, color=col, marker=mk, label=lab)
    a.axvspan(12.5, 41, color=GRID, alpha=0.5, lw=0)
    a.text(13.2, 1540, "M1 (to 25 h), M2 damaged", color=MUTED)
    a.set(xlabel="operating time [h]", ylabel="V_rms [V]  (session mean ± SD)", title="(a) all devices drop together at 15 h")
    a.legend(loc="lower left")
    b.set(xlabel="capture number in the 15 h session (~4 min)", ylabel="V_rms [V]",
          title="(b) discrete voltage levels within one session")
    save(fig, out, "fig2_session_effect")


def fig_classifier_on_control(d, out):
    e = d[(d.kind == "endurance") & (d.hours >= 5)].dropna(subset=ml.THESIS_FEATURES).copy()
    e["label"] = np.where(e.device == "m3", 0, (e.hours >= 15).astype(int))
    tr = e[e.device == "m2"]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for name, ls in (("RF", "-"), ("LR", "--")):
        est = clone(ml.thesis_models()[name]).fit(tr[ml.THESIS_FEATURES], tr.label)
        for dev in ("m1", "m3"):
            lab, col, mk = DEV[dev]
            g = e[e.device == dev]
            frac = pd.Series(est.predict(g[ml.THESIS_FEATURES]), index=g.index).groupby(g.hours).mean() * 100
            ax.plot(frac.index, frac.values, ls=ls, color=col, marker=mk, label=f"{lab}, {name}")
    ax.axvline(12.5, color=MUTED, lw=0.8)
    ax.set(xlabel="operating time [h]", ylabel="captures classified 'degraded' [%]", ylim=(-5, 105),
           title="Thesis models trained on M2")
    ax.legend(loc="lower right", fontsize=6.5)
    save(fig, out, "fig3_classifier_on_control")


def fig_scope_settings(d, out):
    e = d[(d.kind == "endurance") & (d.hours >= 5)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for (dev, h), g in e.groupby(["device", "hours"]):
        lab, col, mk = DEV[dev]
        m = g.groupby("i_scale").i_mA.mean()
        if len(m) > 1:
            a.plot(m.index, m.values, color=col, marker=mk, alpha=0.9)
        n = g.groupby("dt_ns").n_discharges.mean()
        if len(n) > 1:
            b.plot(n.index, n.values, color=col, marker=mk, alpha=0.6)
    for dev, (lab, col, mk) in DEV.items():
        a.plot([], [], color=col, marker=mk, label=lab)
    a.set_xscale("log", base=2)
    a.set_xticks([0.04, 0.1, 0.2], ["0.04", "0.1", "0.2"])
    a.set_xlim(0.033, 0.24)
    a.set(xlabel="current channel [A/div]", ylabel="I_rms [mA]", title="(a) same session, different A/div")
    a.legend(loc="upper left")
    b.set_xticks([10, 20])
    b.set(xlim=(7, 23), xlabel="sampling interval [ns]", ylabel="pulses per half cycle",
          title="(b) same session, different sampling")
    save(fig, out, "fig4_scope_settings")


def fig_sweep(d, out):
    sw = d[d.hours == 20]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for dev, (lab, col, mk) in DEV.items():
        g = sw[sw.device == dev].groupby("supply_V")[["v_max", "v_breakdown", "u_b"]].mean()
        ax.plot(g.v_max, g.v_breakdown, color=col, marker=mk, label=f"{lab}: thesis V_b")
        ax.plot(g.v_max, g.u_b, color=col, marker=mk, ls="--", mfc="white", label=f"{lab}: U_b")
    x = np.array([2150, 2720])
    ax.plot(x, x, color=MUTED, lw=0.9, ls=":", label="identity V_b = V_max")
    ax.set(xlabel="V_max [V]  (supply input 8.5–14 V)", ylabel="voltage [V]", title="Supply-voltage sweep at 20 h")
    ax.legend(loc="center", bbox_to_anchor=(0.62, 0.45), fontsize=6.3)
    save(fig, out, "fig5_supply_sweep")


def fig_supply_load(d, out):
    e = d[(d.kind == "endurance") & (d.hours >= 5)]
    sm = e.groupby(["device", "hours"])[["v_rms", "f_kHz", "c_d_pF"]].mean().reset_index()
    base = sm[sm.hours <= 10].groupby("device")[["v_rms", "f_kHz"]].mean()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for dev, (lab, col, mk) in DEV.items():
        g = sm[sm.device == dev]
        a.scatter(g.c_d_pF, g.f_kHz, color=col, marker=mk, s=22, label=f"{lab} sessions", zorder=3)
        dv = g.v_rms - base.loc[dev, "v_rms"]
        df_ = (g.f_kHz - base.loc[dev, "f_kHz"]) * 1e3
        b.scatter(dv, df_, color=col, marker=mk, s=22, label=lab, zorder=3)
        for h, x, y in zip(g.hours, dv, df_):
            if h in (15, 45, 50, 60):
                b.annotate(f"{h} h", (x, y), xytext=(3, 3), textcoords="offset points", color=MUTED, fontsize=6.5)
    sw = d[d.hours == 20].groupby(["device", "supply_V"])[["v_rms", "f_kHz", "c_d_pF"]].mean().reset_index()
    a.scatter(sw.c_d_pF, sw.f_kHz, facecolors="none", edgecolors=INK, s=22, label="supply sweep (20 h)", zorder=2)
    a.set(xlabel="effective dielectric capacitance ζ [pF]", ylabel="operating frequency [kHz]",
          title="(a) the supply frequency follows the load")
    a.legend(loc="upper right")
    k = np.mean([np.polyfit(g.v_rms, g.f_kHz, 1)[0] for _, g in sw.groupby("device")]) * 1e3
    xs = np.array([-60, 60])
    b.plot(xs, k * xs, color=MUTED, lw=1, ls=":")
    b.text(-150, -1150, "supply-input direction\n(sweep slope)", color=MUTED, fontsize=6.5)
    b.axhline(0, color=MUTED, lw=0.6)
    b.axvline(0, color=MUTED, lw=0.6)
    b.set(xlabel="ΔV_rms from own 5–10 h baseline [V]", ylabel="Δf [Hz]", title="(b) session changes in the (V, f) plane")
    save(fig, out, "fig6_supply_load")


def main(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d = load(root)
    fig_micrographs(root, out)
    fig_session_effect(d, out)
    fig_classifier_on_control(d, out)
    fig_scope_settings(d, out)
    fig_sweep(d, out)
    fig_supply_load(d, out)
    print("figures written to", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
