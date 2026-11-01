"""Case-study figures: equivalent-circuit inverse model and self-referenced anomaly detection.

    python portfolio/make_figures.py "<archive root>" portfolio/figures
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import verify_raw_archive as vra  # noqa: E402

from dbd_diagnostics import circuit_fit, electrical, io, ml, waveform  # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6, "savefig.dpi": 300,
    "legend.frameon": False,
})
FEATURE_SETS = {"thesis 14 features": ml.THESIS_FEATURES, "V_rms only": ["v_rms"],
                "circuit model + frequency": ["C_d", "C_cell", "U_b", "spread", "frequency"]}


def one(job):
    f, meta = job
    r = vra.extract(job)
    if r is None or meta["hours"] < 5:
        return None
    c = circuit_fit.fit(io.load_csv(f))
    r.update(C_d=c.C_d * 1e12, C_cell=c.C_cell * 1e12, U_b=c.U_b, spread=c.spread, r2=c.r2)
    return r


def alarm_rates(d):
    """Mahalanobis distance to each device's own 5-10 h baseline; threshold = 95th percentile of a
    leave-one-session-out null distribution built from the two baseline sessions."""
    rows = []
    for name, fs in FEATURE_SETS.items():
        for dev, g in d.dropna(subset=fs).groupby("device"):
            base = g[g.hours <= 10]

            def fit_score(tr):
                mu = tr[fs].median()
                sd = (tr[fs] - mu).abs().median() * 1.4826 + 1e-12
                lw = LedoitWolf().fit(((tr[fs] - mu) / sd).to_numpy())
                return lambda x: np.sqrt(lw.mahalanobis(((x[fs] - mu) / sd).to_numpy()))

            null = np.concatenate([fit_score(base[base.hours != h])(base[base.hours == h]) for h in base.hours.unique()])
            alarm = pd.Series(fit_score(base)(g) > np.percentile(null, 95), index=g.index).groupby(g.hours).mean()
            rows += [dict(features=name, device=dev, hours=h, alarm=100 * a) for h, a in alarm.items()]
    return pd.DataFrame(rows)


def fig_circuit(d, root, out):
    # (a) one M2 capture: measured Q-V loop and the fitted array model
    f = root / "전류전압 측정" / "2번소스" / "20200718 M2 5시간" / "ALL0012.CSV"
    wf = io.load_csv(f)
    fit = circuit_fit.fit(wf)
    t, v = wf.t, wf.v - wf.v.mean()
    w = waveform.integer_period_window(t, v)
    t, v, i = t[w.slice], v[w.slice], electrical.remove_offset(t[w.slice], wf.i[w.slice])
    q = electrical.charge(t, i, center=True)
    n = int(round(w.period / np.median(np.diff(t))))
    qm = circuit_fit.simulate_charge(np.r_[v[:n], v], fit.C_d, fit.C_g, fit.U_b, fit.spread)[n:]
    qm -= qm.mean()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1, 1.1]})
    a.plot(v[:n], q[:n] * 1e9, color=MUTED, lw=1.0, label="measured")
    a.plot(v[:n], qm[:n] * 1e9, color=BLUE, lw=1.8, label=f"40-cell model (R² = {fit.r2:.3f})")
    a.set(xlabel="applied voltage [V]", ylabel="charge [nC]", title="(a) inverse fit of one capture")
    a.legend(loc="upper left")
    # (b) bias from the sampling interval (10 -> 20 ns), same session: Lissajous vs model
    e = d[d.kind == "endurance"].copy()
    e["c_total_pF"] = e.c_total * 1e12
    X0 = pd.get_dummies(e.device + e.hours.astype(str)).astype(float)
    X0["dt20"] = (e.dt_ns == 20).astype(float)
    X0["la"] = np.log2(e.i_scale / 0.1)
    labels, vals, cols = [], [], []
    for lab, y, col in (("U_b, Lissajous", "u_b", ORANGE), ("U_b, circuit model", "U_b", BLUE),
                        ("C_cell, Lissajous", "c_total_pF", ORANGE), ("C_cell, circuit model", "C_cell", BLUE)):
        A, yy = X0.to_numpy(), e[y].to_numpy()
        bta = np.linalg.lstsq(A, yy, rcond=None)[0]
        r = yy - A @ bta
        cov = r @ r / (len(yy) - A.shape[1]) * np.linalg.pinv(A.T @ A)
        labels.append(lab)
        vals.append(abs(bta[-2] / np.sqrt(cov[-2, -2])))
        cols.append(col)
    y = np.arange(len(labels))[::-1]
    b.barh(y, vals, color=cols, height=0.55)
    b.axvline(2, color=MUTED, lw=0.8, ls=":")
    b.text(2.15, y[0] + 0.32, "|t| = 2", color=MUTED, fontsize=6.5)
    b.set_yticks(y, labels)
    b.set(xlabel="|t| of the 20 ns vs 10 ns effect (same session)", title="(b) bias from the scope's sampling interval")
    b.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "fig_circuit_model.png", bbox_inches="tight")
    plt.close(fig)


def fig_anomaly(rates, out):
    s = []
    for name in FEATURE_SETS:
        x = rates[rates.features == name]
        s.append((name, x[x.device.isin(["m1", "m2"]) & (x.hours >= 15)].alarm.mean(),
                  x[(x.device == "m3") & (x.hours >= 15)].alarm.mean()))
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    y = np.arange(len(s))[::-1]
    ax.barh(y + 0.17, [r[1] for r in s], height=0.32, color=BLUE, label="alarms on damaged M1, M2 (≥ 15 h)")
    ax.barh(y - 0.17, [r[2] for r in s], height=0.32, color=ORANGE, label="false alarms on control M3 (≥ 15 h)")
    for yy, r in zip(y, s):
        ax.text(r[1] + 1, yy + 0.17, f"{r[1]:.0f} %", va="center", color=INK, fontsize=6.8)
        ax.text(r[2] + 1, yy - 0.17, f"{r[2]:.0f} %", va="center", color=INK, fontsize=6.8)
    ax.set_yticks(y, [r[0] for r in s])
    ax.set(xlim=(0, 85), xlabel="captures flagged [%]", title="Self-referenced anomaly detection (each device's own 5–10 h baseline)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=6.5)
    fig.savefig(out / "fig_anomaly_detection.png", bbox_inches="tight")
    plt.close(fig)


def main(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = vra.find_captures(root / "전류전압 측정", root / "전압에 따른 QV 변화" / "I 기준으로 계산")
    with Pool() as p:
        d = pd.DataFrame([r for r in p.map(one, jobs, chunksize=2) if r])
    d = d[d.frequency.between(42e3, 50e3)].replace([np.inf, -np.inf], np.nan)
    fig_circuit(d, root, out)
    rates = alarm_rates(d[d.kind == "endurance"])
    rates.to_csv(out / "anomaly_rates.csv", index=False)
    fig_anomaly(rates, out)
    print(rates.pivot_table(index=["features", "device"], columns="hours", values="alarm").round(0).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
