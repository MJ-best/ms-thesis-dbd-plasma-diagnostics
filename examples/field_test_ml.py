"""Damage model for devices with the thesis cell structure: lab devices for learning, ATI as the field test.

    python examples/field_test_ml.py "<archive>/M!M2M3실험데이터종합" "<aging archive>/μ-DBD (large) Aging 데이터 모음" [docs/img]

Lab: M1 and M2 (erosion confirmed by SEM/EDS, labelled damaged from 15 h) and the undamaged control M3,
5-60 h, all on one supply. Field test: ATI #1-3 (same cell structure, larger area, another supply), run
until the discharge stopped. The field data carry no damage label and never change the lab labels.

Prints:
  1. data per device
  2. raw features: lab vs field separability, and lab-trained classifiers applied to the field
  3. supervised classifier on self-referenced features, leave one lab device out
  4. health model (self-referenced log ratios, pooled normal variation), leave one device out
  5. more data: the healthy pool with and without the field devices
Writes field_test_ml.png.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ati_aging_test as ati  # noqa: E402
import verify_raw_archive as vra  # noqa: E402

from dbd_diagnostics import circuit_fit, features, io  # noqa: E402

RAW = ["f", "V", "I", "E", "C_cell", "zeta"]  # absolute features, as in the thesis classifiers
REL = ["C_cell", "V", "f"]  # measured reliably on both rigs; used as log ratios to the device's own start
R = ["r_" + k for k in REL]
NO_EARLY_LIFE = {"ATI #1"}  # only its last 6 days were kept, so its first sessions are no healthy reference
ALARM_Q = 95  # alarm threshold: 95th percentile of the health score over the healthy pool


def load(path, field):
    stamp, dt = ati.header(path)
    try:
        wf = io.load_csv(path)
    except ValueError:  # CH2 saved with unit "V"
        a = np.genfromtxt(path, delimiter=",", skip_header=18, usecols=(0, 1, 3), invalid_raise=False)
        wf = io.Waveform(t=a[:, 0], v=a[:, 1], i=a[:, 2], meta={})
    t, i = wf.t, wf.i
    if np.any(np.diff(t) <= 0):  # time column saved with too few digits: rebuild it from the header
        t = t[0] + dt * np.arange(len(t))
    if field and np.corrcoef(i - i.mean(), np.gradient(wf.v))[0, 1] < 0:  # probe reversed on some ATI sessions
        i = -i
    return io.Waveform(t=t, v=wf.v, i=i, meta=wf.meta), stamp, dt


def extract(job):
    path, meta = job
    wf, stamp, dt = load(path, meta["domain"] == "field")
    out = dict(meta, file=Path(path).name, time=stamp, dt_ns=round(dt * 1e9))
    if dt > 5e-8:  # overview capture
        return out
    try:
        f = features.extract(wf)
        c = circuit_fit.fit(wf)
    except ValueError:  # record shorter than one period (1-2 ns captures)
        return out
    out.update(f=f["frequency"] / 1e3, V=f["v_rms"], I=f["i_rms"] * 1e3, E=f["energy"] * 1e6,
               zeta=f["c_dielectric"] * 1e12, C_cell=c.C_cell * 1e12, r2=c.r2)
    return out


def find_jobs(lab_root, field_root):
    lab_root = Path(lab_root)
    jobs = []
    for f, m in vra.find_captures(lab_root / "전류전압 측정", lab_root / "전압에 따른 QV 변화" / "I 기준으로 계산"):
        if m["kind"] == "endurance" and m["hours"] >= 5:
            jobs.append((f, dict(domain="lab", device=m["device"].upper(), session=f"{m['hours']} h",
                                 hours=m["hours"], note="")))
    for r in ati.find_captures(field_root):
        jobs.append((r["path"], dict(domain="field", device=r["device"], session=r["session"], hours=np.nan,
                                     note=r["note"])))
    return jobs


def sessions(d):
    s = (d.groupby(["domain", "device", "session"])
           .agg(date=("date", "min"), hours=("hours", "first"), note=("note", "first"), n=("file", "size"),
                **{k: (k, "median") for k in RAW}).reset_index().sort_values(["device", "date"]).reset_index(drop=True))
    s["clean"] = s.note.isin(["", "after IPA cleaning"])
    s["label"] = np.where(s.domain == "field", np.nan, ((s.device != "M3") & (s.hours >= 15)).astype(float))
    s["days_left"] = [(pd.Timestamp(ati.END[dev]) + pd.Timedelta("1D") - t).total_seconds() / 86400
                      if dom == "field" else np.nan for dom, dev, t in zip(s.domain, s.device, s.date)]
    # the device's own reference: lab 5-10 h, field the first three clean sessions
    s["baseline"] = False
    for dev, g in s.groupby("device"):
        s.loc[g[g.hours <= 10].index if g.domain.iloc[0] == "lab" else g[g.clean].index[:3], "baseline"] = True
    for k in REL:  # log ratio to the reference; a reference session is compared with the other reference sessions
        for dev, g in s.groupby("device"):
            b = g[g.baseline][k]
            s.loc[g.index, "r_" + k] = [np.log(x / b.drop(i, errors="ignore").median()) for i, x in g[k].items()]
    return s


def healthy(s, exclude=NO_EARLY_LIFE):
    return s[(s.label == 0) | ((s.domain == "field") & s.baseline & ~s.device.isin(exclude))]


def health_model(pool):
    """Mahalanobis distance of the log-ratio vector, covariance of the healthy pool (centred on 'no change')."""
    lw = LedoitWolf(assume_centered=True).fit(pool[R].to_numpy())
    score = lambda x: np.sqrt(lw.mahalanobis(x[R].to_numpy()))  # noqa: E731
    return score, np.percentile(score(pool), ALARM_Q), lw.covariance_


def leave_one_device_out(s, domains, exclude=NO_EARLY_LIFE):
    out = []
    for dev, g in s.groupby("device"):
        pool = healthy(s[(s.device != dev) & s.domain.isin(domains)], exclude)
        score, thr, _ = health_model(pool)
        out.append(g.assign(D=score(g), thr=thr, n_pool=len(pool)))
    r = pd.concat(out)
    r["alarm"] = r.D > r.thr
    r["persistent"] = False
    for _, g in r[r.clean & ~r.baseline].groupby("device"):  # two consecutive regular sessions above threshold
        a = g.alarm
        r.loc[g.index, "persistent"] = a & (a.shift(1, fill_value=False) | a.shift(-1, fill_value=False))
    return r


def summary(r):
    lab = r[(r.domain == "lab") & ~r.baseline]
    dmg, ctl = lab[lab.label == 1], lab[lab.device == "M3"]
    out = {"damaged sessions flagged": f"{dmg.alarm.sum()}/{len(dmg)}", "control M3 false alarms": f"{ctl.alarm.sum()}/{len(ctl)}",
           "control M3 persistent": f"{ctl.persistent.sum()}/{len(ctl)}"}
    for dev in ("ATI #2", "ATI #3"):
        g = r[(r.device == dev) & r.clean & ~r.baseline]
        first = g[g.persistent].days_left.max()
        out[f"{dev} alarms"] = f"{g.alarm.sum()}/{len(g)}"
        out[f"{dev} first persistent"] = "none" if np.isnan(first) else f"{first:.0f} d before stop"
    return out


def report(d, s):
    pd.set_option("display.width", 200)
    print("1. data (detail captures, sessions)")
    print(s.groupby(["domain", "device"]).agg(captures=("n", "sum"), sessions=("n", "size")).to_string())

    print("\n2. raw features")
    y = (d.domain == "field").astype(int)
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), d[RAW], y, cv=5,
                          method="predict_proba")[:, 1]
    print(f"  lab vs field, 5-fold AUC = {roc_auc_score(y, p):.3f}")
    print("  " + "  ".join(f"{k}: lab {d[d.domain == 'lab'][k].median():.4g}, field {d[d.domain == 'field'][k].median():.4g}"
                           for k in RAW))
    lab, fld = d[d.domain == "lab"], d[d.domain == "field"].merge(s[["device", "session", "baseline"]])
    for name, m in (("RF", RandomForestClassifier(500, random_state=0, n_jobs=-1)), ("LR", LogisticRegression(max_iter=5000))):
        est = make_pipeline(StandardScaler(), m).fit(lab[RAW], lab.label)
        call = pd.Series(est.predict(fld[RAW]), index=fld.index)
        print(f"  {name} trained on lab, captures called damaged: " +
              ", ".join(f"{dev} {100 * call[g.index].mean():.0f} % (own first sessions {100 * call[g[g.baseline].index].mean():.0f} %)"
                        for dev, g in fld.groupby("device")))

    print("\n3. logistic regression on the log ratios, leave one lab device out (sessions called damaged)")
    ls = s[s.domain == "lab"]
    for dev in ("M1", "M2", "M3"):
        tr, te = ls[ls.device != dev], ls[ls.device == dev]
        est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(tr[R], tr.label)
        print(f"  test {dev}: {est.predict(te[R]).mean():.0%} called damaged, truth {te.label.mean():.0%} damaged")

    print("\n4. health model, leave one device out, pool = healthy sessions of the other lab and field devices")
    r = leave_one_device_out(s, ["lab", "field"])
    for dev, g in r.groupby("device"):
        tag = (lambda x: f"{x.hours:g} h") if g.domain.iloc[0] == "lab" else (lambda x: f"{x.days_left:.0f} d")
        print(f"  {dev} (pool {g.n_pool.iloc[0]} sessions, threshold {g.thr.iloc[0]:.2f}): " +
              " ".join(f"{'(' if x.baseline else ''}{tag(x)} {x.D:.1f}{')' if x.baseline else ''}"
                       f"{'!' if x.persistent else '*' if x.alarm else ''}{'' if x.clean else '?'}" for x in g.itertuples()))
    print("  ( ) own reference sessions; * above threshold; ! two consecutive regular sessions above; ? noted session")
    print("  " + "; ".join(f"{k} {v}" for k, v in summary(r).items()))

    print("\n5. more data: the same model with the healthy pool from the lab only")
    r_lab = leave_one_device_out(s, ["lab"])
    print("  lab only:     " + "; ".join(f"{k} {v}" for k, v in summary(r_lab).items()))
    print("  lab + field:  " + "; ".join(f"{k} {v}" for k, v in summary(r).items()))
    r_all = leave_one_device_out(s, ["lab", "field"], exclude=set())
    print("  lab + field, ATI #1 first sessions also taken as healthy:  " +
          "; ".join(f"{k} {v}" for k, v in summary(r_all).items()))
    print("  control M3 against pools of growing size:")
    steps = pool_steps(s)
    for name, n, fa in steps:
        print(f"    {name:34s} {n:2d} sessions  false alarms {fa}")
    return r, steps


def pool_steps(s):
    m3 = s[(s.device == "M3") & ~s.baseline]
    pool = healthy(s[s.device.isin(["M1", "M2"])])
    rows = []
    for name, add in (("M1, M2 reference sessions", None), ("+ ATI #2 first sessions", "ATI #2"),
                      ("+ ATI #3 first sessions", "ATI #3")):
        if add:
            pool = pd.concat([pool, healthy(s[s.device == add])])
        score, thr, _ = health_model(pool)
        rows.append((name, len(pool), f"{(score(m3) > thr).sum()}/{len(m3)}"))
    return rows


def figure(d, s, r, steps, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, AQUA, YELLOW, INK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#52514e"
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.8, "axes.edgecolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": "#e4e3df",
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "legend.frameon": False})
    fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.6))

    a = ax[0, 0]  # (a) raw features
    groups = (("lab, healthy (M1/M2 5-10 h, M3)", (d.domain == "lab") & (d.label == 0), BLUE),
              ("lab, damaged (M1/M2 ≥ 15 h)", d.label == 1, ORANGE), ("field ATI #2", d.device == "ATI #2", AQUA),
              ("field ATI #3", d.device == "ATI #3", YELLOW))
    for lab, m, c in groups:
        a.plot(d[m].f, d[m].C_cell, "o", ms=2.6, color=c, alpha=.6, mec="none", label=lab)
    a.set_yscale("log")
    a.set_yticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
    a.minorticks_off()
    a.set(xlabel="drive frequency [kHz]", ylabel="C_cell [pF]", title="(a) raw features separate the rigs, not the damage")
    a.legend(loc="center", bbox_to_anchor=(0.6, 0.45), markerscale=1.8)

    a = ax[0, 1]  # (b) self-referenced plane
    _, thr, cov = health_model(healthy(s))
    ang = np.linspace(0, 2 * np.pi, 200)
    L = np.linalg.cholesky(cov[:2, :2])
    e = thr * L @ np.vstack([np.cos(ang), np.sin(ang)])
    pct = lambda x: 100 * (np.exp(x) - 1)  # noqa: E731
    a.fill(pct(e[1]), pct(e[0]), color="#e4e3df", alpha=.7, lw=0)
    a.plot(pct(e[1]), pct(e[0]), color=INK, lw=.8, ls=":")
    sg = (("healthy", (s.domain == "lab") & (s.label == 0) & ~s.baseline, BLUE), ("damaged", s.label == 1, ORANGE),
          ("ATI #2", (s.device == "ATI #2") & s.clean & ~s.baseline, AQUA), ("ATI #3", (s.device == "ATI #3") & s.clean & ~s.baseline, YELLOW))
    for lab, m, c in sg:
        a.plot(pct(s[m].r_V), pct(s[m].r_C_cell), "o", ms=4.5, color=c, mec="white", mew=.7)
    a.text(38, 104, "shaded: normal range\nof the healthy pool", ha="right", va="top", fontsize=6.5, color=INK)
    a.set(xlabel="V_rms vs own start [%]", ylabel="C_cell vs own start [%]",
          title="(b) change from each device's own start, per session")

    a = ax[1, 0]  # (c) field test timeline
    for dev, c in (("ATI #2", AQUA), ("ATI #3", YELLOW)):
        g = r[(r.device == dev)]
        reg, odd = g[g.clean], g[~g.clean]
        a.plot(reg.days_left, reg.D / reg.thr, "-", color=c, lw=1.1, alpha=.6)
        a.plot(reg.days_left, reg.D / reg.thr, "o", color=c, ms=4.5, mec="white", mew=.7, label=dev)
        a.plot(odd.days_left, odd.D / odd.thr, "o", mfc="white", mec=c, ms=4.5, mew=1.0)
        p = reg[reg.persistent]
        a.plot(p.days_left, p.D / p.thr, "o", mfc="none", mec=INK, ms=8, mew=.8)
    a.axhline(1, color=INK, lw=.8, ls=":")
    first = r[(r.device == "ATI #3") & r.persistent].days_left.max()
    a.annotate(f"persistent alarm\n{first:.0f} days before the stop", xy=(first, 2.05), xytext=(245, 2.2),
               fontsize=6.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=.6))
    a.set_xlim(255, -8)
    a.set(xlabel="days before the discharge stopped", ylabel="health score / alarm threshold",
          title="(c) field test: the lab-calibrated model on ATI")
    a.legend(loc="upper left")

    a = ax[1, 1]  # (d) more data
    names = [x[0] for x in steps]
    fa = [int(x[2].split("/")[0]) / int(x[2].split("/")[1]) * 100 for x in steps]
    y = np.arange(len(steps))[::-1]
    a.barh(y, fa, color=BLUE, height=.55)
    for yy, v, st in zip(y, fa, steps):
        a.text(v + 2, yy, f"{st[2]} sessions  (pool {st[1]})", va="center", fontsize=6.8, color="#0b0b0b")
    a.set_yticks(y, names)
    a.set(xlim=(0, 140), xlabel="false alarms on the undamaged control M3 [%]",
          title="(d) more healthy devices in the pool, fewer false alarms")
    a.grid(axis="y", visible=False)
    fig.text(0.01, 0.005, "Lab: M1-M3 (labels from SEM/EDS). Field test: ATI #2, #3 (no damage labels; ATI #1 has no early-life "
             "data).\n(c) hollow: sessions noted as distorted / odd / 13.5 V input; ring: alarm in two consecutive regular sessions.",
             fontsize=6.2, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(Path(out) / "field_test_ml.png", bbox_inches="tight")


def main(lab_root, field_root, out="."):
    with Pool() as p:
        d = pd.DataFrame(p.map(extract, find_jobs(lab_root, field_root), chunksize=2))
    d = d.dropna(subset=RAW)
    d = d[(d.r2 > 0.9) & ((d.domain == "lab") & d.f.between(42, 50) | (d.domain == "field") & d.f.between(100, 115))].copy()
    d["date"] = pd.to_datetime(d.time, format="%d-%b-%y %H:%M:%S")
    d["label"] = np.where(d.domain == "field", np.nan, ((d.device != "M3") & (d.hours >= 15)).astype(float))
    s = sessions(d)
    r, steps = report(d, s)
    figure(d, s, r, steps, out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
