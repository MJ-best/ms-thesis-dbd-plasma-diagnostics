"""Do the physics assumptions, the statistics and the ML of the thesis point the same way? (raw archive, M1-M3)

    python examples/consistency_check.py "<archive>/전류전압 측정" "<archive>/전압에 따른 QV 변화/I 기준으로 계산" [docs/img]

Uses the captures of examples/verify_raw_archive.py: the endurance sessions from 5 h and the 20 h supply
sweep. M3 was discharged alongside M1 and M2 under the same conditions and was not damaged, so it is the
healthy comparison device of every session.

Prints the tables of section 10 of docs/raw-data-verification.md:
  1. physics: equivalent circuit, Paschen / Townsend, partial surface discharging, pulse counting
  2. damage effect per feature against M3 (session-level difference-in-differences)
  3. ML: the thesis protocol, M3 in training (leave one session out), features relative to M3
  4. ML on single periods, two from every capture: shuffled at random, a session held out, a device held out
  5. why V_rms: power supply-load coupling, the damage chain and its predicted size, effect-to-noise, permutation and
     bootstrap checks
Writes consistency_check.png and vrms_mechanism.png.
"""
import itertools
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_raw_archive as vra  # noqa: E402

from dbd_diagnostics import features, io, plots  # noqa: E402
from dbd_diagnostics.ml import THESIS_FEATURES as F  # noqa: E402

# features whose value does not depend on the scope's A/div or sampling period (raw-data verification, sec. 5)
ROBUST = ["v_rms", "frequency", "energy", "c_dielectric", "c_gas", "c_total", "q_max", "u_b", "v_on_lissajous"]
DID = {"v_rms": "V_rms", "v_on_lissajous": "V_on (discharge initiation)", "q_max": "ΔQ", "c_total": "C_total",
       "c_dielectric": "C_d (discharge slope)", "energy": "E per cycle", "i_rms": "I_rms", "n_discharges": "pulses"}
S = lambda m: make_pipeline(StandardScaler(), m)  # noqa: E731
MODELS = {  # the six thesis models (2023), then newer tabular methods
    "LR": S(LogisticRegression(max_iter=5000)), "DT": DecisionTreeClassifier(random_state=0),
    "RF": RandomForestClassifier(1000, random_state=0, n_jobs=-1), "GB": GradientBoostingClassifier(random_state=0),
    "SVM-lin": S(SVC(kernel="linear", probability=True, random_state=0)), "kNN": S(KNeighborsClassifier(3)),
    "LR-L1": S(LogisticRegression(penalty="l1", C=0.3, solver="liblinear")), "SVM-RBF": S(SVC(probability=True, random_state=0)),
    "ExtraTrees": ExtraTreesClassifier(1000, random_state=0, n_jobs=-1),
    "HistGB": HistGradientBoostingClassifier(random_state=0), "MLP": S(MLPClassifier((32, 16), max_iter=3000, random_state=0)),
}
THESIS_MODELS = ["LR", "DT", "RF", "GB", "SVM-lin", "kNN"]
proba = lambda m, X: m.predict_proba(X)[:, 1]  # noqa: E731
pct = lambda x: 100 * (np.exp(x) - 1)  # noqa: E731


def load(iv_dir, sweep_dir):
    jobs = vra.find_captures(iv_dir, sweep_dir)
    with Pool() as p:
        d = pd.DataFrame([r for r in p.map(vra.extract, jobs, chunksize=2) if r])
    qv = Path(next(f for f, m in jobs if m["kind"] == "endurance" and m["device"] == "m3" and m["hours"] == 10)).parent
    return d, qv


def period_job(job):
    f, meta = job
    wf = io.load_csv(f)
    if wf.dt > 5e-8:  # overview capture
        return []
    return [dict(r, **meta, file=Path(f).name, period=k, dt_ns=round(wf.dt * 1e9))
            for k, r in enumerate(features.extract_periods(wf))]


def load_periods(iv_dir, sweep_dir):
    jobs = [j for j in vra.find_captures(iv_dir, sweep_dir) if j[1]["kind"] == "endurance" and j[1]["hours"] >= 5]
    with Pool() as p:
        return pd.DataFrame([r for rows in p.map(period_job, jobs, chunksize=2) for r in rows])


def clean(d):
    """Drop captures whose period detection failed (half or double the drive period)."""
    bad = ~d.frequency.between(40e3, 55e3)
    print(f"{bad.sum()} captures dropped (period detection failed): " +
          ", ".join(f"{r.device} {r.hours} h {r.file}" for r in d[bad].itertuples()))
    return d[~bad].reset_index(drop=True)


def physics(d):
    e = d[(d.kind == "endurance") & (d.hours >= 5)]
    print("1. physics")
    print(f"  discharge slope C_d > no-discharge slope C_total in {(e.c_total < e.c_dielectric).mean():.0%} of {len(e)} captures")
    print("  U_b estimates [V], median per device (onset / geometry / energy):",
          e.groupby("device")[["u_b", "u_b_geometry", "u_b_energy"]].median().round(0).to_dict("index"))
    manley = 4 * e.c_dielectric * e.u_b * (e.v_max - e.u_b * (e.c_dielectric + e.c_gas) / e.c_dielectric)
    print(f"  measured E / ideal Manley E: median {(e.energy / manley).median():.2f}")
    pd_ = 760 * 0.01  # Torr cm, 100 um gap in air
    paschen = [365 * pd_ / np.log(15 * pd_ / np.log(1 + 1 / g)) for g in (0.1, 0.01)]
    ub = e.u_b.quantile([.25, .5, .75]).to_numpy()
    ad = 15 * pd_ * np.exp(-365 * pd_ / ub)
    print(f"  Paschen breakdown {paschen[0]:.0f}-{paschen[1]:.0f} V (gamma 0.1-0.01); measured U_b IQR {ub[0]:.0f}-{ub[2]:.0f} V")
    print(f"  alpha*d at U_b {ad[0]:.1f}-{ad[2]:.1f} (streamer/Meek needs ~18-20); implied gamma "
          f"{1 / np.expm1(ad[2]):.3f}-{1 / np.expm1(ad[0]):.3f}")
    w = d[(d.kind == "sweep") | ((d.kind == "endurance") & (d.hours == 20))]
    sweep = w.groupby(["device", "supply_V"])[["v_max", "c_dielectric"]].median()
    print("  supply sweep, discharge slope C_d [pF] at 8.5 -> 14 V input:",
          {dev: f"{g.c_dielectric.iloc[0] * 1e12:.1f} -> {g.c_dielectric.iloc[-1] * 1e12:.1f}" for dev, g in sweep.groupby("device")})
    print("  pulses per half period by sampling [ns]:", e.groupby(["device", "dt_ns"]).n_discharges.median().unstack().round(1).to_dict("index"))
    print(f"  thesis V_b / V_max = {(e.v_breakdown / e.v_max).mean():.3f} ± {(e.v_breakdown / e.v_max).std():.3f}; "
          f"corr(T_e, V_b) = {np.corrcoef(e.te, e.v_breakdown)[0, 1]:.4f}")
    return dict(paschen=paschen, sweep=sweep)


def did(e):
    """Change from 5-10 h to 15 h-end minus M3's change over the same sessions (session means of log x)."""
    lx, lv = np.log(e[list(DID)].abs()), np.log(e.v_max)
    h = (e.device == "m3") | (e.hours <= 10)  # healthy captures: their voltage dependence is removed in the adjusted test
    hv = lv[h] - lv[h].groupby(e.device[h]).transform("mean")
    rows = []
    for k in DID:
        y = lx[k][h] - lx[k][h].groupby(e.device[h]).transform("mean")
        slope = np.polyfit(hv, y, 1)[0]
        for adj, col in ((False, lx[k]), (True, lx[k] - slope * lv)):
            s = col.groupby([e.device, e.hours]).mean()
            for dev, last in (("m1", 25), ("m2", 40)):
                dd = (s.loc[dev] - s.loc["m3"].loc[s.loc[dev].index]).loc[:last]
                pre, post = dd.loc[:10], dd.loc[15:]
                rows.append(dict(feature=DID[k], device=dev, adjusted=adj, did=100 * (post.mean() - pre.mean()),
                                 p=stats.ttest_ind(post, pre, equal_var=False).pvalue))
    return pd.DataFrame(rows)


def self_referenced(e, feats):
    """log ratio to the device's own 5-10 h; a reference session is compared with the other one."""
    out = e[["device", "hours", "y"]].copy()
    for k in feats:
        lx, col = np.log(e[k].abs()), pd.Series(index=e.index, dtype=float)
        for dev, g in e.groupby("device"):
            base = g.hours <= 10
            for h, gh in g.groupby("hours"):
                col[gh.index] = lx[gh.index] - lx[g.index[base & (g.hours != h) if h <= 10 else base]].median()
        out[k] = col
    return out


def leave_one_session_out(L, model, feats):
    pr = pd.Series(index=L.index, dtype=float)
    for h in sorted(L.hours.unique()):
        trn, tst = L[L.hours != h], L[L.hours == h]
        pr[tst.index] = proba(clone(model).fit(trn[feats], trn.y), tst[feats])
    within = [roc_auc_score(g.y, pr[g.index]) for _, g in L[L.hours.between(15, 40)].groupby("hours")]
    return dict(within_session_auc=np.mean(within), detected=(pr[L.y == 1] > .5).mean(),
                M3_false_alarm=(pr[(L.device == "m3") & (L.hours >= 15)] > .5).mean(),
                reference_false_alarm=(pr[L.hours <= 10] > .5).mean())


def ml(e):
    warnings.filterwarnings("ignore")
    e = e[np.isfinite(e[F]).all(axis=1)].reset_index(drop=True).assign(y=lambda x: ((x.device != "m3") & (x.hours >= 15)).astype(int))
    print("\n3. ML: six thesis models and five newer ones")
    tr, te, m3 = e[e.device == "m2"], e[e.device == "m1"], e[e.device == "m3"]
    p1 = []
    for name, m in MODELS.items():
        f = clone(m).fit(tr[F], tr.y)
        p = proba(f, te[F])
        p1.append(dict(model=name, acc_M1=accuracy_score(te.y, p > .5), auc_M1=roc_auc_score(te.y, p),
                       M3_called_damaged=(proba(f, m3[m3.hours >= 15][F]) > .5).mean()))
    p1 = pd.DataFrame(p1).set_index("model")
    print("  3a. thesis protocol (14 raw features, train M2, test M1), then applied to the healthy M3 at 15-60 h")
    print(p1.round(2).to_string())

    L = self_referenced(e, sorted(set(F) | set(ROBUST)))
    p2 = pd.DataFrame({name: leave_one_session_out(L, m, F) for name, m in MODELS.items()}).T
    p2r = pd.DataFrame({name: leave_one_session_out(L, m, ROBUST) for name, m in MODELS.items()}).T
    print("  3b. self-referenced features, M3 in training as healthy, leave one session out")
    print(p2.round(2).to_string())
    print(f"      setting-robust features only: within-session AUC {p2r.within_session_auc.min():.2f}-"
          f"{p2r.within_session_auc.max():.2f}, reference-session false alarms {p2r.reference_false_alarm.min():.0%}-"
          f"{p2r.reference_false_alarm.max():.0%}")
    share = e.assign(a004=e.i_scale < .05, period=np.where(e.hours <= 10, "5-10 h", "15 h+")).groupby(["device", "period"]).a004.mean()
    print("      share of 0.04 A/div captures:", {f"{d} {p}": f"{v:.0%}" for (d, p), v in share.items()})
    for name in ("RF", "HistGB"):
        imp = [permutation_importance(clone(MODELS[name]).fit(L[L.hours != h][F], L[L.hours != h].y), L[L.hours == h][F],
                                      L[L.hours == h].y, scoring="roc_auc", n_repeats=20, random_state=0).importances_mean
               for h in range(15, 45, 5)]
        top = pd.Series(np.mean(imp, axis=0), index=F).sort_values(ascending=False).head(4)
        print(f"      permutation importance {name}: " + ", ".join(f"{k} {v:.3f}" for k, v in top.items()))

    med3 = e[e.device == "m3"].groupby("hours")[F].median()
    C = e[e.device != "m3"].copy()
    for k in F:
        C[k] = np.log(np.abs(C[k] / med3.loc[C.hours, k].to_numpy()))
    p3 = []
    for name, m in list(MODELS.items()) + [("V_rms only (LR)", S(LogisticRegression()))]:
        feats = ["v_rms"] if "only" in name else F
        r = {"model": name}
        for test in ("m1", "m2"):
            trn, tst = C[C.device != test], C[C.device == test]
            r[f"auc_to_{test}"] = roc_auc_score(tst.y, proba(clone(m).fit(trn[feats], trn.y), tst[feats]))
        p3.append(r)
    p3 = pd.DataFrame(p3).set_index("model")
    print("  3c. features relative to M3 in the same session, train one damaged device, test the other (AUC)")
    print(p3.round(2).to_string())
    return p1, p3


def periods_ml(p, seeds=(0, 1, 2)):
    """Single periods as rows, two from every capture, so every capture, session and sampling period weighs the same."""
    warnings.filterwarnings("ignore")
    p = p[p.frequency.between(40e3, 55e3) & np.isfinite(p[F]).all(axis=1)].reset_index(drop=True)
    p = p.assign(y=((p.device != "m3") & (p.hours >= 15)).astype(int))
    caps = list(p.groupby(["device", "hours", "file"]).indices.values())
    models = dict(MODELS, **{"V_rms only (LR)": S(LogisticRegression())})
    print(f"\n4. ML on single periods: {len(p)} periods from {len(caps)} captures, two per capture, {len(seeds)} draws")
    print("  drive frequency per device [kHz] (each device had its own driving power supply):",
          (p.groupby("device").frequency.median() / 1e3).round(2).to_dict())
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        b = p.loc[np.concatenate([rng.choice(g, 2, replace=False) if len(g) >= 2 else g for g in caps])].reset_index(drop=True)
        med3 = b[b.device == "m3"].groupby("hours")[F].median()
        c = b[b.device != "m3"].copy()
        for k in F:
            m = med3.loc[c.hours, k].to_numpy()
            c[k] = (c[k] - m) / np.abs(m)  # relative to M3 in the same session (some single-period values are 0)
        for name, model in models.items():
            feats = ["v_rms"] if "only" in name else F
            pr = pd.Series(index=b.index, dtype=float)  # E1: periods shuffled at random into 5 folds
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(b[feats], b.y):
                pr.iloc[te] = proba(clone(model).fit(b[feats].iloc[tr], b.y.iloc[tr]), b[feats].iloc[te])
            r = dict(seed=seed, model=name, E1_bal_acc=balanced_accuracy_score(b.y, pr > .5), E1_auc=roc_auc_score(b.y, pr),
                     E1_M3_false_alarm=(pr[(b.device == "m3") & (b.hours >= 15)] > .5).mean(),
                     E1_ref_false_alarm=(pr[b.hours <= 10] > .5).mean())
            r.update({f"E2_{k}": v for k, v in leave_one_session_out(b, model, feats).items()})  # E2: a session held out
            for dev in ("m1", "m2"):  # E4: relative to M3, trained on one damaged device, tested on the other
                trn, tst = c[c.device != dev], c[c.device == dev]
                r[f"E4_auc_{dev}"] = roc_auc_score(tst.y, proba(clone(model).fit(trn[feats], trn.y), tst[feats]))
            if "only" not in name:  # E2 with the voltage and the drive frequency alone
                vf = leave_one_session_out(b, model, ["v_rms", "frequency"])
                r["E2_vf_within_session_auc"], r["E2_vf_reference_false_alarm"] = vf["within_session_auc"], vf["reference_false_alarm"]
            rows.append(r)
        if seed == seeds[0]:
            for name in ("RF", "GB"):
                imp = [permutation_importance(clone(MODELS[name]).fit(b[b.hours != h][F], b[b.hours != h].y), b[b.hours == h][F],
                                              b[b.hours == h].y, scoring="roc_auc", n_repeats=10, random_state=0).importances_mean
                       for h in range(15, 45, 5)]
                top = pd.Series(np.mean(imp, axis=0), index=F).sort_values(ascending=False).head(4)
                print(f"  E2 permutation importance {name}: " + ", ".join(f"{k} {v:.3f}" for k, v in top.items()))
    s = pd.DataFrame(rows).groupby("model").mean(numeric_only=True).drop(columns="seed").loc[list(models)]
    print("  E1 shuffled periods (5-fold) | E2 a session held out | E4 relative to M3, the other damaged device; mean of draws")
    print(s.round(2).to_string())
    print(f"  max sd over draws: {pd.DataFrame(rows).groupby('model').std(numeric_only=True).drop(columns='seed').max().max():.2f}")
    return s


MECH = {"v_rms": "V_rms", "frequency": "f", "v_max": "V_max", "i_rms": "I_rms", "energy": "E", "c_dielectric": "C_d",
        "c_gas": "C_g", "c_total": "C_total", "q_max": "ΔQ", "q_per_v": "ΔQ/V_max", "von_frac": "V_on/V_max",
        "u_b": "U_b", "n_discharges": "pulses", "ne_mean": "n_e", "te": "T_e", "v_min": "V_min (thesis)"}


def vrms_mechanism(d, n_boot=300, seed=0):
    """Why V_rms: power supply-load coupling, the damage chain, its size, effect-to-noise and randomisation checks."""
    x = d.assign(q_per_v=d.q_max / d.v_max, von_frac=d.v_on_lissajous / d.v_max)
    e = x[(x.kind == "endurance") & (x.hours >= 5)].reset_index(drop=True)
    w = x[(x.kind == "sweep") | ((x.kind == "endurance") & (x.hours == 20))]
    slope = lambda a, b: np.polyfit(np.log(a), np.log(b), 1)[0]  # noqa: E731
    print("\n5. why V_rms: mechanism and robustness")
    sw = w.groupby(["device", "supply_V"])[["v_rms", "q_per_v"]].median()
    m3 = e[e.device == "m3"].groupby("hours")[["v_rms", "q_per_v"]].median()
    sweep_el = {dev: slope(g.v_rms, g.q_per_v) for dev, g in sw.groupby("device")}
    m3_el = slope(m3.v_rms, m3.q_per_v)
    S = -1 / m3_el  # closed-loop sensitivity -dlnV/dln(ΔQ/V) of each driving power supply to its load, from the healthy device
    print(f"  5a. dln(ΔQ/V)/dlnV when the SOURCE moves V (sweep): " + ", ".join(f"{k} {v:+.2f}" for k, v in sweep_el.items()) +
          f"; when the LOAD moves V (healthy M3, sessions): {m3_el:+.2f} -> S = -dlnV/dln(ΔQ/V) = {S:.2f}")

    lx = np.log(e[list(MECH)].abs()).assign(device=e.device, hours=e.hours)
    s = lx.groupby(["device", "hours"])[list(MECH)].median()
    diff = {dev: (s.loc[dev] - s.loc["m3"].loc[s.loc[dev].index]).loc[:last] for dev, last in (("m1", 25), ("m2", 40))}
    did = pd.DataFrame({dev: dd.loc[15:].mean() - dd.loc[:10].mean() for dev, dd in diff.items()})
    pred = -S * did.loc["q_per_v"]
    print("  5b. damage chain against M3 (session medians, %): " +
          "; ".join(f"{dev}: V_on/V_max {100 * did.loc['von_frac', dev]:+.1f}, ΔQ/V {100 * did.loc['q_per_v', dev]:+.1f}, "
                    f"V_rms observed {100 * did.loc['v_rms', dev]:+.2f}, predicted -S·ΔQ/V {100 * pred[dev]:+.2f}" for dev in did))
    r = pd.concat([(dd - dd.loc[:10].mean()).assign(device=dev) for dev, dd in diff.items()]).reset_index()
    rho = stats.spearmanr(r.von_frac, r.v_rms)
    print(f"  5c. per session (device minus M3, {len(r)} sessions): Spearman(V_on/V_max, V_rms) {rho.correlation:+.2f} (p {rho.pvalue:.3f});"
          f" damaged sessions discharging earlier than M3: {int(((r.hours >= 15) & (r.von_frac < 0)).sum())}/{int((r.hours >= 15).sum())}")
    t = e.assign(t=pd.to_datetime(e.meta_time, format="%d-%b-%y %H:%M:%S")).groupby(["device", "hours"]).t.median()
    for dev, dd in diff.items():
        gap = [(t.loc[("m3", h)] - t.loc[(dev, h)]).total_seconds() / 60 for h in dd.index]
        g = stats.spearmanr(gap, dd.v_rms)
        print(f"      measurement gap to M3 vs V_rms difference, {dev}: {min(gap):.0f}-{max(gap):.0f} min, Spearman {g.correlation:+.2f} (p {g.pvalue:.2f})")

    noise = lx[lx.hours <= 10].groupby(["device", "hours"])[list(MECH)].std().median()  # within-session spread
    en = did.div(noise, axis=0)
    same = np.sign(did.m1) == np.sign(did.m2)
    score = en.abs().min(axis=1).where(same, 0).sort_values(ascending=False)
    print("  5d. effect / within-session noise (smaller of the two devices; 0 if the devices disagree in sign):")
    print("      " + ", ".join(f"{MECH[k]} {score[k]:.1f} (noise {100 * noise[k]:.1f} %)" for k in score.index[:8]))

    perm = {}
    for k, sign in (("v_rms", -1), ("von_frac", -1), ("q_per_v", 1), ("c_total", 1), ("energy", 1)):
        z = []
        for dev, dd in diff.items():
            st = lambda pre: sign * (dd[k].drop(list(pre)).mean() - dd[k].loc[list(pre)].mean())  # noqa: E731
            null = np.array([st(c) for c in itertools.combinations(dd.index, 2)])
            z.append(((st([5, 10]) - null.mean()) / null.std(), (null - null.mean()) / null.std()))
        comb = np.add.outer(z[0][1], z[1][1]).ravel()
        perm[k] = (comb >= z[0][0] + z[1][0] - 1e-12).mean()
    print(f"  5e. exact permutation test over which sessions are pre-damage ({len(comb)} labelings, both devices): " +
          ", ".join(f"{MECH[k]} p {v:.3f}" for k, v in perm.items()))

    rng = np.random.default_rng(seed)
    groups = list(lx.groupby(["device", "hours"]).indices.values())

    def aucs(f):
        med = f.groupby(["device", "hours"])[list(MECH)].median()
        out = {}
        for k in MECH:
            a = []
            for dev, last in (("m1", 25), ("m2", 40)):
                g = f[(f.device == dev) & (f.hours <= last)]
                a.append(roc_auc_score(g.hours >= 15, g[k].to_numpy() - med.loc["m3", k].loc[g.hours].to_numpy()))
            a = np.array(a)
            out[k] = max(a.mean(), 1 - a.mean()) if (a > .5).all() or (a < .5).all() else .5  # same direction in both
        return pd.Series(out)
    auc = aucs(lx)
    first = pd.DataFrame([aucs(lx.iloc[np.concatenate([rng.choice(g, len(g)) for g in groups])]).rank(ascending=False) == 1
                          for _ in range(n_boot)]).mean()
    print(f"  5f. univariate AUC against M3 (mean of M1, M2): " + ", ".join(f"{MECH[k]} {v:.2f}" for k, v in auc.sort_values(ascending=False)[:6].items()))
    print(f"      bootstrap over captures ({n_boot}): ranked first " + ", ".join(f"{MECH[k]} {v:.0%}" for k, v in first[first > 0].sort_values(ascending=False).items()))
    return dict(sw=sw, m3=m3, sweep_el=sweep_el, m3_el=m3_el, S=S, did=did, pred=pred, r=r, rho=rho, score=score, noise=noise,
                perm=perm, auc=auc, first=first)


def figure_mechanism(mk, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, AQUA, YELLOW, INK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#52514e", "#e4e3df"
    DEV = {"m1": (BLUE, "o", "M1"), "m2": (ORANGE, "s", "M2"), "m3": (AQUA, "D", "M3")}
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.8, "axes.edgecolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": GREY,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "legend.frameon": False})
    fig, ax = plt.subplots(2, 2, figsize=(8, 7))

    a = ax[0, 0]  # (a) power supply moves V vs load moves V
    for dev, g in mk["sw"].groupby("device"):
        g = np.log(g.droplevel(0))
        c, m, lab = DEV[dev]
        a.plot(100 * (g.v_rms - g.v_rms.iloc[0]), 100 * (g.q_per_v - g.q_per_v.iloc[0]), m + "-", color=c, ms=5, mec="white", mew=.7,
               lw=1, label=f"{lab}: supply input 8.5 → 14 V")
    m3 = np.log(mk["m3"])
    base = m3.loc[:10].mean()
    a.plot(100 * (m3.v_rms - base.v_rms), 100 * (m3.q_per_v - base.q_per_v), "D", color=AQUA, mfc="white", mew=1.2, ms=5,
           label="M3 time points, supply fixed")
    a.axhline(0, color=INK, lw=.6); a.axvline(0, color=INK, lw=.6)
    a.set(xlabel="change of V_rms [%]", ylabel="change of charge per volt ΔQ/V_max [%]",
          title=f"(a) Load-driven V changes run opposite (S = {mk['S']:.2f})")
    a.legend(loc="upper right")

    a = ax[0, 1]  # (b) the damage chain and the predicted size
    steps = [("V_on/V_max", "von_frac"), ("ΔQ/V_max", "q_per_v"), ("V_rms\nobserved", "v_rms")]
    xs = np.arange(len(steps) + 1)
    for dev, dx in (("m1", -.12), ("m2", .12)):
        c, m, lab = DEV[dev]
        vals = [100 * mk["did"].loc[k, dev] for _, k in steps] + [100 * mk["pred"][dev]]
        a.plot(xs + dx, vals, m, color=c, ms=6, mec="white", mew=.7, label=lab)
    a.axhline(0, color=INK, lw=.8)
    a.set_yscale("symlog", linthresh=5)
    a.set_yticks([-30, -10, -5, -2, 0, 2, 5, 10], ["-30", "-10", "-5", "-2", "0", "2", "5", "10"])
    a.set_xticks(xs, [s for s, _ in steps] + ["V_rms\npredicted"])
    a.set(ylabel="damaged minus M3 [%] (symlog)", title="(b) Earlier discharge → more charge → supply voltage drops")
    a.legend(loc="upper right")
    a.grid(axis="x", visible=False)

    a = ax[1, 0]  # (c) time point by time point
    r = mk["r"]
    for dev in ("m1", "m2"):
        c, m, lab = DEV[dev]
        for dmg, fill in ((False, "white"), (True, c)):
            g = r[(r.device == dev) & ((r.hours >= 15) == dmg)]
            a.plot(100 * g.von_frac, 100 * g.v_rms, m, color=c, mfc=fill, ms=6, mew=1.1,
                   label=f"{lab} {'damaged' if dmg else 'before damage'}")
    a.axhline(0, color=INK, lw=.6); a.axvline(0, color=INK, lw=.6)
    a.set(xlabel="V_on/V_max, device minus M3 [%]", ylabel="V_rms, device minus M3 [%]",
          title=f"(c) Earlier discharge, larger V_rms drop (Spearman {mk['rho'].correlation:+.2f}, p {mk['rho'].pvalue:.3f})")
    a.legend(loc="upper left")

    a = ax[1, 1]  # (d) effect-to-noise ranking
    sc = mk["score"].iloc[:10][::-1]
    a.barh(range(len(sc)), sc, color=[BLUE if k == "v_rms" else "#9ec3ee" for k in sc.index], height=.6)
    for y, (k, v) in enumerate(sc.items()):
        a.text(v + .08, y, f"noise {100 * mk['noise'][k]:.1f} %" + (f"; first in {mk['first'].get(k, 0):.0%} of resamples" if mk['first'].get(k, 0) > 0 else ""),
               va="center", fontsize=6.6, color=INK)
    a.set_yticks(range(len(sc)), [MECH[k] for k in sc.index])
    a.set(xlim=(0, 8.5), xlabel="damage effect / noise within a time point (smaller of M1, M2)",
          title="(d) V_rms has by far the largest effect relative to its noise")
    a.grid(axis="y", visible=False)
    fig.text(.01, .004, "Medians per time point (all sources at the same discharge time, same measurement conditions); M3 (healthy) "
             "removes the effects shared by all sources.\n(a) relative to 8.5 V input (sweep) or to M3's 5-10 h (time points). "
             "(b) predicted = -S·ΔQ/V_max. (c) filled: time points after damage.\n(d) 0 for features whose change has opposite "
             "signs in M1 and M2; 'first' = share of 300 capture resamples in which the feature has the highest AUC against M3.",
             fontsize=6.2, color=INK)
    fig.tight_layout(rect=(0, .055, 1, 1))
    fig.savefig(Path(out) / "vrms_mechanism.png", bbox_inches="tight")


def figure(d, qv, ph, dd, p1, p3, ps, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, AQUA, INK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e4e3df"
    DEV = {"m1": (BLUE, "o", "M1 (damaged)"), "m2": (ORANGE, "s", "M2 (damaged)"), "m3": (AQUA, "D", "M3 (healthy)")}
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.8, "axes.edgecolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": GREY,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "legend.frameon": False})
    fig = plt.figure(figsize=(8, 13.8))
    gs = fig.add_gridspec(4, 2)
    ax = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(3)])

    a = ax[0, 0]  # (a) Lissajous graph of the M3 10 h capture closest to the median of its time point
    s = d[(d.kind == "endurance") & (d.device == "m3") & (d.hours == 10) & (d.dt_ns == 10)]
    t, v, i, q, L = plots.one_period(io.load_csv(qv / s.loc[(s.c_dielectric - s.c_dielectric.median()).abs().idxmin(), "file"]))
    a.plot(v / 1e3, q * 1e9, color=INK, lw=.8)
    for k, (ft, fd) in enumerate(zip(L.fit_total, L.fit_dielectric)):
        sg = 1 if k == 0 else -1
        st, sd = np.linspace(-sg * L.V_max, sg * L.V_on, 20), np.linspace(sg * L.V_on, sg * L.V_max, 20)
        a.plot(st / 1e3, (ft.slope * st + sg * ft.intercept) * 1e9, "--", color=BLUE, lw=1.4,
               label=f"no discharge: C_total {L.C_total * 1e12:.1f} pF" if k == 0 else None)
        a.plot(sd / 1e3, (fd.slope * sd + sg * fd.intercept) * 1e9, "--", color=ORANGE, lw=1.4,
               label=f"discharge: C_d {L.C_dielectric * 1e12:.1f} pF" if k == 0 else None)
    a.set(xlabel="applied voltage [kV]", ylabel="charge [nC]", title="(a) Two slopes in every Lissajous graph: the cells discharge")
    a.legend(loc="upper left")
    a.text(.98, .04, f"M3, 10 h; U_b = {L.U_b:.0f} V", transform=a.transAxes, ha="right", fontsize=6.8, color=INK)

    a = ax[0, 1]  # (b) partial surface discharging in the supply sweep
    for dev, g in ph["sweep"].groupby("device"):
        c, mk, lab = DEV[dev]
        a.plot(g.v_max / 1e3, g.c_dielectric * 1e12, "-", marker=mk, color=c, ms=5, mec="white", mew=.7, lw=1.2, label=lab)
    a.set(xlabel="peak applied voltage V_max [kV] (supply input 8.5-14 V)", ylabel="discharge slope C_d [pF]",
          title="(b) Not all cells discharge: the discharge slope grows with voltage")
    a.text(.98, .3, "a fixed dielectric capacitance\nwould give flat lines", transform=a.transAxes, ha="right",
           fontsize=6.8, color=INK)
    a.legend(loc="upper left")

    a = ax[1, 0]  # (c) burning voltage vs Paschen, thesis V_b
    e = d[(d.kind == "endurance") & (d.hours >= 5)]
    a.axvspan(ph["paschen"][0] / 1e3, ph["paschen"][1] / 1e3, color=GREY, lw=0)
    for y, (dev, g) in enumerate(e.groupby("device")):
        c, mk, lab = DEV[dev]
        for col, fill in (("u_b", c), ("v_breakdown", "white")):
            qq = g[col].quantile([.25, .5, .75]) / 1e3
            a.plot([qq.iloc[0], qq.iloc[2]], [2 - y, 2 - y], color=c, lw=1.4)
            a.plot(qq.iloc[1], 2 - y, mk, color=c, mfc=fill, ms=6, mew=1.2)
        a.text(1.25, 2 - y, lab, va="center", ha="left", fontsize=6.8, color=INK)
    a.text(.62, 2.65, "shade: Paschen, 100 µm air\nfilled: burning voltage U_b", ha="left", va="center", fontsize=6.8, color=INK)
    a.text(2.4, 2.65, "hollow: thesis V_b\n(Eq. 15) = V_max", ha="right", va="center", fontsize=6.8, color=INK)
    a.text(2.4, -.55, "at U_b: alpha·d = 3-5, streamer needs ~18-20\n-> Townsend discharge", ha="right", va="center",
           fontsize=6.8, color=INK)
    a.set(xlim=(.6, 2.45), ylim=(-.85, 3.1), yticks=[], xlabel="voltage [kV]",
          title="(c) U_b sits in the Paschen range; thesis V_b is V_max")
    a.grid(axis="y", visible=False)

    a = ax[1, 1]  # (d) damage effect per feature against M3
    raw = dd[~dd.adjusted]
    feats = list(DID.values())[::-1]
    a.axhspan(len(feats) - 1.5, len(feats) - .5, color=GREY, lw=0)
    for dev, dy in (("m1", .14), ("m2", -.14)):
        c, mk, lab = DEV[dev]
        g = raw[raw.device == dev].set_index("feature").loc[feats]
        y = np.arange(len(feats)) + dy
        sig = g.p < .05
        a.plot(g.did[~sig], y[~sig], mk, mfc="white", mec=c, ms=5.5, mew=1.2)
        a.plot(g.did[sig], y[sig], mk, color=c, mec="white", ms=6, mew=.7)
        a.plot([], [], mk, color=c, ms=5.5, label=lab)
    a.axvline(0, color=INK, lw=.8)
    a.set_yticks(range(len(feats)), feats)
    a.set(xlabel="change after damage minus M3's change [%]", xlim=(-40, 26),
          title="(d) Against M3, only V_rms drops alike in both devices")
    a.legend(loc="lower right", title="filled: p < 0.05 (time-point level)", title_fontsize=6.5, frameon=True,
             facecolor="white", edgecolor="none", framealpha=1)
    a.grid(axis="y", visible=False)

    names = list(MODELS)
    y = np.arange(len(names))[::-1]
    a = ax[2, 0]  # (e) thesis protocol
    for yy, n in zip(y, names):
        a.plot([100 * p1.loc[n, "M3_called_damaged"], 100 * p1.loc[n, "acc_M1"]], [yy, yy], color=GREY, lw=2.5, zorder=1)
    a.plot(100 * p1.acc_M1, y, "o", color=BLUE, ms=6, mec="white", mew=.7, label="accuracy on M1 (damaged)")
    a.plot(100 * p1.M3_called_damaged, y, "o", color=ORANGE, ms=6, mec="white", mew=.7, label="healthy M3 called damaged, 15-60 h")
    a.set_yticks(y, [f"{n}{' *' if n in THESIS_MODELS else ''}" for n in names])
    a.set(xlim=(30, 102), xlabel="% of captures", title="(e) Thesis protocol: high accuracy, healthy M3 also called damaged")
    a.legend(loc="upper left")
    a.grid(axis="y", visible=False)

    a = ax[2, 1]  # (f) features relative to M3
    vr = p3.loc["V_rms only (LR)"]
    a.axvspan(min(vr), max(vr), color=GREY, lw=0)
    a.axvline(vr.mean(), color=INK, lw=1)
    a.axvline(.5, color=INK, lw=.8, ls=":")
    a.text(vr.mean(), -1, "V_rms alone", fontsize=6.8, color=INK, ha="center", va="center")
    a.text(.5, -1, "chance", fontsize=6.8, color=INK, ha="center", va="center")
    for yy, n in zip(y, names):
        lo, hi = p3.loc[n, ["auc_to_m1", "auc_to_m2"]].min(), p3.loc[n, ["auc_to_m1", "auc_to_m2"]].max()
        a.plot([lo, hi], [yy, yy], color=BLUE, lw=1.4)
        a.plot(p3.loc[n, ["auc_to_m1", "auc_to_m2"]].mean(), yy, "o", color=BLUE, ms=5.5, mec="white", mew=.7)
    a.set_yticks(y, [f"{n}{' *' if n in THESIS_MODELS else ''}" for n in names])
    a.set(xlim=(.1, 1), ylim=(-1.5, len(names) - .5), xlabel="AUC on the other damaged device (bar: M2→M1 and M1→M2)",
          title="(f) With M3 as reference, no model beats V_rms alone")
    a.grid(axis="y", visible=False)

    a = fig.add_subplot(gs[3, :])  # (g) single periods: shuffled vs held out
    x = np.arange(len(ps))
    e4 = ps[["E4_auc_m1", "E4_auc_m2"]].mean(axis=1)
    for xx, lo, hi in zip(x, np.minimum(e4, ps.E2_within_session_auc), ps.E1_auc):
        a.plot([xx, xx], [lo, hi], color=GREY, lw=2.5, zorder=1)
    a.plot(x, ps.E1_auc, "o", color="#eda100", ms=6.5, mec="white", mew=.7, label="E1 periods shuffled at random (5-fold)")
    a.plot(x, ps.E2_within_session_auc, "s", color=INK, ms=5.5, mec="white", mew=.7,
           label="E2 a time point held out (damaged vs M3 within the time point)")
    a.plot(x, e4, "D", color=BLUE, ms=5.5, mec="white", mew=.7, label="E4 relative to M3, the other damaged device")
    a.axhline(.5, color=INK, lw=.8, ls=":")
    a.set_xticks(x, [f"{n}{' *' if n in THESIS_MODELS else ''}".replace(" only (LR)", " alone") for n in ps.index],
                 rotation=25, ha="right")
    a.set(ylim=(.3, 1.02), ylabel="AUC",
          title="(g) Single periods, two per capture: shuffling looks excellent; held-out time points and devices do not")
    a.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    a.grid(axis="x", visible=False)

    fig.text(.01, .004, f"Raw archive of July 2020: M1, M2 damaged from 15 h (SEM/EDS); M3 discharged alongside under the same "
             f"conditions, not damaged. {len(e)} captures from 5 h; supply sweep at 20 h.\n(c) bars: interquartile range. "
             "(e)-(g) * thesis models (2023); others are newer tabular methods. (f) shade: V_rms alone in both directions. "
             "(g) mean of three draws.",
             fontsize=6.2, color=INK)
    fig.tight_layout(rect=(0, .025, 1, 1))
    fig.savefig(Path(out) / "consistency_check.png", bbox_inches="tight")


def report(d):
    pd.set_option("display.width", 200)
    ph = physics(d)
    e = d[(d.kind == "endurance") & (d.hours >= 5)].reset_index(drop=True)
    dd = did(e)
    print("\n2. damage effect against M3 [%], session means, pre 5-10 h, post 15 h-end")
    t = dd.pivot_table(index="feature", columns=["adjusted", "device"], values=["did", "p"]).loc[list(DID.values())]
    print(t.round(3).to_string())
    print("   (adjusted: after removing the healthy dependence on V_max)")
    p1, p3 = ml(e)
    return ph, dd, p1, p3


def main(iv_dir, sweep_dir, out="."):
    d, qv = load(iv_dir, sweep_dir)
    d = clean(d)
    ph, dd, p1, p3 = report(d)
    ps = periods_ml(load_periods(iv_dir, sweep_dir))
    figure(d, qv, ph, dd, p1, p3, ps, out)
    figure_mechanism(vrms_mechanism(d), out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
