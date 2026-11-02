"""Verification against the complete raw-data archive (devices M1, M2, M3; data not distributed here).

    python examples/verify_raw_archive.py "<archive>/전류전압 측정" "<archive>/전압에 따른 QV 변화/I 기준으로 계산"

The archive holds every endurance session of the three devices that ran in parallel in July 2020
(M1 0-25 h, M2 0-40 h, M3 0-60 h; 20 detail captures per session) and the supply-voltage sweep at
20 h (8.5 / 9 / 10 / 14 V supply input). M3 was discharged alongside M1 and M2 under the same
conditions and was not damaged; the thesis did not use it, so it serves as an undamaged control measured
10-33 min after M1 in every session.

Prints the tables of docs/raw-data-verification.md:
  1. session means (common-mode V_rms changes)          4. features vs oscilloscope settings
  2. difference-in-differences against M3                5. supply-voltage sweep (V_b vs U_b)
  3. thesis classifiers applied to the undamaged M3
"""
import re
import sys
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone

from dbd_diagnostics import features, io, ml

SESSION = re.compile(r"(\d{8})\s*M(\d)\s*(\d+)시간")
SWEEP = re.compile(r"M(\d)\s*([\d.]+)\s*V")
COLS = ["v_rms", "v_max", "i_mA", "c_d_pF", "c_t_pF", "q_nC", "u_b", "power", "n_discharges"]


def nfc(s):
    """macOS stores Korean file names decomposed (NFD); compare them in NFC."""
    return unicodedata.normalize("NFC", s)


def find_captures(iv_dir, sweep_dir):
    jobs = []
    for src in sorted(Path(iv_dir).iterdir()):
        for d in sorted(src.iterdir()) if src.is_dir() else []:
            m = SESSION.match(nfc(d.name))
            if d.is_dir() and m and "오류" not in nfc(d.name):  # "20200718 M3 0시간 - 데이터 자체가 오류"
                meta = dict(kind="endurance", device=f"m{m.group(2)}", hours=int(m.group(3)), supply_V=8.5)
                jobs += [(f, meta) for f in sorted(d.glob("ALL*.CSV"))]
    for d in sorted(Path(sweep_dir).iterdir()):
        m = SWEEP.match(nfc(d.name))
        if m and float(m.group(2)) != 8.5:  # 8.5 V is the 20 h endurance session itself
            meta = dict(kind="sweep", device=f"m{m.group(1)}", hours=20, supply_V=float(m.group(2)))
            jobs += [(f, meta) for f in sorted(d.glob("ALL*.CSV"))]
    m1_9v = Path(iv_dir) / "1번소스" / "20200721 M1 20시간" / "9V"
    jobs += [(f, dict(kind="sweep", device="m1", hours=20, supply_V=9.0)) for f in sorted(m1_9v.glob("ALL*.CSV"))]
    return jobs


def extract(job):
    f, meta = job
    wf = io.load_csv(f)
    if wf.dt > 5e-8:  # overview capture (ALL0021)
        return None
    r = features.extract(wf)
    r.update(meta, file=f.name, i_scale=float(wf.meta["Vertical Scale"][1]), dt_ns=round(wf.dt * 1e9))
    r.update(i_mA=r["i_rms"] * 1e3, c_d_pF=r["c_dielectric"] * 1e12, c_t_pF=r["c_total"] * 1e12, q_nC=r["q_max"] * 1e9)
    return r


def did_table(e):
    """Change after damage minus the change of the undamaged M3 over the same sessions (session means)."""
    s = e[e.hours.between(5, 40)].groupby(["hours", "device"])[COLS].mean()
    rows = []
    for dev, last in (("m1", 25), ("m2", 40)):
        for c in COLS:
            d = (s[c].xs(dev, level="device") - s[c].xs("m3", level="device")).dropna().loc[:last]
            pre, post = d.loc[:10], d.loc[15:]
            a = e[(e.device == dev) & e.hours.between(5, 10)][c]
            b = e[(e.device == dev) & e.hours.between(15, last)][c]
            ca = e[(e.device == "m3") & e.hours.between(5, 10)][c]
            cb = e[(e.device == "m3") & e.hours.between(15, last)][c]
            rows.append({"device": dev, "feature": c, "naive": b.mean() - a.mean(),
                         "M3 change": cb.mean() - ca.mean(),
                         "p (M3)": stats.ttest_ind(ca, cb, equal_var=False).pvalue,
                         "DiD": post.mean() - pre.mean(),
                         "p (DiD, sessions)": stats.ttest_ind(pre, post, equal_var=False).pvalue})
    return pd.DataFrame(rows)


def settings_effect(e, y):
    """Within-session (fixed-effect) OLS of y on log2(A/div) and the 20 ns sampling flag."""
    X = pd.get_dummies(e.device + e.hours.astype(str)).astype(float)
    X["log2_Adiv"] = np.log2(e.i_scale / 0.1)
    X["dt20"] = (e.dt_ns == 20).astype(float)
    ok = e[y].notna().to_numpy()
    A, v = X[ok].to_numpy(), e[y].to_numpy()[ok]
    b = np.linalg.lstsq(A, v, rcond=None)[0]
    res = v - A @ b
    cov = res @ res / (len(v) - A.shape[1]) * np.linalg.pinv(A.T @ A)
    out = {"feature": y}
    for k, name in ((-2, "per doubling of A/div"), (-1, "20 ns vs 10 ns")):
        t = b[k] / np.sqrt(cov[k, k])
        out[name] = b[k]
        out[f"p ({name})"] = 2 * stats.t.sf(abs(t), len(v) - A.shape[1])
    return out


def main(iv_dir, sweep_dir):
    with Pool() as pool:
        rows = [r for r in pool.map(extract, find_captures(iv_dir, sweep_dir), chunksize=4) if r]
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    e = df[(df.kind == "endurance") & (df.hours >= 5)].copy()
    e["label"] = np.where(e.device == "m3", 0, (e.hours >= 15).astype(int))
    pd.set_option("display.width", 200)

    print("1. V_rms session means [V] (all devices measured within ~15 min of each other)")
    print(e.groupby(["hours", "device"]).v_rms.mean().unstack().round(0).to_string())

    print("\n2. difference-in-differences vs M3 (pre 5-10 h, post 15 h-end)")
    print(did_table(e).to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    print("\n3. % of M3 captures classified 'degraded' by the thesis models trained on M2")
    tr = e[e.device == "m2"].dropna(subset=ml.THESIS_FEATURES)
    m3 = e[e.device == "m3"].dropna(subset=ml.THESIS_FEATURES)
    fp = {name: pd.Series(clone(m).fit(tr[ml.THESIS_FEATURES], tr.label).predict(m3[ml.THESIS_FEATURES]),
                          index=m3.index).groupby(m3.hours).mean() * 100
          for name, m in ml.thesis_models().items()}
    print(pd.DataFrame(fp).T.round(0).to_string())

    print("\n4. within-session effect of oscilloscope settings")
    print(pd.DataFrame([settings_effect(e, y) for y in ("i_mA", "n_discharges", "c_d_pF", "u_b", "v_rms")])
          .to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    print("\n5. supply-voltage sweep at 20 h: slope against V_max")
    sw = df[(df.hours == 20) & (df.kind.isin(["endurance", "sweep"])) & (df.device.isin(["m1", "m2", "m3"]))]
    print(sw.groupby(["device", "supply_V"])[["v_rms", "v_max", "v_breakdown", "u_b", "te"]].mean().round(1).to_string())
    for dev, g in sw.groupby("device"):
        s_vb, s_ub = stats.linregress(g.v_max, g.v_breakdown), stats.linregress(g.v_max, g.u_b)
        print(f"  {dev}: dV_b/dV_max = {s_vb.slope:.2f} (r={s_vb.rvalue:.2f})   dU_b/dV_max = {s_ub.slope:.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
