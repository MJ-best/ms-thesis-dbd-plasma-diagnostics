"""Aging test on three ATI devices run until the discharge stopped (data not distributed here).

    python examples/ati_aging_test.py "<archive>/μ-DBD (large) Aging 데이터 모음" docs/img

ATI #1-3 are copies of one medical-device source (same cell structure as the thesis devices, larger
area), run in nitrogen (1.5 lpm, DC 13 V input) until the discharge stopped. There is no control
device, so this is a test only and is never mixed with the thesis data (M1-M3).

Prints the tables of docs/ati-aging-test.ko.md and writes ati_aging_test.png:
  1. session table aligned to the day the discharge stopped
  2. H1: monotone trends per device, and in the common 0-490 h window
  3. same-day voltage changes at a fixed device state
  4. supply behaviour (I_rms nearly constant, V vs C elasticity) and the probe/stray step
  5. H3: 'loaded state' (C_cell >= 1.25 x own baseline) relative to the end of life
"""
import re
import sys
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dbd_diagnostics import circuit_fit, features, io

END = {"ATI #1": "2019-12-09", "ATI #2": "2020-02-24", "ATI #3": "2020-11-13"}  # last day with data
ATI1 = {"1203": "", "20191203 ATI 시험분석표/20191203 재측정": "same-day repeat", "1204A": "after IPA cleaning",
        "1205": "", "1209-1": "", "1209-2": "after IPA cleaning"}
# Excluded ATI #1 folders: A1209 (copy of the Line-DBD 0 h captures), 1211 (copy of ATI #2 2019-12-11),
# 1121 and 1125 (79 / 48 kHz, Lissajous R^2 < 0.1), 1204 morning "방전X" (no discharge, 46 kHz).
# Cumulative discharge hours from the operating logs (ATI #2: daily log; ATI #3: summary table of 2020-08-17).
ATI2_HOURS = {"20191211": 0, "20191212": 0, "20191217": 35.8, "20191223": 66.3, "20191227": 93.2, "20200106": 156.4,
              "20200113": 186.4, "20200120": 244.4, "20200128": 283.0, "20200203": 333.9, "20200219": 431.3,
              "20200224": 469.4}
ATI3_HOURS = {"0313": 97, "0328": 223, "0418": 317, "0502": 374, "0520": 490, "0526": 534, "0529": 562, "0610": 633,
              "0619": 705, "0704": 817, "0711": 853, "0718": 904, "0727": 933, "0731": 976, "0810": 987}
ATI3_SETUP_STEP = "2020-05-10"  # current and C_cell drop together between 05-02 and 05-20 (probe / stray change)


def nfc(s):
    return unicodedata.normalize("NFC", s)


def header(f):
    meta = {}
    with open(f, errors="replace") as fh:
        for _ in range(20):
            parts = [p.strip() for p in fh.readline().split(",")]
            if len(parts) >= 2:
                meta[parts[0]] = parts[1]
    return meta.get("Time"), float(meta.get("Sampling Period") or "nan")


def find_captures(root):
    root = Path(root)
    rows = []
    for f in root.rglob("*.[cC][sS][vV]"):
        if f.name.startswith("._"):
            continue
        rel = [nfc(p) for p in f.relative_to(root).parts[:-1]]
        sub = "/".join(rel[1:])
        if rel[0] == "1번 ATI" and sub in ATI1:
            dev, ses, note = "ATI #1", sub, ATI1[sub]
        elif rel[:2] == ["2번 ATI", "2번 ATI 전류전압 데이터"] and len(rel) > 2 and re.match(r"\d{8}", rel[2]):
            dev, ses, note = "ATI #2", rel[2][:8], ""
        elif rel[:2] == ["3번 ATI", "3번 ATI 전류전압 측정데이터"] and len(rel) > 2 and re.match(r"\d{4}A", rel[2]):
            if "소스 전압" in rel[2]:
                continue
            dev, ses = "ATI #3", rel[2]
            note = ("distorted" if "일그러짐" in ses else "13.5 V input" if "13.5V" in ses
                    else "odd" if "이상" in ses else "same-day repeat" if "재측정" in ses else "")
        else:
            continue
        time, dt = header(f)
        if time and dt <= 5e-8:  # detail captures only
            rows.append(dict(device=dev, session=ses, note=note, path=str(f), file=f.name, time=time, depth=len(rel)))
    c = pd.DataFrame(rows).sort_values("depth")  # copies of a capture: keep the one in the session folder itself
    c = c.drop_duplicates(subset=["device", "time", "file"]).drop(columns="depth")
    c = c[~((c.device == "ATI #2") & (c.session == "20200108"))]  # 4 A/div: no usable current
    return c.to_dict("records")


def extract(row):
    try:
        wf = io.load_csv(row["path"])
    except ValueError:  # CH2 saved with unit "V"
        a = np.genfromtxt(row["path"], delimiter=",", skip_header=18, usecols=(0, 1, 3), invalid_raise=False)
        wf = io.Waveform(t=a[:, 0], v=a[:, 1], i=a[:, 2], meta={})
    if np.corrcoef(wf.i - wf.i.mean(), np.gradient(wf.v))[0, 1] < 0:  # probe reversed: current must lead voltage
        wf = io.Waveform(t=wf.t, v=wf.v, i=-wf.i, meta=wf.meta)
    out = dict(row)
    try:
        f = features.extract(wf)
        out.update(f=f["frequency"], V=f["v_rms"], I=f["i_rms"] * 1e3, C_cell=f["c_total"] * 1e12,
                   zeta=f["c_dielectric"] * 1e12)
        out["cf_C_cell"] = circuit_fit.fit(wf).C_cell * 1e12
    except Exception:  # noqa: BLE001
        pass
    return out


def hours(dev, ses, date):
    if dev == "ATI #1":
        return 867.9 if date.strftime("%Y-%m-%d") == END[dev] else np.nan
    if dev == "ATI #2":
        return ATI2_HOURS.get(ses, np.nan)
    return ATI3_HOURS.get(ses[:4], np.nan)  # not logged after 2020-08-10


def sessions(d):
    d = d[d.f.between(100e3, 115e3)].copy()
    d["date"] = pd.to_datetime(d.time, format="%d-%b-%y %H:%M:%S")
    s = (d.groupby(["device", "session"])
           .agg(date=("date", "min"), note=("note", "first"), n=("file", "size"), f=("f", "median"),
                V=("V", "median"), I=("I", "median"),
                C_cell=("C_cell", "median"), zeta=("zeta", "median"), cf_C_cell=("cf_C_cell", "median"))
           .reset_index().sort_values(["device", "date"]))
    s["days_left"] = [(pd.Timestamp(END[a]) + pd.Timedelta("1D") - t).total_seconds() / 86400
                      for a, t in zip(s.device, s.date)]
    s["hours"] = [hours(a, b, t) for a, b, t in zip(s.device, s.session, s.date)]
    return s


def report(s):
    show = s.assign(session=s.session.str.slice(0, 22), date=s.date.dt.strftime("%Y-%m-%d %H:%M"))
    print("1. sessions\n" + show.round(1).to_string(index=False))
    clean = s[s.note.isin(["", "after IPA cleaning"])]

    print("\n2. H1: Spearman rho vs elapsed time")
    for dev, g in clean.groupby("device"):
        for y in ("C_cell", "zeta", "V"):
            r, p = spearmanr(-g.days_left, g[y])
            print(f"  {dev} {y:6s} n={len(g):2d} rho={r:+.2f} p={p:.2g}  {g[y].iloc[0]:.0f} -> {g[y].iloc[-1]:.0f}")
    print("   common window 0-490 h")
    for dev, g in clean[clean.hours <= 490].groupby("device"):
        print(f"  {dev} n={len(g)}  " + "  ".join(f"{y} rho={spearmanr(g.hours, g[y])[0]:+.2f} (p={spearmanr(g.hours, g[y])[1]:.2f})"
                                               for y in ("C_cell", "zeta", "V")))

    print("\n3. same-day voltage changes, ATI #3")
    g3 = s[s.device == "ATI #3"].set_index("session")
    for a, b in (("0921A (정상파형)", "0923A 13.5V로 파워올린후 측정"), ("0928A (정상파형)", "0928A1 (파형일그러짐)"),
                 ("1007A (정상파형)", "1007A1 (파형일그러짐)")):
        print(f"  {a[:5]} -> {b[:6]}: dV = {g3.V[b] - g3.V[a]:+.0f} V, dC_cell = {g3.C_cell[b] - g3.C_cell[a]:+.1f} pF")

    print("\n4. supply and probe")
    for dev, g in clean.groupby("device"):
        cv = lambda x: 100 * x.std() / x.mean()  # noqa: E731
        print(f"  {dev}: CV I_rms {cv(g.I):.0f} %, V_rms {cv(g.V):.0f} %, C_cell {cv(g.C_cell):.0f} %")
    x3 = clean[clean.device == "ATI #3"]
    print(f"  ATI #3 elasticity d ln V / d ln C_cell = {np.polyfit(np.log(x3.C_cell), np.log(x3.V), 1)[0]:.2f}")
    a = g3.loc[["0313A", "0328A", "0418A"], ["I", "C_cell", "V", "f"]].median()  # 3 sessions before the step
    b = g3.loc[["0520A", "0526A", "0529A"], ["I", "C_cell", "V", "f"]].median()  # 3 sessions after it
    print(f"  probe step (3 sessions before vs after 05-10): dI = {b.I - a.I:+.1f} mA, dC = {b.C_cell - a.C_cell:+.1f} pF, "
          f"dV = {b.V - a.V:+.0f} V; omega dC V = {2 * np.pi * b.f * (b.C_cell - a.C_cell) * 1e-12 * b.V * 1e3:+.1f} mA")
    print(f"  Lissajous vs circuit-model C_cell: median |diff| {np.nanmedian(abs(s.C_cell - s.cf_C_cell)):.1f} pF")

    print("\n5. H3: loaded state = C_cell >= 1.25 x own baseline (median of the first 3 clean sessions;"
          " ATI #3 re-baselined after the 05-02/05-20 probe step)")
    for dev, g in s.groupby("device"):
        g = g.copy()
        g["seg"] = (g.date > pd.Timestamp(ATI3_SETUP_STEP)) if dev == "ATI #3" else False
        for seg, h in g.groupby("seg"):
            base = h[h.note.isin(["", "after IPA cleaning"])].C_cell.head(3).median()
            for r in h[h.C_cell >= 1.25 * base].itertuples():
                print(f"  {dev} {r.session[:22]:22s} {r.date:%Y-%m-%d}  {r.days_left:5.0f} d left  "
                      f"C_cell {r.C_cell:5.1f} / {base:.1f} = {r.C_cell / base:.2f}  V {r.V:.0f}")
        print(f"  {dev}: {len(g)} sessions checked")


def figure(s, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"ATI #1": "#2a78d6", "ATI #2": "#eb6834", "ATI #3": "#1baf7a"}
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "axes.edgecolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df",
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "legend.frameon": False})
    odd = s.note.isin(["distorted", "odd", "13.5 V input"])
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for k, (y, lab, title) in enumerate((("C_cell", "C_cell (Lissajous) [pF]", "(a) load capacitance seen through the probe"),
                                         ("V", "V_rms [V]", "(b) voltage"),
                                         ("I", "I_rms [mA]", "(c) current: the supply acts close to a current source"))):
        a = ax.flat[k]
        for dev, g in s.groupby("device"):
            n, o = g[~odd[g.index]], g[odd[g.index]]
            a.plot(n.days_left, n[y], "-", color=col[dev], lw=1.2, alpha=.55)
            a.plot(n.days_left, n[y], "o", color=col[dev], ms=4.2, mec="white", mew=.8, label=dev)
            a.plot(o.days_left, o[y], "o", mfc="white", mec=col[dev], ms=4.2, mew=1.0)
        a.set_xlim(255, -8)
        a.set(xlabel="days before the discharge stopped", ylabel=lab, title=title)
    ax[0, 0].legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
    a = ax[1, 1]
    for dev, g in s[~odd].groupby("device"):
        a.plot(g.C_cell, g.V, "o", color=col[dev], ms=4.2, mec="white", mew=.8)
    c = np.linspace(33, 115, 50)
    a.plot(c, 1400 * 40 / c, color="#52514e", lw=.8, ls=":")
    a.text(88, 470, "V ∝ 1/C", fontsize=6.8, color="#52514e")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xticks([40, 60, 80, 100], ["40", "60", "80", "100"])
    a.set_yticks([500, 700, 1000, 1500], ["500", "700", "1000", "1500"])
    a.minorticks_off()
    a.set(xlabel="C_cell [pF]", ylabel="V_rms [V]", title="(d) voltage falls because the load grows")
    fig.text(0.01, 0.005, "Hollow: sessions noted as distorted / odd / 13.5 V input. Test data only, no control; "
             "not mixed with the thesis devices.", fontsize=6.3, color="#52514e")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(Path(out) / "ati_aging_test.png", bbox_inches="tight")


def main(root, out):
    with Pool() as p:
        d = pd.DataFrame(p.map(extract, find_captures(root), chunksize=2))
    s = sessions(d)
    report(s)
    figure(s, out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
