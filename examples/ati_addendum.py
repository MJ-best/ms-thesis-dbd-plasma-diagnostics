"""Addendum, not part of the thesis: does the V_rms mechanism hold on the large-area ATI sources? (raw archive, ATI #1-3)

    python examples/ati_addendum.py "<archive>/μ-DBD (large) Aging 데이터 모음" [docs/img]

ATI #1-3 have the cell structure of M1-M3 on a larger area. They ran in nitrogen (1.5 lpm) at about 108 kHz, each on
its own driving power supply (inverter behind a DC 13 V power supply), and were discharged one after another. No
source was measured alongside them as M3 was, so there are no damage labels and no healthy comparison group. What
can be tested is the second half of the mechanism of docs/raw-data-verification.md 10.5, eq. (5): a change of the
load moves V_rms by -S times its relative size, and S is set by the driving power supply and the load.

Prints the tables of docs/ati-addendum.md:
  1. measurement dates and cumulative discharge time (operating logs)
  2. S of each source, with the estimator used for M3
  3. which feature follows the change of the source most precisely (between-date / within-date SD)
  4. equivalent circuit calibrated on the operating point of ATI #2 and the S of ATI #3; the thesis damage and the
     measured change of ATI #3 on it
  5. the equivalent circuit of the thesis sources on a larger area: S and the V_rms signal of damage
Writes ati_addendum.png and ati_circuit.png (the ATI equivalent circuit with simulated and measured waveforms).
"""
import datetime as dt
import re
import sys
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vrms_simulation as vs  # noqa: E402

from dbd_diagnostics import electrical, features, io, waveform  # noqa: E402

SRC = ["ATI #1", "ATI #2", "ATI #3"]
LOGS = {"ATI #1": "1번 ATI/20191209 ATI aging test.xlsx", "ATI #2": "2번 ATI/20200224 NEW ATI Aging Test.xlsx",
        "ATI #3": "3번 ATI/3번 ATI Aging 누적방전시간/20201124 3번쨰 ATI Aging 누적 1541시간.xlsx"}
# ATI #1: only its last week was saved at the drive frequency (earlier folders are 46-79 kHz or copies of other sources)
ATI1 = ["1203", "20191203 ATI 시험분석표/20191203 재측정", "1204A", "1205", "1209-1", "1209-2"]
# ATI #3 folders the experimenter marked: distorted waveform, odd current and voltage, 13.5 V supply input, the
# power-supply voltage alone, and a same-day repeat; "잘못측정한 데이터" (wrongly measured) is a separate folder
ATI3_MARKED = "일그러짐|13.5V|이상|소스 전압|재측정"
# ATI #3: between 05-02 and 05-20 C_total fell by 45 % and I_rms by 18 % while V_rms rose: a change of the
# measurement setup or of the mounting, not of the load alone. S is taken after it.
SETUP_STEP = "2020-05-11"
FEAT = {"v_rms": "V_rms", "i_rms": "I_rms", "frequency": "f", "energy": "E", "c_total": "C_total", "c_dielectric": "C_d",
        "q_max": "ΔQ", "q_per_v": "ΔQ/V_max", "von_frac": "V_on/V_max", "u_b": "U_b"}
S_M3 = 0.52  # docs/raw-data-verification.md 10.5, step 4
M_CIRCUIT = dict(C_d=34.9e-12, C_g=16.7e-12, U_b=898.3, spread=0.153, R=2.126e5, E=4194.0)  # 10.6, fitted to M1-M3
N_ATI = 490  # cell pairs: 126 x C_total(ATI) / C_total(M1-M3) = 126 x 47.5 / 12.15
SCOPE = dict(sample=4e-9, adc_v=40.0, adc_i=1.6e-3, noise_i=1e-3)  # 4 ns, 1 kV/div and 40 mA/div, 8 bit


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


def source(rel):
    """Source of a capture folder, or None when the folder is not used."""
    sub = "/".join(rel[1:])
    if rel[0] == "1번 ATI" and sub in ATI1:
        return "ATI #1"
    if rel[:2] == ["2번 ATI", "2번 ATI 전류전압 데이터"] and len(rel) > 2 and re.match(r"\d{8} ", rel[2]):
        return None if rel[2].startswith("20200108") else "ATI #2"  # 20200108: 4 A/div, no usable current
    if rel[:2] == ["3번 ATI", "3번 ATI 전류전압 측정데이터"] and len(rel) == 3 and re.match(r"\d{4}A", rel[2]):
        return None if re.search(ATI3_MARKED, rel[2]) else "ATI #3"
    return None


def find_captures(root):
    root = Path(root)
    rows = []
    for f in root.rglob("*.[cC][sS][vV]"):
        rel = [nfc(p) for p in f.relative_to(root).parts[:-1]]
        src = source(rel) if rel and not f.name.startswith("._") else None
        if src is None:
            continue
        time, step = header(f)
        if time and step <= 5e-8:  # detail captures only
            rows.append(dict(source=src, folder="/".join(rel[1:]), path=str(f), file=f.name, time=time, step=step))
    # the same capture saved in two folders of one source counts once (copies in other sources' folders are not selected)
    return pd.DataFrame(rows).drop_duplicates(subset=["source", "time", "file"]).to_dict("records")


def load(row):
    try:
        wf = io.load_csv(row["path"])
    except ValueError:  # CH2 saved with the unit "V"
        a = np.genfromtxt(row["path"], delimiter=",", skip_header=18, usecols=(0, 1, 3), invalid_raise=False)
        wf = io.Waveform(t=a[:, 0], v=a[:, 1], i=a[:, 2], meta={})
    if len(np.unique(wf.t)) < 0.9 * len(wf.t):  # time column rounded to too few digits: rebuild it
        wf = io.Waveform(t=np.arange(len(wf.v)) * row["step"], v=wf.v, i=wf.i, meta=wf.meta)
    if np.corrcoef(wf.i - wf.i.mean(), np.gradient(wf.v))[0, 1] < 0:  # current probe reversed: current leads voltage
        wf = io.Waveform(t=wf.t, v=wf.v, i=-wf.i, meta=wf.meta)
    return wf


def extract(row):
    out = dict(row)
    wf = load(row)
    try:
        f = features.extract(wf)
    except ValueError:  # no full period, or a Lissajous branch too short to fit
        return out
    out.update({k: f[k] for k in ("v_rms", "i_rms", "frequency", "energy", "c_total", "c_dielectric", "q_max", "v_max",
                                  "v_on_lissajous", "u_b")})
    return out


def operating_log(path):
    """Cumulative discharge hours from an operating log: (time, hours) points.

    Daily rows give date, start, end and duration. Later sheets add period rows ("a ~ b", "N2-k 방전시간", duration);
    the last sheet lists all periods, so daily rows are used only before the first period."""
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    day = lambda x: dt.datetime.strptime(str(x), "%Y%m%d")  # noqa: E731
    daily, periods = {}, []
    for k, ws in enumerate(wb.worksheets):
        for r in ws.iter_rows(values_only=True):
            r = [v for v in r if v is not None]
            if len(r) >= 4 and isinstance(r[0], int) and isinstance(r[1], dt.time) and isinstance(r[3], dt.time):
                daily[r[0]] = r[3].hour + r[3].minute / 60 + r[3].second / 3600
            elif (k == len(wb.worksheets) - 1 and len(r) >= 3 and isinstance(r[1], str) and "방전시간" in r[1]
                  and isinstance(r[2], dt.timedelta)):
                a, _, b = str(r[0]).partition("~")
                periods.append((day(a.strip()), day((b or a).strip()), r[2].total_seconds() / 3600))
    first = min((p[0] for p in periods), default=dt.datetime.max)
    spans = sorted([(day(d), day(d), h) for d, h in daily.items() if day(d) < first] + periods)
    pts, total = [], 0.0
    for a, b, h in spans:
        pts += [(a, total), (b + dt.timedelta(days=1), total + h)]
        total += h
    return pts


def hours_at(pts, when):
    x = np.array([pd.Timestamp(p[0]).timestamp() for p in pts])
    return float(np.interp(pd.Timestamp(when).timestamp(), x, [p[1] for p in pts]))


def dates(d, logs):
    d = d[d.v_rms.notna() & d.frequency.between(100e3, 115e3)].copy()
    d["when"] = pd.to_datetime(d.time, format="%d-%b-%y %H:%M:%S")
    d["date"] = d.when.dt.strftime("%Y-%m-%d")
    d["q_per_v"], d["von_frac"] = d.q_max / d.v_max, d.v_on_lissajous / d.v_max
    s = d.groupby(["source", "date"]).agg(n=("file", "size"), when=("when", "min"),
                                          **{k: (k, "median") for k in list(FEAT) + ["v_max"]}).reset_index()
    s["hours"] = [hours_at(logs[a], t) for a, t in zip(s.source, s.when)]
    s["after_step"] = (s.source != "ATI #3") | (s.date > SETUP_STEP)
    return d, s


def load_sensitivity(g, n_boot=2000, seed=0):
    """S = -1 / (dln(ΔQ/V_max)/dlnV) over measurement dates, as for M3 (V is the precise variable), with a
    bootstrap over dates."""
    slope = lambda a: np.polyfit(np.log(a.v_rms), np.log(a.q_per_v), 1)[0]  # noqa: E731
    lo = hi = np.nan
    if len(g) >= 6:  # with fewer dates a resample is mostly repeats of one or two dates
        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(n_boot):
            b = g.iloc[rng.integers(0, len(g), len(g))]
            if b.date.nunique() > 2:
                boot.append(-1 / slope(b))
        lo, hi = np.percentile(boot, [2.5, 97.5])
    r2 = np.corrcoef(np.log(g.v_rms), np.log(g.q_per_v))[0, 1] ** 2
    di = np.polyfit(np.log(g.q_per_v), np.log(g.i_rms), 1)[0]
    return dict(n=len(g), S=-1 / slope(g), lo=lo, hi=hi, r2=r2, dlnI=di, range_v=np.ptp(np.log(g.v_rms)),
                range_q=np.ptp(np.log(g.q_per_v)))


def precision(d, s):
    """Between-date SD of the date medians over the pooled within-date SD, both of the log feature."""
    lg = np.log(d[list(FEAT)].where(d[list(FEAT)] > 0)).assign(date=d.date)
    within = lg.groupby("date")[list(FEAT)].var().mean() ** .5
    between = lg.groupby("date")[list(FEAT)].median().std()
    return (between / within).sort_values(ascending=False), within


# ---- equivalent circuit ------------------------------------------------------------------------------------------

def simulate(p, f, n=N_ATI, dev=None, seeds=(1,)):
    dev = dev or vs.device(p["C_d"], p["C_g"], p["U_b"], p["spread"], n=n)
    rows = []
    for s in seeds:
        t, v, Q = vs.run(dev, E=p["E"], R=p["R"], f=f, seed=s)
        rows += vs.measure(t, v, Q, seed=s + 50, **SCOPE)
    return pd.DataFrame(rows).median(numeric_only=True)


def sensitivity(p, f, n):
    """S of the circuit: -dlnV/dln(ΔQ/V_max) for a 3 % heavier load (as in vrms_simulation.fit)."""
    dev = vs.device(p["C_d"], p["C_g"], p["U_b"], p["spread"], n=n)
    h = vs.truth(*vs.run(dev, E=p["E"], R=p["R"], f=f))
    e = vs.truth(*vs.run(dict(dev, Cd=dev["Cd"] * 1.03, Cg=dev["Cg"] * 1.03), E=p["E"], R=p["R"], f=f))
    return -np.log(e["v_rms"] / h["v_rms"]) / np.log(e["q_per_v"] / h["q_per_v"])


def calibrate(target, S, f):
    """Source (C_d, C_g, U_b, spread) and driving power supply (R, E) from a healthy operating point and S."""
    keys = ["v_rms", "v_max", "c_total", "c_dielectric", "q_per_v", "von_frac", "u_b", "energy"]

    def unpack(x):
        return dict(C_d=np.exp(x[0]) * 1e-12, C_g=np.exp(x[1]) * 1e-12, U_b=x[2], spread=abs(x[3]), R=np.exp(x[4]) * 1e3,
                    E=x[5])

    def loss(x):
        p = unpack(x)
        if not (100 < p["U_b"] < 2000 and 1e3 < p["R"] < 1e7 and p["E"] > 500):
            return 1e6
        o = simulate(p, f)
        if not np.isfinite(o[keys]).all():
            return 1e6
        w = {"v_rms": 3, "v_max": 3}
        return (sum(w.get(k, 1) * ((o[k] - target[k]) / target[k]) ** 2 for k in keys)
                + 3 * ((sensitivity(p, f, N_ATI) - S) / S) ** 2)

    res = minimize(loss, [np.log(62), np.log(230), 345, 0.16, np.log(78), 6300], method="Nelder-Mead",
                   options=dict(maxfev=160, xatol=1e-3, fatol=1e-4))
    p = unpack(res.x)
    o = simulate(p, f, seeds=(1, 2))
    o["S"] = sensitivity(p, f, N_ATI)
    return p, pd.DataFrame({"data": pd.Series({k: target[k] for k in keys} | {"S": S}), "model": o[keys + ["S"]]})


def ati_cases(p, f, k, u):
    """Changes of the source on the calibrated circuit, without a fit (noise-free waveforms): the thesis damage
    (exposed electrode, Townsend factor 0.83) in every cell; the discharge initiation level x u; all capacitances
    x k; both, with k and u as measured on ATI #3. "discharging": charge per volt above the no-discharge C_t."""
    dev = vs.device(p["C_d"], p["C_g"], p["U_b"], p["spread"], n=N_ATI)
    c_t = p["C_d"] * p["C_g"] / (p["C_d"] + p["C_g"])
    cases = {"exposed electrode, every cell (thesis damage)": (vs.exposed(dev, N_ATI), 1.0),
             f"discharge initiation level ×{u:.2f}": (dict(dev, Ub=dev["Ub"] * u), 1.0),
             f"capacitance ×{k:.2f}": (dict(dev, Cd=dev["Cd"] * k, Cg=dev["Cg"] * k), k),
             f"capacitance ×{k:.2f} and initiation level ×{u:.2f}": (dict(dev, Cd=dev["Cd"] * k, Cg=dev["Cg"] * k,
                                                                         Ub=dev["Ub"] * u), k)}
    h = vs.truth(*vs.run(dev, E=p["E"], R=p["R"], f=f))
    rows = []
    for name, (d, kc) in cases.items():
        o = vs.truth(*vs.run(d, E=p["E"], R=p["R"], f=f))
        rows.append(dict(case=name, v_rms=100 * (o["v_rms"] / h["v_rms"] - 1), v_max=o["v_max"] / h["v_max"],
                         q_per_v=o["q_per_v"] / h["q_per_v"], discharging=o["q_per_v"] / (c_t * kc) > 1.02))
    return pd.DataFrame(rows)


def area_scaling(areas=(1, 2, 4, 8), vmax=2243.0):
    """The thesis sources' circuit (10.6) on a larger area with the same driving power supply output impedance; E raised
    to keep V_max. S, and the V_rms change for 48 exposed cells (fixed damaged area) and for 38 % of the cells."""
    p = M_CIRCUIT
    rows = []
    for a in areas:
        n = int(round(126 * a))
        dev = vs.device(p["C_d"] * a, p["C_g"] * a, p["U_b"], p["spread"], n=n)
        E = p["E"]
        for _ in range(4):
            E *= vmax / vs.truth(*vs.run(dev, E=E, R=p["R"]))["v_max"]
        q = dict(p, C_d=p["C_d"] * a, C_g=p["C_g"] * a, E=E)
        h = vs.truth(*vs.run(dev, E=E, R=p["R"]))
        dv = lambda k: 100 * (vs.truth(*vs.run(vs.exposed(dev, k), E=E, R=p["R"]))["v_rms"] / h["v_rms"] - 1)  # noqa: E731
        rows.append(dict(area=a, n=n, E=E, S=sensitivity(q, vs.F0, n), cells48=dv(48), share38=dv(int(round(48 * a)))))
    return pd.DataFrame(rows)


def main(root, out="."):
    pd.set_option("display.width", 200)
    logs = {k: operating_log(Path(root) / v) for k, v in LOGS.items()}
    with Pool() as pool:
        d = pd.DataFrame(pool.map(extract, find_captures(root), chunksize=2))
    d, s = dates(d, logs)

    print("1. measurement dates (medians) and cumulative discharge time from the operating logs")
    for k in SRC:
        g = s[s.source == k]
        print(f"  {k}: logged {logs[k][-1][1]:.0f} h ({logs[k][0][0]:%Y-%m-%d} - {logs[k][-1][0] - dt.timedelta(days=1):%Y-%m-%d}); "
              f"{len(g)} dates with captures, {g.n.sum()} captures, {g.hours.min():.0f}-{g.hours.max():.0f} h")
    show = s.assign(i_rms=s.i_rms * 1e3, q_per_v=s.q_per_v * 1e12, c_total=s.c_total * 1e12, frequency=s.frequency / 1e3)
    print(show[["source", "date", "hours", "n", "v_rms", "i_rms", "q_per_v", "von_frac", "c_total", "frequency", "after_step"]]
          .round(2).to_string(index=False))

    print(f"\n2. load sensitivity S = -1 / (dln(ΔQ/V_max)/dlnV) over measurement dates (M3: {S_M3}); 95 % bootstrap over dates")
    sens = {}
    for name, g in (("ATI #1", s[s.source == "ATI #1"]), ("ATI #2", s[s.source == "ATI #2"]),
                    ("ATI #2 without the first date", s[(s.source == "ATI #2") & (s.date > "2019-12-11")]),
                    ("ATI #3 after the setup step", s[(s.source == "ATI #3") & s.after_step]),
                    ("ATI #3 all dates", s[s.source == "ATI #3"])):
        r = sens[name] = load_sensitivity(g)
        ci = f"[{r['lo']:.2f}, {r['hi']:.2f}]" if np.isfinite(r["lo"]) else "(too few dates)"
        print(f"  {name:28s} n {r['n']:2d}  S {r['S']:.2f} {ci:15s}  R² {r['r2']:.2f}  "
              f"range lnV {r['range_v']:.2f}, ln(ΔQ/V) {r['range_q']:.2f}  dlnI_rms/dln(ΔQ/V) {r['dlnI']:+.2f}")

    print("\n3. which feature follows the change of the source: between-date SD / within-date SD (log), within-date noise")
    ranks = {}
    for name, keep in (("ATI #2", (d.source == "ATI #2")), ("ATI #3 after the setup step", (d.source == "ATI #3") & (d.date > SETUP_STEP))):
        r, noise = ranks[name] = precision(d[keep], s)
        print(f"  {name}: " + ", ".join(f"{FEAT[k]} {v:.1f} ({100 * noise[k]:.1f} %)" for k, v in r.items()))

    ref = d[d.source == "ATI #2"][list(FEAT) + ["v_max"]].median()
    f0 = float(ref.frequency)
    print(f"\n4. equivalent circuit, {N_ATI} cell pairs at {f0 / 1e3:.1f} kHz, calibrated on the operating point of ATI #2 "
          "(medians of all captures) and the S of ATI #3 (the only source whose load changed enough to fix S)")
    p, calib = calibrate(ref, sens["ATI #3 after the setup step"]["S"], f0)
    print("  parameters: " + ", ".join(f"{k} {v * 1e12:.1f} pF" if k in ("C_d", "C_g") else f"{k} {v:.4g}" for k, v in p.items()))
    c = calib.assign(**{"model/data": calib.model / calib.data})
    pf = [k for k in c.index if k.startswith(("c_", "q_"))]
    c.loc[pf, ["data", "model"]] *= 1e12
    c.loc["energy", ["data", "model"]] *= 1e6
    print("  (pF, µJ)\n" + c.round(3).to_string())
    a3 = s[(s.source == "ATI #3") & s.after_step].assign(u_min=lambda x: x.v_max * (1 + x.von_frac) / 2)  # eq. (1)
    first, last = a3.head(3).median(numeric_only=True), a3.tail(3).median(numeric_only=True)
    k, u = last.c_total / first.c_total, last.u_min / first.u_min
    print(f"  ATI #3, first three vs last three dates after the setup step: C_total x{k:.2f}, discharge initiation level "
          f"U_min = (V_max + V_on)/2 {first.u_min:.0f} -> {last.u_min:.0f} V (x{u:.2f}), V_on/V_max {first.von_frac:.2f} -> {last.von_frac:.2f}")
    # why the Lissajous graph changed: slopes (C_total, C_d), height (ΔQ), width (V_max), area (E = 2 U_b ΔQ, eq. (2))
    manley = lambda x: 4 * x.c_dielectric * x.u_b * (x.v_max - x.u_min)  # noqa: E731
    lq = pd.DataFrame({"first 3": first, "last 3": last}).loc[["c_total", "c_dielectric", "q_max", "v_max", "u_b", "energy", "i_rms"]]
    lq.loc["energy (Manley, eq. 2)"] = [manley(first), manley(last)]
    lq["ratio"] = lq["last 3"] / lq["first 3"]
    unit = {"c_total": 1e12, "c_dielectric": 1e12, "q_max": 1e9, "energy": 1e6, "energy (Manley, eq. 2)": 1e6, "i_rms": 1e3}
    for key, scale in unit.items():
        lq.loc[key, ["first 3", "last 3"]] *= scale
    print("  Lissajous graph of ATI #3 (pF, nC, V, µJ, mA):\n" + lq.round(2).to_string())
    print("  Paschen burning voltage of a 100 µm gap in nitrogen (A 12, B 342): " + ", ".join(
        f"γ {g}: {vs.paschen(g, A=12.0, B=342.0):.0f} V" for g in (0.01, 0.1)))
    cases = ati_cases(p, f0, k, u)
    cases.loc[len(cases)] = dict(case="data ATI #3", v_rms=100 * (last.v_rms / first.v_rms - 1), v_max=last.v_max / first.v_max,
                                 q_per_v=last.q_per_v / first.q_per_v, discharging=True)
    print("  predicted change (no fit; V_max and ΔQ/V_max as ratios):\n" + cases.round(3).to_string(index=False))
    d3 = d[(d.source == "ATI #3") & (d.date > SETUP_STEP)]
    reps = []  # the capture closest to the median V_rms of the first and of the last date after the setup step
    for day in (a3.date.iloc[0], a3.date.iloc[-1]):
        g = d3[d3.date == day]
        reps.append(g.loc[(g.v_rms - g.v_rms.median()).abs().idxmin()].to_dict() | {"hours": a3.set_index("date").hours[day]})

    print("\n5. the thesis sources' circuit on a larger area, same driving power supply output impedance, V_max held at 2243 V")
    area = area_scaling()
    print(area.round(2).to_string(index=False))
    figure(s, sens, ranks, cases, area, out)
    figure_circuit(p, f0, k, u, reps, out)


def figure(s, sens, ranks, cases, area, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, AQUA, YELLOW, INK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.8, "axes.edgecolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": GREY,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "legend.frameon": False})
    fig, ax = plt.subplots(2, 2, figsize=(8, 7))

    a = ax[0, 0]  # (a) V_rms against the load, each source around its own mean
    for name, key, col in (("ATI #1", "ATI #1", BLUE), ("ATI #2", "ATI #2", ORANGE), ("ATI #3", "ATI #3 after the setup step", AQUA)):
        g = s[(s.source == name) & s.after_step]
        x, y = np.log(g.q_per_v), np.log(g.v_rms)
        x, y = x - x.mean(), y - y.mean()
        r = sens[key]
        ci = f" [{r['lo']:.2f}, {r['hi']:.2f}]" if np.isfinite(r["lo"]) else ""
        a.plot(x, y, "o", color=col, ms=5, mec="white", mew=.7, label=f"{name}: S {r['S']:.2f}{ci}, {r['n']} dates")
    xs = np.linspace(-.6, .7, 2)
    for S, ls, lab in ((S_M3, ":", f"thesis sources (M3): S {S_M3}"), (1.0, "--", "current source: S 1")):
        a.plot(xs, -S * xs, ls, color=INK, lw=.9, label=lab)
    a.set(xlabel="ln(ΔQ/V_max), around the source's mean", ylabel="ln V_rms, around the source's mean",
          title=f"(a) V_rms falls as the load grows (ATI #3: S {sens['ATI #3 after the setup step']['S']:.2f})")
    a.set_ylim(top=1.05)  # room for the legend above the points
    a.legend(loc="upper right")

    a = ax[0, 1]  # (b) precision ranking on ATI #3
    r = ranks["ATI #3 after the setup step"][0][::-1]
    y = np.arange(len(r))
    a.barh(y, r.values, color=[BLUE if k == "v_rms" else GREY for k in r.index], height=.62)
    a.set_yticks(y, [FEAT[k] for k in r.index])
    for yy, v in zip(y, r.values):
        a.text(v + .3, yy, f"{v:.1f}", va="center", fontsize=6.8, color=INK)
    a.set(xlabel="between-date SD / within-date SD", title="(b) ATI #3: V_rms follows the change most precisely")
    a.grid(axis="y", visible=False)
    a.set_axisbelow(True)

    a = ax[1, 0]  # (c) changes of the source on the calibrated circuit vs ATI #3
    y = np.arange(len(cases))[::-1]
    a.axvline(0, color=INK, lw=.6)
    for yy, r in zip(y, cases.itertuples()):
        data = r.case.startswith("data")
        col = AQUA if data else BLUE
        a.plot([0, r.v_rms], [yy, yy], color=col, lw=1.4, alpha=.5)
        a.plot(r.v_rms, yy, "o", ms=6, mew=1.2, mec=col, mfc=col if r.discharging else "white")
        a.text(r.v_rms, yy - .3, f"{r.v_rms:+.0f} %" + ("" if r.discharging else ", discharge stops"), ha="center",
               va="top", fontsize=6.8, color=INK)
    a.set_yticks(y, [c.replace(" and ", "\nand ").replace(", every cell", ",\nevery cell") for c in cases.case])
    a.set(xlim=(-75, 10), ylim=(-.8, len(cases) - .5), xlabel="V_rms change [%]", title="(c) ATI circuit: thesis damage vs ATI #3")
    a.grid(axis="y", visible=False)

    a = ax[1, 1]  # (d) area scaling with the thesis driving power supply
    a.axhline(0, color=INK, lw=.6)
    a.plot(area.area, area.share38, "-o", color=BLUE, ms=5, mec="white", mew=.7, label="38 % of cells exposed (same share)")
    a.plot(area.area, area.cells48, "-s", color=ORANGE, ms=5, mec="white", mew=.7, label="48 cells exposed (same area)")
    a.set_xscale("log", base=2)
    a.set_xticks(area.area, [f"×{a:g}\nS {S:.2f}" for a, S in zip(area.area, area.S)])
    a.minorticks_off()
    a.set(xlabel="area of the thesis source (S of its driving power supply and load)", ylabel="V_rms change [%]",
          title="(d) The thesis sources' circuit on a larger area")
    a.legend(loc="lower right", bbox_to_anchor=(1, .3))
    fig.text(0.01, 0.005, "Addendum, not part of the thesis. ATI sources: no healthy comparison group and no damage labels.",
             fontsize=6.3, color=INK)
    fig.tight_layout(rect=(0, .02, 1, 1))
    fig.savefig(Path(out) / "ati_addendum.png", bbox_inches="tight")


def figure_circuit(p, f, k, u, reps, out):
    """The ATI equivalent circuit, the applied and air-gap voltages and the Lissajous graphs it gives for the healthy
    source and the change of ATI #3, next to the measured ATI #3 captures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

    BLUE, ORANGE, AQUA, INK, GREY, PALE = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#e4e3df", "#f4f3ef"
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.6, "axes.edgecolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": GREY,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "legend.frameon": False})
    dev = vs.device(p["C_d"], p["C_g"], p["U_b"], p["spread"], n=N_ATI)
    big = dict(dev, Cd=dev["Cd"] * k, Cg=dev["Cg"] * k)
    cell = int(np.argmin(np.abs(np.log(dev["Ub"] / np.median(dev["Ub"])))))  # a cell with the median burning voltage
    sims = []
    for lab, dd, col, ls in (("healthy (calibrated)", dev, INK, "-"), (f"C ×{k:.2f}", big, ORANGE, "--"),
                             (f"C ×{k:.2f}, U_b ×{u:.2f}", dict(big, Ub=dev["Ub"] * u), BLUE, "-")):
        t, v, Q, U = vs.run(dd, E=p["E"], R=p["R"], f=f, trace=[cell])
        w = waveform.one_period_window(t, v)
        ph = 360 * (t[w.slice] - t[w.start]) / w.period
        sims.append(dict(lab=lab, col=col, ls=ls, ph=ph, v=v[w.slice], q=Q[w.slice], ug=U[w.slice, 0], ub=dd["Ub"][cell]))
    meas = []
    for r, ls in zip(reps, ("--", "-")):
        wf = load(r)
        w = waveform.one_period_window(wf.t, wf.v)
        t, v, i = wf.t[w.slice], wf.v[w.slice], wf.i[w.slice]
        i = electrical.remove_offset(t, i)
        meas.append(dict(lab=f"ATI #3, {r['date']} ({r['hours']:.0f} h)", ls=ls, ph=360 * (t - t[0]) / w.period,
                         v=v - v.mean(), q=electrical.charge(t, i, center=True)))

    fig = plt.figure(figsize=(8, 9.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.05])
    ax_c = fig.add_subplot(gs[0, :])
    ax_v, ax_u, ax_s, ax_m = (fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 0]),
                              fig.add_subplot(gs[2, 1]))

    a = ax_c  # (a) circuit
    a.set(xlim=(0, 10), ylim=(0, 4.4), title=f"(a) ATI equivalent circuit: driving power supply and {N_ATI} discharge cells in parallel")
    a.axis("off")
    line = lambda xs, ys, **kw: a.plot(xs, ys, color=kw.pop("color", INK), lw=kw.pop("lw", 1.2), solid_capstyle="round", **kw)  # noqa: E731

    def cap(x, y, label):
        line([x, x], [y + .32, y + .08]); line([x - .22, x + .22], [y + .08, y + .08], lw=1.8)
        line([x - .22, x + .22], [y - .08, y - .08], lw=1.8); line([x, x], [y - .08, y - .32])
        a.text(x - .3, y, label, ha="right", va="center", fontsize=6.6, color=INK)

    def draw_cell(x):
        line([x, x], [3.7, 3.35]); cap(x, 3.03, "C_d,i"); line([x, x], [2.71, 2.45])
        line([x - .45, x + .45], [2.45, 2.45]); line([x - .45, x + .45], [1.25, 1.25])
        line([x - .45, x - .45], [2.45, 2.17]); line([x - .45, x - .45], [1.53, 1.25]); cap(x - .45, 1.85, "C_g,i")
        a.add_patch(Rectangle((x + .2, 1.55), .5, .6, fc="white", ec=INK, lw=1.2))
        a.text(x + .45, 1.85, "±U_b,i", ha="center", va="center", fontsize=6.2, color=INK)
        line([x + .45, x + .45], [2.45, 2.15]); line([x + .45, x + .45], [1.55, 1.25]); line([x, x], [1.25, .7])

    a.add_patch(FancyBboxPatch((.2, .55), 2.75, 3.55, boxstyle="round,pad=.05", fc=PALE, ec=GREY, lw=.8))
    a.text(1.57, 4.18, "driving power supply (Thévenin)", ha="center", fontsize=6.8, color=INK)
    a.add_patch(Circle((.85, 2.2), .38, fc="white", ec=INK, lw=1.2))
    a.text(.85, 2.2, "~", ha="center", va="center", fontsize=13, color=INK)
    a.text(1.32, 2.2, f"E sin ωt\n{p['E'] / 1e3:.1f} kV\n{f / 1e3:.0f} kHz", ha="left", va="center", fontsize=6.6, color=INK)
    line([.85, .85], [2.58, 3.7]); line([.85, 1.35], [3.7, 3.7])
    line(np.linspace(1.35, 2.45, 13), 3.7 + .13 * np.array([0, 1, -1] * 4 + [0]))
    a.text(1.9, 3.38, f"R ≈ {p['R'] / 1e3:.0f} kΩ", ha="center", fontsize=6.6, color=INK)
    line([2.45, 9.4], [3.7, 3.7]); line([.85, .85], [1.82, .7]); line([.85, 9.4], [.7, .7])
    a.plot(3.25, 3.7, "o", color=INK, ms=4)
    a.text(3.25, 4.0, "V (HV probe)", ha="center", fontsize=6.6, color=INK)
    a.add_patch(FancyArrowPatch((3.5, 3.55), (4.05, 3.55), arrowstyle="-|>", mutation_scale=8, color=INK, lw=.8))
    a.text(3.78, 3.3, "I", ha="center", fontsize=6.6, color=INK)
    for x in (4.75, 6.85, 9.1):
        draw_cell(x)
    a.text(7.95, 3.0, "· · ·", ha="center", va="center", fontsize=12, color=INK)
    a.text(7.0, 4.05, f"{N_ATI} cell pairs in parallel (Manley), C_total ≈ 3.9 × the thesis sources; U_b,i spread over the cells\n"
           "a discharging cell holds its air-gap voltage at ±U_b,i", ha="center", va="bottom", fontsize=6.6, color=INK)
    a.text(6.85, .3, f"ATI #3 after ~1,000 h, measured: C_total ×{k:.2f}, discharge initiation threshold U_min ×{u:.2f}\n"
           f"→ in the model every cell: C_d,i and C_g,i ×{k:.2f}, U_b,i ×{u:.2f}", ha="center", va="center", fontsize=6.6,
           color=BLUE)

    a = ax_v  # (b) applied voltage over one period
    for sm in sims:
        a.plot(sm["ph"], sm["v"] / 1e3, ls=sm["ls"], color=sm["col"], lw=1.3, label=f"model: {sm['lab']}")
    for m in meas:
        a.plot(m["ph"], m["v"] / 1e3, ls=m["ls"], color=AQUA, lw=1.1, label=m["lab"])
    a.set(xlim=(0, 360), xticks=range(0, 361, 90), xlabel="phase [°]", ylabel="applied voltage [kV]",
          title="(b) Applied voltage: the capacitance growth halves it")
    a.legend(loc="lower left", fontsize=6)

    a = ax_u  # (c) air-gap voltage of one cell
    for sm in sims:
        a.plot(sm["ph"], sm["ug"], ls=sm["ls"], color=sm["col"], lw=1.3, label=sm["lab"])
    for sm, txt in ((sims[0], "U_b,i"), (sims[2], f"U_b,i ×{u:.2f}")):
        a.axhline(sm["ub"], color=sm["col"], lw=.8, ls=":")
        a.text(358, sm["ub"] - 12, txt, ha="right", va="top", fontsize=6.4, color=sm["col"])
    a.set(xlim=(0, 360), xticks=range(0, 361, 90), xlabel="phase [°]", ylabel="air-gap voltage of one cell [V]",
          title="(c) One cell: without a lower threshold it never discharges")
    a.legend(loc="lower left", fontsize=6)

    a = ax_s  # (d), (e) Lissajous graphs on the same axes
    for sm in sims:
        a.plot(sm["v"] / 1e3, sm["q"] * 1e9, ls=sm["ls"], color=sm["col"], lw=1.3, label=sm["lab"])
    a.set(xlabel="applied voltage [kV]", ylabel="charge [nC]", title="(d) Lissajous graph, model")
    a.legend(loc="upper left", fontsize=6)
    b = ax_m
    for m in meas:
        b.plot(m["v"] / 1e3, m["q"] * 1e9, ls=m["ls"], color=AQUA, lw=1.2, label=m["lab"])
    b.set(xlabel="applied voltage [kV]", ylabel="charge [nC]", title="(e) Lissajous graph, ATI #3 measured")
    b.legend(loc="upper left", fontsize=6)
    lim_v = 1.08 * max(np.abs(x["v"]).max() for x in sims + meas) / 1e3
    lim_q = 1.12 * max(np.abs(x["q"]).max() for x in sims + meas) * 1e9
    for ax in (a, b):
        ax.set(xlim=(-lim_v, lim_v), ylim=(-lim_q, lim_q))
    fig.text(.01, .004, f"Model: calibrated on the operating point of ATI #2 and the S of ATI #3 (C_d {p['C_d'] * 1e12:.0f} pF, "
             f"C_g {p['C_g'] * 1e12:.0f} pF, U_b {p['U_b']:.0f} V); noise-free. Measured: the capture nearest to the median V_rms "
             "of the first and last date after the setup change.", fontsize=6, color=INK)
    fig.tight_layout(rect=(0, .02, 1, 1))
    fig.savefig(Path(out) / "ati_circuit.png", bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ".")
