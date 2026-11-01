"""What the measured current of M1-M3 is made of: spectrum and distortion of the raw captures (data not distributed here).

    python examples/current_spectrum_check.py "<archive>/전류전압 측정" ["<aging archive>/μ-DBD (large) Aging 데이터 모음"] [docs/img]

Checks the two numbers quoted in step 3 of docs/dbd-physics-primer.ko.md (share of the current power above
1 MHz, total harmonic distortion of the voltage) and what they mean. Every session of M1-M3 holds 10 captures
at 10 ns sampling (5 us/div, 2 periods) followed by 10 at 20 ns (10 us/div, 4 periods), so every number is
split by sampling period. ATI #2 / #3 (aging test, kept separate from the thesis data) are the comparison.

Prints:
  1. the primer's numbers (5 h session, all 20 captures per device)
  2. the same split by sampling period, for the 5 h session and for all sessions, with other definitions
  3. where the >1 MHz power sits: frequency bands and lag-1 autocorrelation (is it resolved by the sampling?)
  4. when it occurs: share of the >1 MHz energy in the half-cycles where |v| rises (discharge phase)
  5. the step in I_rms at the 10 ns -> 20 ns switch, and whether thinning the 10 ns records reproduces it
and writes current_spectrum_check.png.
"""
import re
import sys
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from dbd_diagnostics import electrical, io, waveform

SESSION = re.compile(r"(\d{8})\s*M(\d)\s*(\d+)시간")
F_HF = 1e6  # boundary between the drive harmonics and the discharge pulses / ringing
BANDS = (0, 1e6, 5e6, 10e6, 25e6, np.inf)
ATI_SESSIONS = {"ATI #2": ("20200113",), "ATI #3": ("0313A", "1113A")}  # as in the primer


def nfc(s):
    return unicodedata.normalize("NFC", s)


def find_m(iv_dir):
    jobs = []
    for src in sorted(Path(iv_dir).iterdir()):
        for d in sorted(src.iterdir()) if src.is_dir() else []:
            m = SESSION.match(nfc(d.name))
            if d.is_dir() and m and "오류" not in nfc(d.name):  # "20200718 M3 0시간 - 데이터 자체가 오류"
                jobs += [dict(path=str(f), device=f"M{m.group(2)}", session=m.group(1), hours=int(m.group(3)))
                         for f in sorted(d.glob("ALL*.CSV"))]
    return jobs


def find_ati(root):
    sys.path.insert(0, str(Path(__file__).parent))
    from ati_aging_test import find_captures

    return [dict(path=c["path"], device=c["device"], session=c["session"], hours=np.nan) for c in find_captures(root)
            if c["session"].startswith(ATI_SESSIONS.get(c["device"], ("-",)))]


def load(path):
    try:
        wf = io.load_csv(path)
    except ValueError:  # ATI: CH2 saved with unit "V"
        a = np.genfromtxt(path, delimiter=",", skip_header=18, usecols=(0, 1, 3), invalid_raise=False)
        wf = io.Waveform(t=a[:, 0], v=a[:, 1], i=a[:, 2], meta={})
    if np.corrcoef(wf.i - wf.i.mean(), np.gradient(wf.v))[0, 1] < 0:  # probe reversed: current must lead voltage
        wf = io.Waveform(t=wf.t, v=wf.v, i=-wf.i, meta=wf.meta)
    return wf


def window(t, v, i):
    w = waveform.integer_period_window(t, v - v.mean())
    tt = t[w.slice]
    return tt, v[w.slice] - v[w.slice].mean(), electrical.remove_offset(tt, i[w.slice]), w.period


def split(t, v, i, period):
    """Spectra over an integer number of periods, the current split at F_HF, and the |v|-rising mask."""
    n = len(t)
    F = np.fft.rfftfreq(n, t[1] - t[0])
    X, Y = np.fft.rfft(v), np.fft.rfft(i)
    i_hf = np.fft.irfft(np.where(F > F_HF, Y, 0), n)
    v_lp = np.fft.irfft(np.where(F < 10.5 / period, X, 0), n)  # drive harmonics 1-10: a clean phase reference
    rising = v_lp * np.gradient(v_lp, t) > 0
    return F, np.abs(X) ** 2, np.abs(Y) ** 2, i_hf, rising


def measure(job):
    wf = load(job["path"])
    out = dict(job, file=Path(job["path"]).name, dt_ns=round(wf.dt * 1e9, 1))
    if wf.dt <= 0 or wf.dt > 5e-8:
        return dict(out, skipped="time column coarser than the sampling period" if wf.dt <= 0 else "overview capture")
    try:
        t, v, i, T = window(wf.t, wf.v, wf.i)
    except ValueError:
        return dict(out, skipped="shorter than one period")
    m = int(round((t[-1] - t[0]) / T))  # periods in the window: harmonic h sits in bin h m
    F, Pv, Pi, i_hf, rising = split(t, v, i, T)
    H = np.arange(2, 41) * m
    H = H[H < len(F)]
    w = np.hanning(len(i))
    Ph = np.abs(np.fft.rfft(i * w)) ** 2
    lsb = np.diff(np.unique(np.round(wf.i, 9)))
    stamp = wf.meta.get("Time") or ""
    stamp = stamp[0] if isinstance(stamp, tuple) else stamp
    out.update(
        f=1 / T, periods=m, V=np.sqrt(np.mean(v ** 2)), I=1e3 * np.sqrt(np.mean(i ** 2)),
        I_lf=1e3 * np.sqrt(np.mean((i - i_hf) ** 2)), I_hf=1e3 * np.sqrt(np.mean(i_hf ** 2)),
        hf=Pi[F > F_HF].sum() / Pi.sum(), hf_hann=Ph[F > F_HF].sum() / Ph.sum(),
        hf_05=Pi[F > 0.5e6].sum() / Pi.sum(), hf_5=Pi[F > 5e6].sum() / Pi.sum(),
        # as first computed: all voltage power above 1.5 f over the fundamental bin +-1
        thd_all=np.sqrt(Pv[F > 1.5 / T].sum() / Pv[m - 1:m + 2].sum()),
        thd=np.sqrt(Pv[H].sum() / Pv[m]),  # harmonics 2-40 only
        adc_floor=(lsb[lsb > 0].min() ** 2 / 12) / np.mean(i ** 2),  # quantisation noise / current power
        rising=rising.mean(), hf_rising=(i_hf[rising] ** 2).sum() / (i_hf ** 2).sum(),
        ac1=np.corrcoef(i_hf[rising][:-1], i_hf[rising][1:])[0, 1],
        time=pd.to_datetime(stamp, format="%d-%b-%y %H:%M:%S", errors="coerce"))
    for a, b in zip(BANDS[:-1], BANDS[1:]):
        out[f"band_{a / 1e6:g}"] = Pi[(F > a) & (F <= b)].sum() / Pi.sum()
    if out["dt_ns"] == 10:  # keep every 2nd sample: the 20 ns record a plain sampler would have taken
        t2, v2, i2, _ = window(wf.t[::2], wf.v[::2], wf.i[::2])
        F2, P2 = np.fft.rfftfreq(len(i2), t2[1] - t2[0]), np.abs(np.fft.rfft(i2)) ** 2
        out.update(I_thin=1e3 * np.sqrt(np.mean(i2 ** 2)), hf_thin=P2[F2 > F_HF].sum() / P2.sum())
    return out


def pct(x):
    return (100 * x).round(1)


def report(d, skipped):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    m = d[d.device.str.startswith("M")]
    m5 = m[m.hours == 5]
    print("== 1. primer numbers: 5 h session, all 20 captures per device (median)")
    print(pd.DataFrame({"hf >1 MHz %": pct(m5.groupby("device").hf.median()),
                        "THD (all >1.5 f) %": pct(m5.groupby("device").thd_all.median())}).to_string())

    print("\n== 2. split by sampling period (median); ADC floor = quantisation noise power / current power")
    cols = dict(hf="hf %", hf_hann="hf Hann %", hf_05="hf >0.5 MHz %", hf_5="hf >5 MHz %", thd_all="THD all %",
                thd="THD h2-40 %", adc_floor="ADC floor %")
    for name, g in (("5 h session", m5), ("all sessions", m), ("ATI, comparison sessions", d[d.device.str.startswith("ATI")])):
        t = g.groupby(["device", "dt_ns"])
        tab = pd.concat([pct(t[list(cols)].median()).rename(columns=cols),
                         t[["I", "I_lf", "I_hf"]].median().round(1).rename(columns=lambda c: c + " mA"),
                         t.size().rename("n")], axis=1)
        print(f"-- {name}\n{tab.to_string()}")

    print("\n== 3. where the current power sits, all sessions (% of total, median); "
          "ac1 = lag-1 autocorrelation of the >1 MHz current while |v| rises")
    bands = [c for c in d.columns if c.startswith("band_")]
    names = {c: f"{a / 1e6:g}-{b / 1e6:g} MHz".replace("-inf MHz", " MHz-Nyquist") for c, a, b in zip(bands, BANDS[:-1], BANDS[1:])}
    t = d.groupby(["device", "dt_ns"])
    print(pd.concat([pct(t[bands].median()).rename(columns=names), t.ac1.median().round(2)], axis=1).to_string())

    print("\n== 4. when it occurs, all sessions: share of time with |v| rising (discharge half) and of the >1 MHz energy "
          "in it, % (median)")
    print(pct(t[["rising", "hf_rising"]].median()).rename(columns={"rising": "time", "hf_rising": ">1 MHz energy"}).to_string())

    print("\n== 5. the 10 ns -> 20 ns switch (consecutive captures of one session)")
    ses = m.groupby(["device", "session", "hours"])
    dev = m.I - ses.I.transform("median")
    k = m.file.str.extract(r"(\d+)", expand=False).astype(int)
    pos = k.groupby([m.device, m.session, m.hours, m.dt_ns]).rank()
    print("I_rms minus session median [mA], by position in its block (mean over all sessions):")
    print(dev.groupby([m.dt_ns, pos]).mean().unstack().round(1).to_string())
    blocks = m.groupby(["device", "session", "hours", "dt_ns"]).time.agg(["min", "max"]).unstack("dt_ns")
    span = (blocks[("max", 20.0)] - blocks[("min", 10.0)]).dt.total_seconds() / 60
    print(f"both blocks taken within {span.median():.1f} min (median, max {span.max():.1f} min); "
          f"10 ns block first in {(blocks[('min', 10.0)] < blocks[('min', 20.0)]).mean():.0%} of sessions")
    s = m[m.dt_ns == 10].groupby("device")[["I", "I_thin", "hf", "hf_thin"]].median()
    print("10 ns captures thinned to 20 ns (every 2nd sample):")
    print(pd.concat([s[["I", "I_thin"]].round(1), pct(s[["hf", "hf_thin"]])], axis=1).to_string())

    if len(skipped):
        print("\nskipped:", skipped.groupby(["device", "skipped"]).size().to_dict())


def figure(d, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "axes.edgecolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df",
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "legend.frameon": False})
    ink, col = "#52514e", {"M, 10 ns": "#2a78d6", "M, 20 ns": "#eb6834", "ATI, 4 ns": "#1baf7a"}
    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.15, 1), hspace=0.15, wspace=0.3)
    av, ai, ac = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])

    m5 = d[(d.device == "M1") & (d.hours == 5) & (d.dt_ns == 10)].sort_values("file")
    r = m5.iloc[(m5.I - m5.I.median()).abs().argmin()]  # the most typical 10 ns capture
    wf = load(r.path)
    t, v, i, T = window(wf.t, wf.v, wf.i)
    _, _, _, i_hf, rising = split(t, v, i, T)
    us = (t - t[0]) * 1e6
    for a in (av, ai):
        edges = np.flatnonzero(np.diff(np.r_[0, rising.astype(int), 0]))
        for s0, s1 in zip(edges[::2], edges[1::2]):
            a.axvspan(us[s0], us[min(s1, len(us) - 1)], color="#f1f0ec", lw=0, zorder=0)
        a.set_xlim(us[0], us[-1])
    av.plot(us, v / 1e3, color=ink, lw=0.9)
    av.set_ylabel("v [kV]")
    av.tick_params(labelbottom=False)
    av.set_title(f"M1, 5 h, {r.file} (10 ns); shaded: |v| rising", loc="left")
    ai.plot(us, i * 1e3, color=col["M, 10 ns"], lw=0.5, label="measured i")
    ai.plot(us, (i - i_hf) * 1e3, color=ink, lw=1.1, label="below 1 MHz")
    ai.set_ylabel("i [mA]")
    ai.set_xlabel("time [µs]")
    ai.legend(loc="lower left", ncol=2)

    groups = {"M, 10 ns": d[d.device.str.startswith("M") & (d.hours == 5) & (d.dt_ns == 10)],
              "M, 20 ns": d[d.device.str.startswith("M") & (d.hours == 5) & (d.dt_ns == 20)],
              "ATI, 4 ns": d[d.device.str.startswith("ATI") & (d.dt_ns == 4)]}
    for name, g in groups.items():
        if g.empty:
            continue
        nyq = 0.5 / (g.dt_ns.iloc[0] * 1e-9)
        grid = np.logspace(4, np.log10(nyq), 400)
        curves = []
        for p in g.path:
            wf = load(p)
            t, v, i, T = window(wf.t, wf.v, wf.i)
            F = np.fft.rfftfreq(len(i), t[1] - t[0])
            P = np.abs(np.fft.rfft(i)) ** 2
            curves.append((np.cumsum(P) / P.sum())[np.searchsorted(F, grid, side="right") - 1])  # power in bins <= f
        ac.step(grid / 1e6, 100 * np.median(curves, axis=0), where="post", color=col[name], lw=1.6,
                label=f"{name} (n = {len(curves)})")
        ac.axvline(nyq / 1e6, color=col[name], lw=0.8, ls=":")
    ac.axvline(F_HF / 1e6, color=ink, lw=0.8)
    ac.text(F_HF / 1e6 * 1.1, 3, "1 MHz", color=ink, fontsize=6.8)
    ac.set_xscale("log")
    ac.set_ylim(0, 102)
    ac.set_xlabel("frequency [MHz]  (dotted: Nyquist)")
    ac.set_ylabel("current power below f [%]")
    ac.set_title("cumulative current spectrum (M: 5 h session; median)", loc="left")
    ac.legend(loc="upper left")
    fig.savefig(Path(out) / "current_spectrum_check.png", bbox_inches="tight")


def main(iv_dir, ati_root=None, out="."):
    jobs = find_m(iv_dir) + (find_ati(ati_root) if ati_root else [])
    with Pool() as p:
        d = pd.DataFrame(p.map(measure, jobs))
    skipped = d[d["skipped"].notna()] if "skipped" in d else d.iloc[:0]
    ok = d.drop(skipped.index)
    report(ok, skipped)
    figure(ok, out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
