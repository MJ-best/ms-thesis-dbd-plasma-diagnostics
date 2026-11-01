"""Discharge-pulse detection, number of discharges and onset voltage (thesis Sec. 3.1.4).

Two detectors are provided:

* :func:`thesis_peaks` – the algorithm of the original notebooks
  (``scipy.signal.find_peaks`` + ``peak_prominences``; prominence threshold =
  median prominence x ``w``; IQR outlier removal on peak widths). Index
  handling is fixed: the original code ran it on a slice ``x[T0:T1]`` and then
  used the *slice-relative* indices on the full-cycle arrays.
* :func:`robust_peaks` – noise-referenced detector on the *discharge* current
  (displacement current removed with the equivalent circuit), threshold
  ``k`` x robust noise sigma (MAD). Its pulse count does not depend on the
  distribution of the pulses themselves, so it is comparable between captures.

Both are limited by the measurement chain of the thesis (10 ns sampling, 60 MHz
current probe, 4-8 mA/LSB): ns-scale micro-discharges are under-sampled, so
counts are *lower bounds* and peak currents are attenuated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths


def thesis_peaks(x: np.ndarray, w: float = 1.5) -> np.ndarray:
    """Original two-pass prominence detector (indices relative to ``x``)."""
    peaks, _ = find_peaks(x, width=2)
    if len(peaks) == 0:
        return peaks
    prom = peak_prominences(x, peaks)[0]
    thr = float(np.median(prom)) * w
    new_peaks, _ = find_peaks(x, prominence=thr)
    if len(new_peaks) < 4:
        return new_peaks
    widths = peak_widths(x, new_peaks, rel_height=0.5)[0]
    q1, q3 = np.percentile(widths, [25, 75])
    iqr = q3 - q1
    keep = (widths <= q3 + 1.5 * iqr) & (widths >= q1 - 1.5 * iqr)
    return new_peaks[keep]


def robust_noise_sigma(x: np.ndarray) -> float:
    """Gaussian-equivalent noise level from the median absolute deviation."""
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def robust_peaks(i_dis: np.ndarray, k: float = 5.0, min_distance: int = 2) -> np.ndarray:
    """Peaks of |i_dis| above ``k`` x MAD-sigma (indices into ``i_dis``)."""
    a = np.abs(i_dis)
    sigma = robust_noise_sigma(i_dis)
    if sigma == 0:
        sigma = np.std(i_dis) or 1.0
    peaks, _ = find_peaks(a, height=k * sigma, distance=min_distance)
    return peaks


@dataclass
class HalfCycle:
    polarity: int  # +1 for the rising half (v goes to +V_max), -1 for the falling half
    start: int  # index of the voltage zero crossing that begins the half cycle
    peak: int  # index of the voltage extremum (discharge stops here, thesis T2/T4)
    stop: int  # index of the next zero crossing


def half_cycles(v: np.ndarray) -> list[HalfCycle]:
    """Split one period (starting at a rising zero crossing) into two halves.

    The discharge in a half cycle runs from the onset (after the zero crossing)
    to the voltage extremum; pulses after the extremum belong to the decaying
    tail and are excluded, as in the thesis (T1-T2 and T3-T4).
    """
    n = len(v)
    k_max, k_min = int(np.argmax(v)), int(np.argmin(v))
    # falling zero crossing between max and min
    seg = v[k_max:k_min] if k_min > k_max else v[k_min:k_max]
    zc = np.where(np.diff(np.sign(seg)) != 0)[0]
    mid = (k_max if k_min > k_max else k_min) + (int(zc[0]) + 1 if len(zc) else len(seg) // 2)
    if k_max < k_min:
        return [HalfCycle(+1, 0, k_max, mid), HalfCycle(-1, mid, k_min, n)]
    return [HalfCycle(-1, 0, k_min, mid), HalfCycle(+1, mid, k_max, n)]


@dataclass
class PulseStats:
    n_discharges: float  # mean number of pulses per half cycle (onset -> extremum)
    v_onset: float  # mean |v| at the first pulse of each half cycle (thesis V_min)
    peak_currents: np.ndarray  # |i| at the detected pulses [A] (all half cycles)
    peak_indices: np.ndarray
    onset_indices: list  # index of the first pulse of each half cycle (None if no pulse)
    extremum_indices: list  # index of the voltage extremum of each half cycle


def pulse_statistics(v: np.ndarray, i: np.ndarray, method: str = "robust", w: float = 1.5,
                     k: float = 5.0) -> PulseStats:
    """Count pulses between each zero crossing and the following voltage extremum.

    ``i`` should be the discharge current for ``method='robust'`` and the raw
    total current for ``method='thesis'`` (which works on v*i like the notebooks).
    """
    counts, onsets, peaks_all, on_idx, ext_idx = [], [], [], [], []
    for hc in half_cycles(v):
        sl = slice(hc.start, hc.peak + 1)
        if method == "thesis":
            x = (v * i)[sl]
            p = thesis_peaks(x, w=w)
        elif method == "robust":
            p = robust_peaks(i[sl], k=k)
        else:
            raise ValueError(method)
        p = p + hc.start  # <- the index offset that the original code forgot
        # keep pulses whose current has the polarity of the half cycle
        p = p[np.sign(i[p]) == hc.polarity] if method == "robust" else p
        counts.append(len(p))
        if len(p):
            onsets.append(abs(v[p[0]]))
        on_idx.append(int(p[0]) if len(p) else None)
        ext_idx.append(hc.peak)
        peaks_all.append(p)
    idx = np.concatenate(peaks_all) if peaks_all else np.array([], int)
    return PulseStats(
        n_discharges=float(np.mean(counts)) if counts else 0.0,
        v_onset=float(np.mean(onsets)) if onsets else float("nan"),
        peak_currents=np.abs(i[idx]) if len(idx) else np.array([]),
        peak_indices=idx,
        onset_indices=on_idx,
        extremum_indices=ext_idx,
    )
