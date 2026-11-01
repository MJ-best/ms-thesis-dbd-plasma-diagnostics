"""Period detection and integer-period windowing.

All cycle-averaged quantities (RMS, energy, charge) must be evaluated over an
*integer number of periods*; otherwise RMS values are biased and the current
integral picks up the probe/ADC offset as a linear drift. For the thesis data
(8-bit ADC, 8 mA/LSB on the current channel, 50 us record) an uncorrected
offset of a single LSB integrates to 8 mA x 21.5 us ~ 170 nC per period, i.e.
~3x the discharge charge (~60 nC). Offset removal is therefore not optional.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PeriodWindow:
    start: int  # index of first sample (rising zero crossing of v)
    stop: int  # index one past the last sample (next rising zero crossing)
    period: float  # [s]

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def frequency(self) -> float:
        return 1.0 / self.period


def rising_zero_crossings(t: np.ndarray, v: np.ndarray, hysteresis: float | None = None) -> np.ndarray:
    """Interpolated times of rising zero crossings of ``v - mean(v)``.

    A Schmitt-trigger style hysteresis (default: 10 % of the amplitude) makes
    the detection robust against quantisation noise and discharge ripple.
    """
    x = v - np.mean(v)
    amp = 0.5 * (np.max(x) - np.min(x))
    h = 0.1 * amp if hysteresis is None else hysteresis
    out = []
    armed = False
    last_neg = 0
    for k in range(len(x)):
        if x[k] < -h:
            armed = True
            last_neg = k
        elif armed and x[k] >= 0:
            # linear interpolation between the last sample < 0 and this one
            j = k - 1
            while j > last_neg and x[j] >= 0:
                j -= 1
            x0, x1 = x[j], x[j + 1]
            frac = 0.0 if x1 == x0 else -x0 / (x1 - x0)
            out.append(t[j] + frac * (t[j + 1] - t[j]))
            armed = False
    return np.asarray(out)


def estimate_frequency(t: np.ndarray, v: np.ndarray) -> float:
    """Driving frequency [Hz] from rising zero crossings (falls back to FFT)."""
    zc = rising_zero_crossings(t, v)
    if len(zc) >= 2:
        return 1.0 / float(np.mean(np.diff(zc)))
    # fallback: FFT peak with parabolic interpolation
    x = v - np.mean(v)
    n = len(x)
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    k = int(np.argmax(spec[1:])) + 1
    if 1 <= k < len(spec) - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    dt = float(np.median(np.diff(t)))
    return k / (n * dt)


def one_period_window(t: np.ndarray, v: np.ndarray, which: int = 0) -> PeriodWindow:
    """Sample window covering exactly one period, starting at a rising zero crossing."""
    zc_idx = []
    x = v - np.mean(v)
    amp = 0.5 * (np.max(x) - np.min(x))
    h = 0.1 * amp
    armed = False
    for k in range(len(x)):
        if x[k] < -h:
            armed = True
        elif armed and x[k] >= 0:
            zc_idx.append(k)
            armed = False
    if len(zc_idx) < which + 2:
        raise ValueError("record shorter than one full period of the applied voltage")
    a, b = zc_idx[which], zc_idx[which + 1]
    return PeriodWindow(start=a, stop=b, period=float(t[b] - t[a]))


def integer_period_window(t: np.ndarray, v: np.ndarray) -> PeriodWindow:
    """Largest window spanning an integer number of periods (>= 1)."""
    x = v - np.mean(v)
    amp = 0.5 * (np.max(x) - np.min(x))
    h = 0.1 * amp
    zc_idx = []
    armed = False
    for k in range(len(x)):
        if x[k] < -h:
            armed = True
        elif armed and x[k] >= 0:
            zc_idx.append(k)
            armed = False
    if len(zc_idx) < 2:
        raise ValueError("record shorter than one full period of the applied voltage")
    a, b = zc_idx[0], zc_idx[-1]
    n = len(zc_idx) - 1
    return PeriodWindow(start=a, stop=b, period=float(t[b] - t[a]) / n)
