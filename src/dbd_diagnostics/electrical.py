"""Cycle-averaged electrical quantities (thesis Eqs. 9-12, 23).

Notation follows the thesis: v(t) applied voltage, i(t) total measured current
(displacement + discharge), T = 1/f one period.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid


def rms(t: np.ndarray, x: np.ndarray) -> float:
    """RMS value  sqrt( (1/T) * integral_0^T x(t)^2 dt )  (thesis Eqs. 9, 10).

    ``t, x`` must span an integer number of periods.
    """
    T = t[-1] - t[0]
    return float(np.sqrt(trapezoid(x * x, t) / T))


def remove_offset(t: np.ndarray, i: np.ndarray) -> np.ndarray:
    """Subtract the time-average of the current over the (integer-period) window.

    In periodic steady state the net charge per cycle is zero, so any non-zero
    mean is probe/ADC offset.
    """
    T = t[-1] - t[0]
    return i - trapezoid(i, t) / T


def energy_per_cycle(t: np.ndarray, v: np.ndarray, i: np.ndarray, period: float | None = None) -> float:
    """Electrical energy dissipated per period  E = integral_0^T v(t) i(t) dt  [J].

    If the window spans several periods, pass ``period`` and the result is
    scaled to one period. Note: the thesis' Eq. (11) carries a 1/T factor
    (which makes it the mean *power* in W) but the reported value 0.13 mJ
    (Fig. 25a, unit J) is the energy *per cycle*; see docs/thesis-review.md.
    """
    e = float(trapezoid(v * i, t))
    if period is not None:
        e *= period / (t[-1] - t[0])
    return e


def mean_power(t: np.ndarray, v: np.ndarray, i: np.ndarray) -> float:
    """Mean electrical power  P = (1/T) integral v i dt  [W] over the given window."""
    T = t[-1] - t[0]
    return float(trapezoid(v * i, t) / T)


def charge(t: np.ndarray, i: np.ndarray, center: bool = True) -> np.ndarray:
    """Charge on the electrodes  Q(t) = integral_0^t i(t') dt' (+ const)  [C].

    The thesis writes Eq. (23) as Q = integral_0^T i dt; the Lissajous figure
    needs the running integral Q(t). The integration constant is unknown; with
    ``center=True`` it is chosen so that max(Q) = -min(Q) (symmetric discharge),
    which is required for burning-voltage estimates that use absolute Q.
    The current should be offset-free (see :func:`remove_offset`).
    """
    q = cumulative_trapezoid(i, t, initial=0.0)
    if center:
        q = q - 0.5 * (q.max() + q.min())
    return q
