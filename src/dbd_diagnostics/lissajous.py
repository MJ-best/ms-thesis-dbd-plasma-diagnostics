r"""Charge-voltage (Lissajous) analysis based on the Manley equivalent circuit.

Equivalent circuit (thesis Fig. 17/18): dielectric capacitance ``C_d`` in series
with the gas-gap capacitance ``C_g``; while the gap is conducting, the gap
voltage is clamped at the *burning voltage* ``U_b``.

With ``Q`` the charge on the electrodes (= charge on ``C_d``)::

    u_g(t) = v(t) - Q(t) / C_d                         (gap voltage)
    no discharge:  dQ/dV = C_cell = C_d C_g / (C_d + C_g)   ("C_total", Eq. 24)
    discharge:     Q = C_d (V - U_b)  ->  dQ/dV = C_d  ("C_dielectric")

Consequences used below (all exact in the ideal model, see tests/):

* transferred charge per half cycle   dQ = C_d (V_max - V_on)
* burning voltage                     U_b = V_on - Q_on / C_d
                                          = (1 - C_cell/C_d) * (V_max + V_on) / 2
                                          = V-axis intercept of the discharge line
                                          = E_cycle / (2 dQ)
* Manley:  E_cycle = 4 C_d U_b (V_max - U_min),  U_min = U_b (C_d + C_g) / C_d

**Thesis Eq. (15)**  V_breakdown = V_min + Q_max / C_dielectric  uses
``Q_max = dQ``; substituting ``dQ = C_d (V_max - V_on)`` gives
``V_breakdown = V_max`` identically, i.e. the thesis quantity is the peak
applied voltage, not the gas breakdown voltage. It is kept here only as
:func:`thesis_breakdown_voltage` for reproducibility.

If the discharge does not cover the whole dielectric (partial surface
discharging, Peeters & van de Sanden 2015, thesis ref. [50]) the discharge-phase
slope is an *effective* capacitance ``zeta_diel <= C_d``; the formulas above then
hold with ``C_d -> zeta_diel`` (Pipa & Brandenburg 2019, ref. [55]).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.integrate import trapezoid


@dataclass
class LineFit:
    slope: float
    intercept: float
    slope_stderr: float
    r2: float
    n: int


@dataclass
class LissajousResult:
    C_total: float  # [F] slope of the non-discharge phase (C_cell)
    C_dielectric: float  # [F] slope of the discharge phase (C_d or zeta_diel)
    C_gas: float  # [F] from 1/C_total = 1/C_d + 1/C_g  (thesis Eq. 24)
    V_max: float  # [V] mean |peak applied voltage| of both half cycles
    V_on: float  # [V] mean |applied voltage at discharge onset| (= thesis V_min)
    Q_on: float  # [C] mean |electrode charge at onset| (centered Q)
    dQ: float  # [C] mean charge transferred per half cycle (= thesis Q_max)
    E_cycle: float  # [J] Lissajous area = energy per cycle
    U_b_onset: float  # [V] V_on - Q_on/C_d  (sign-consistent)
    U_b_intercept: float  # [V] V-axis intercept of the discharge-phase line (centered Q)
    U_b_geometry: float  # [V] (1 - C_total/C_d) (V_max + V_on)/2
    U_b_energy: float  # [V] E_cycle / (2 dQ)
    U_min: float  # [V] minimum amplitude for ignition = U_b (C_d+C_g)/C_d
    V_breakdown_thesis: float  # [V] thesis Eq. 15: V_on + dQ/C_d (== V_max)
    fit_total: tuple  # (rising, falling) LineFit of the non-discharge phase
    fit_dielectric: tuple  # (rising, falling) LineFit of the discharge phase

    @property
    def U_b(self) -> float:
        """Recommended burning-voltage estimate (onset/intercept method)."""
        return self.U_b_onset

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("fit_total")
        d.pop("fit_dielectric")
        d["C_total_fit_stderr"] = float(np.mean([f.slope_stderr for f in self.fit_total]))
        d["C_dielectric_fit_stderr"] = float(np.mean([f.slope_stderr for f in self.fit_dielectric]))
        d["C_total_fit_r2"] = float(np.mean([f.r2 for f in self.fit_total]))
        d["C_dielectric_fit_r2"] = float(np.mean([f.r2 for f in self.fit_dielectric]))
        return d


def linear_fit(x: np.ndarray, y: np.ndarray) -> LineFit:
    """Ordinary least squares y = a x + b with slope standard error and R^2.

    (Equivalent to scipy.optimize.curve_fit with a linear model, as used in the
    thesis; the ANOVA F statistic of a simple regression equals
    (slope/stderr)^2.)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    a = np.sum((x - xm) * (y - ym)) / sxx
    b = ym - a * xm
    res = y - (a * x + b)
    sse = float(np.sum(res**2))
    sst = float(np.sum((y - ym) ** 2))
    se = np.sqrt(sse / max(n - 2, 1) / sxx)
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return LineFit(float(a), float(b), float(se), float(r2), n)


def _two_segment(x: np.ndarray, y: np.ndarray, min_frac: float = 0.08):
    """Best breakpoint for a two-segment linear fit (non-discharge -> discharge)."""
    n = len(x)
    lo = max(int(min_frac * n), 3)
    hi = min(int((1 - min_frac) * n), n - 3)
    if hi <= lo:
        raise ValueError("branch too short for a two-segment fit")
    # cumulative sums for O(n) SSE evaluation of every split
    def cums(xx, yy):
        return (np.cumsum(np.ones_like(xx)), np.cumsum(xx), np.cumsum(yy),
                np.cumsum(xx * xx), np.cumsum(xx * yy), np.cumsum(yy * yy))

    def sse_from(c, k):
        m, sx, sy, sxx, sxy, syy = (v[k] for v in c)
        vx = sxx - sx * sx / m
        if vx <= 0:
            return np.inf
        cxy = sxy - sx * sy / m
        vy = syy - sy * sy / m
        return vy - cxy * cxy / vx

    cf = cums(x, y)
    cb = cums(x[::-1], y[::-1])
    best, best_k = np.inf, lo
    for k in range(lo, hi):
        s = sse_from(cf, k - 1) + sse_from(cb, n - k - 1)
        if s < best:
            best, best_k = s, k
    return best_k


@dataclass
class _Branch:
    fit1: LineFit
    fit2: LineFit
    V_on: float
    Q_on: float
    V_peak: float
    Q_peak: float


def _analyse_branch(V: np.ndarray, Q: np.ndarray) -> _Branch:
    k = _two_segment(V, Q)
    f1 = linear_fit(V[:k], Q[:k])
    f2 = linear_fit(V[k:], Q[k:])
    # onset = breakpoint of the two-segment fit (first sample of the discharge
    # phase). For an ideal parallelogram this equals the intersection of the
    # two lines; with a gradual onset (spread of cell thresholds) the data
    # point is more robust than the line intersection.
    V_on, Q_on = V[k], Q[k]
    kp = int(np.argmax(V))  # branch runs from -V_peak to +V_peak
    return _Branch(f1, f2, float(V_on), float(Q_on), float(V[kp]), float(Q[kp]))


def analyse(t: np.ndarray, v: np.ndarray, q: np.ndarray) -> LissajousResult:
    """Analyse one period of ``v(t)`` and centered charge ``q(t)``.

    The period is split at the voltage extrema into a rising branch
    (-V_max -> +V_max) and a falling branch (+V_max -> -V_max); each branch is
    fitted with two straight lines (slope C_total, then slope C_dielectric).
    """
    k_max = int(np.argmax(v))
    k_min = int(np.argmin(v))
    # cyclic ordering: rising = k_min -> k_max, falling = k_max -> k_min
    n = len(v)
    idx = np.arange(n)
    if k_min < k_max:
        rising = idx[k_min:k_max + 1]
        falling = np.r_[idx[k_max:], idx[:k_min + 1]]
    else:
        rising = np.r_[idx[k_min:], idx[:k_max + 1]]
        falling = idx[k_max:k_min + 1]
    br = _analyse_branch(v[rising], q[rising])
    bf = _analyse_branch(-v[falling], -q[falling])  # mirror the falling branch

    C_t = 0.5 * (br.fit1.slope + bf.fit1.slope)
    C_d = 0.5 * (br.fit2.slope + bf.fit2.slope)
    C_g = C_d * C_t / (C_d - C_t) if C_d > C_t else float("nan")
    V_max = 0.5 * (br.V_peak + bf.V_peak)
    V_on = 0.5 * (br.V_on + bf.V_on)
    Q_on = 0.5 * (br.Q_on + bf.Q_on)
    dQ = 0.5 * ((br.Q_peak - br.Q_on) + (bf.Q_peak - bf.Q_on))
    # energy per cycle = closed-loop integral  E = \oint V dQ
    E = float(trapezoid(v, q))
    # close the loop explicitly
    E += 0.5 * (v[-1] + v[0]) * (q[0] - q[-1])
    E = abs(E)
    U_b_onset = 0.5 * ((br.V_on - br.Q_on / br.fit2.slope) + (bf.V_on - bf.Q_on / bf.fit2.slope))
    U_b_icpt = 0.5 * (-br.fit2.intercept / br.fit2.slope - bf.fit2.intercept / bf.fit2.slope)
    U_b_geom = (1 - C_t / C_d) * 0.5 * (V_max + V_on)
    U_b_energy = E / (2 * dQ) if dQ > 0 else float("nan")
    U_min = U_b_onset * (C_d + C_g) / C_d if np.isfinite(C_g) else float("nan")
    return LissajousResult(
        C_total=C_t, C_dielectric=C_d, C_gas=C_g, V_max=V_max, V_on=V_on, Q_on=Q_on,
        dQ=dQ, E_cycle=E, U_b_onset=U_b_onset, U_b_intercept=U_b_icpt, U_b_geometry=U_b_geom,
        U_b_energy=U_b_energy, U_min=U_min,
        V_breakdown_thesis=thesis_breakdown_voltage(V_on, dQ, C_d),
        fit_total=(br.fit1, bf.fit1), fit_dielectric=(br.fit2, bf.fit2),
    )


def thesis_breakdown_voltage(V_min: float, Q_max: float, C_dielectric: float) -> float:
    """Thesis Eq. (13)-(15): V_breakdown = V_min + Q_wall / C_dielectric.

    .. warning:: With Q_wall = Q_max = C_dielectric (V_max - V_min) this is
       identically V_max (peak applied voltage). Use
       :attr:`LissajousResult.U_b` for the gas (burning) voltage.
    """
    return V_min + Q_max / C_dielectric


def gap_voltage(v: np.ndarray, q: np.ndarray, C_d: float) -> np.ndarray:
    """Time-resolved gas-gap voltage  u_g = v - Q/C_d  (Pipa & Brandenburg 2019)."""
    return v - q / C_d


def discharge_current(t: np.ndarray, v: np.ndarray, i: np.ndarray, C_d: float, C_cell: float) -> np.ndarray:
    """Discharge (conduction) current through the gas, displacement removed.

    i_dis = (1 - C_cell/C_d)^-1 * (i - C_cell dv/dt)   (Pipa & Brandenburg 2019)
    """
    dvdt = np.gradient(v, t)
    return (i - C_cell * dvdt) / (1.0 - C_cell / C_d)


def manley_energy(C_d: float, C_g: float, U_b: float, V_max: float) -> float:
    """Manley (1943) energy per cycle  4 C_d U_b (V_max - U_min)."""
    U_min = U_b * (C_d + C_g) / C_d
    return 4.0 * C_d * U_b * (V_max - U_min)
