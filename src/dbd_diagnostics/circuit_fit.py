"""Inverse fit of a multi-cell equivalent circuit to a measured capture.

The Lissajous method fits straight lines to the Q-V figure and returns the
apparent discharge-phase slope zeta, which mixes the dielectric capacitance
with the fraction of the barrier that is discharging [Peeters & van de Sanden
2015]. Here the measured voltage v(t) drives N parallel cells, each a Manley
cell (C_d/N in series with C_g/N) whose gap voltage is clamped once it reaches
the cell's own threshold U_b (1 + s z_k), with z_k the normal quantiles. The
summed electrode charge is compared with the measured Q(t), and

    theta = (C_d, C_g, U_b, s)

are found by least squares. C_d is then the dielectric capacitance itself,
s the cell-to-cell spread of ignition thresholds (gradual onset), and the
residual tells how well a Manley-type array explains the waveform.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from . import electrical, lissajous, waveform
from .io import Waveform


@dataclass
class CircuitFit:
    C_d: float  # [F] dielectric capacitance of the whole array
    C_g: float  # [F] gas-gap capacitance of the whole array
    U_b: float  # [V] median cell threshold (burning voltage)
    spread: float  # relative SD of the cell thresholds
    r2: float  # coefficient of determination of Q(t)
    rmse: float  # [C] residual charge error
    success: bool

    @property
    def C_cell(self) -> float:
        return self.C_d * self.C_g / (self.C_d + self.C_g)


def simulate_charge(v: np.ndarray, C_d: float, C_g: float, U_b: float, spread: float,
                    n_cells: int = 40) -> np.ndarray:
    """Electrode charge of an array of clamped Manley cells driven by ``v``.

    ``v`` should start one full period before the samples that are compared,
    so that the wall charges reach their periodic state.
    """
    z = norm.ppf((np.arange(n_cells) + 0.5) / n_cells)
    ub = np.maximum(U_b * (1 + spread * z), 1.0)
    cd, cg = C_d / n_cells, C_g / n_cells
    ccell = cd * cg / (cd + cg)
    qc = np.zeros(n_cells)  # charge conducted through each gap (wall charge)
    out = np.empty(len(v))
    for k, vk in enumerate(v):
        q = ccell * (vk + qc / cg)
        vg = vk - q / cd
        hi, lo = vg > ub, vg < -ub
        if hi.any():
            q[hi] = cd * (vk - ub[hi])
            qc[hi] = q[hi] - cg * ub[hi]
        if lo.any():
            q[lo] = cd * (vk + ub[lo])
            qc[lo] = q[lo] + cg * ub[lo]
        out[k] = q.sum()
    return out


def fit(wf: Waveform, n_cells: int = 40, decimate: int = 4, x0: dict | None = None) -> CircuitFit:
    """Fit (C_d, C_g, U_b, spread) to one capture."""
    t, v, i = wf.t, wf.v - np.mean(wf.v), wf.i
    w = waveform.integer_period_window(t, v)  # hysteresis-based, robust to ADC noise
    n_per = int(round(w.period / float(np.median(np.diff(t)))))
    t, v, i = t[w.slice], v[w.slice], electrical.remove_offset(t[w.slice], i[w.slice])
    q = electrical.charge(t, i, center=True)
    if x0 is None:  # start from the Lissajous estimate of the first period
        L = lissajous.analyse(t[:n_per], v[:n_per], q[:n_per])
        cd0 = max(L.C_dielectric, 1.2 * L.C_total)
        cg0 = cd0 * L.C_total / (cd0 - L.C_total)
        x0 = {"C_d": cd0, "C_g": cg0, "U_b": max(abs(L.U_b_onset), 300.0), "spread": 0.2}
    sl = slice(None, None, decimate)
    vd, qd = v[sl], q[sl]
    npd = n_per // decimate
    drive = np.r_[vd[:npd], vd]  # one extra period to settle the wall charges
    scale = np.array([1e-12, 1e-12, 1.0, 1.0])

    def resid(p):
        cd, cg, ub, s = p * scale
        qm = simulate_charge(drive, cd, cg, ub, s, n_cells)[npd:]
        qm -= qm.mean()
        return (qm - qd) * 1e9  # nC

    p0 = np.array([x0["C_d"], x0["C_g"], x0["U_b"], x0["spread"]]) / scale
    lb, ub_ = np.array([5.0, 2.0, 100.0, 0.0]), np.array([300.0, 300.0, 3000.0, 1.5])
    res = least_squares(resid, np.clip(p0, lb + 1e-6, ub_ - 1e-6), bounds=(lb, ub_), x_scale="jac",
                        diff_step=1e-3, max_nfev=200)
    r = res.fun
    r2 = 1 - np.sum(r ** 2) / np.sum((qd * 1e9) ** 2)
    cd, cg, ub, s = res.x * scale
    return CircuitFit(C_d=cd, C_g=cg, U_b=ub, spread=s, r2=float(r2),
                      rmse=float(np.sqrt(np.mean(r ** 2))) * 1e-9, success=bool(res.success))
