"""End-to-end feature extraction for one oscilloscope capture.

For every full period in the record the electrical and Lissajous quantities are
computed and then averaged (the original notebooks used only the last period).

Two families of features are returned:

* ``thesis`` – the 14 inputs of thesis Table 4, computed with the thesis
  definitions (incl. V_breakdown = V_min + Q_max/C_d, which equals V_max, and
  the E/N -> D/mu -> T_e chain with N = 2.8e19 cm^-3 and extrapolation);
* ``corrected`` – physically consistent counterparts: mean power, burning
  voltage U_b from the Lissajous figure, E/N in Td, T_e within the D/mu table
  (<= 390 Td; above 175 Td the table is the thesis' extrapolation), n_e from the displacement-free discharge current and U_b.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from . import discharge, electrical, lissajous, plasma
from .io import Waveform


@dataclass
class DeviceConfig:
    """Geometry/gas of the thesis device (micro-gap surface DBD in ambient air)."""
    gap_cm: float = 100e-4  # electrode gap 100 um
    cell_area_cm2: float = 150e-4 * 300e-4  # 150 um x 300 um discharge cell (thesis Eq. 17)
    pressure_torr: float = 760.0
    gas_temperature_K: float = 293.15
    mobility_cm2_Vs: float = plasma.MU_P_AIR / 760.0  # 592.1 cm^2/(V s)
    N_thesis_cm3: float = 2.8e19  # value used in the thesis
    peak_method: str = "robust"  # 'robust' or 'thesis'
    peak_k: float = 5.0  # robust threshold in MAD sigmas
    peak_w: float = 1.5  # thesis prominence multiplier


def _periods(t: np.ndarray, v: np.ndarray) -> list[slice]:
    x = v - np.mean(v)
    amp = 0.5 * (np.max(x) - np.min(x))
    h = 0.1 * amp
    zc, armed = [], False
    for k in range(len(x)):
        if x[k] < -h:
            armed = True
        elif armed and x[k] >= 0:
            zc.append(k)
            armed = False
    return [slice(a, b) for a, b in zip(zc[:-1], zc[1:])]


def extract(wf: Waveform, cfg: DeviceConfig = DeviceConfig()) -> dict:
    """Compute all features of one capture (averaged over its full periods)."""
    rows = extract_periods(wf, cfg)
    with warnings.catch_warnings():  # all-NaN columns (e.g. no pulses in an ideal waveform) stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {k: float(np.nanmean([row[k] for row in rows])) for k in rows[0]}
    out["n_periods"] = len(rows)
    for key in ("Time", "Vertical Scale", "Sampling Period"):
        if key in wf.meta:
            val = wf.meta[key]
            out[f"meta_{key.lower().replace(' ', '_')}"] = val[1] if isinstance(val, tuple) else val
    out["source_file"] = wf.meta.get("source_file", "")
    return out


def extract_periods(wf: Waveform, cfg: DeviceConfig = DeviceConfig()) -> list[dict]:
    """Compute all features of every full period of one capture (one dict per period)."""
    periods = _periods(wf.t, wf.v)
    if not periods:
        raise ValueError("no full period in record")
    rows = []
    for sl in periods:
        t, v, i = wf.t[sl], wf.v[sl], wf.i[sl]
        v = v - np.mean(v)  # remove probe DC offset (as in the MATLAB code)
        i = electrical.remove_offset(t, i)
        q = electrical.charge(t, i, center=True)
        T = t[-1] - t[0]
        r = {}
        r["frequency"] = 1.0 / T
        r["v_rms"] = electrical.rms(t, v)
        r["i_rms"] = electrical.rms(t, i)
        r["energy"] = electrical.energy_per_cycle(t, v, i)
        r["power"] = r["energy"] / T
        L = lissajous.analyse(t, v, q)
        r["c_total"], r["c_dielectric"], r["c_gas"] = L.C_total, L.C_dielectric, L.C_gas
        r["q_max"], r["v_max"] = L.dQ, L.V_max
        r["v_on_lissajous"] = L.V_on
        r["u_b"] = L.U_b_onset
        r["u_b_energy"] = L.U_b_energy
        r["u_b_geometry"] = L.U_b_geometry
        r["c_total_r2"] = float(np.mean([f.r2 for f in L.fit_total]))
        r["c_dielectric_r2"] = float(np.mean([f.r2 for f in L.fit_dielectric]))
        # pulses
        i_dis = lissajous.discharge_current(t, v, i, L.C_dielectric, L.C_total)
        if cfg.peak_method == "thesis":
            ps = discharge.pulse_statistics(v, i, method="thesis", w=cfg.peak_w)
        else:
            ps = discharge.pulse_statistics(v, i_dis, method="robust", k=cfg.peak_k)
        r["n_discharges"] = ps.n_discharges
        r["v_min"] = ps.v_onset
        # thesis Q_max (Eq. 25): integral of i from the first pulse (V_min) to V_max
        dq = [abs(q[e] - q[o]) for o, e in zip(ps.onset_indices, ps.extremum_indices) if o is not None]
        r["q_max_thesis"] = float(np.mean(dq)) if dq else float("nan")
        # thesis V_breakdown (Eq. 15) -- equals ~V_max, kept for reproducibility
        r["v_breakdown"] = lissajous.thesis_breakdown_voltage(ps.v_onset, r["q_max_thesis"], L.C_dielectric)
        # thesis chain: n_e with V_breakdown and total current peaks; T_e at E/N(N=2.8e19), extrapolated
        ip = ps.peak_currents
        if len(ip):
            ne_th = plasma.electron_density(ip, cfg.gap_cm, cfg.cell_area_cm2, r["v_breakdown"], cfg.mobility_cm2_Vs)
        else:
            ne_th = np.array([np.nan])
        r["ne_mean"], r["ne_std"] = float(np.mean(ne_th)), float(np.std(ne_th))
        r["en_thesis_1e16"] = plasma.reduced_field(r["v_breakdown"], cfg.gap_cm, cfg.N_thesis_cm3) / 1e-16
        r["te"] = plasma.electron_temperature_from_field(
            r["v_breakdown"], cfg.gap_cm, cfg.N_thesis_cm3, extrapolate=True)
        # corrected chain: U_b, real gas density, no extrapolation, discharge current
        N = plasma.number_density(cfg.pressure_torr * 133.322368, cfg.gas_temperature_K)
        r["en_td"] = plasma.reduced_field(L.U_b_onset, cfg.gap_cm, N) / plasma.TOWNSEND
        r["te_corrected"] = plasma.electron_temperature_from_field(L.U_b_onset, cfg.gap_cm, N, extrapolate=False)
        idp = np.abs(i_dis[ps.peak_indices]) if len(ps.peak_indices) else np.array([np.nan])
        ne_c = plasma.electron_density(idp, cfg.gap_cm, cfg.cell_area_cm2, L.U_b_onset, cfg.mobility_cm2_Vs)
        r["ne_corrected_mean"] = float(np.mean(ne_c))
        rows.append(r)
    return rows
