"""Synthetic DBD waveforms from the equivalent circuit (for tests and demos).

Two models:

* ``ideal``  – Manley model, gap voltage clamped at +/-U_b (exact parallelogram).
* ``cells``  – N parallel micro-gap cells (the thesis device has 126 cell pairs),
  each with its own breakdown voltage; a cell fires impulsively when its gap
  voltage exceeds its threshold, which produces the train of current spikes seen
  on the oscilloscope. Optional probe bandwidth, sampling and 8-bit ADC
  quantisation emulate the GDS-2102A + P6021A/P6015A measurement chain.

Default parameters reproduce the thesis normal-condition averages
(V_on ~ 0.45-0.5 kV, Q_max ~ 60 nC, E ~ 0.11-0.13 mJ, C_d ~ 34 pF,
C_total ~ 11 pF, f ~ 46.5 kHz) with a burning voltage of ~0.9 kV.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import Waveform


@dataclass
class CircuitParams:
    C_d: float = 34.2e-12  # dielectric capacitance [F]
    C_g: float = 16.9e-12  # gas-gap capacitance [F]
    U_b: float = 900.0  # burning / breakdown voltage of the gap [V]
    V_amp: float = 2240.0  # applied voltage amplitude [V]
    f: float = 46.5e3  # driving frequency [Hz]

    @property
    def C_cell(self) -> float:
        return self.C_d * self.C_g / (self.C_d + self.C_g)


def ideal(params: CircuitParams = CircuitParams(), periods: float = 2.3, fs: float = 1e9) -> Waveform:
    """Exact Manley-model waveform (no noise), steady state after one period."""
    p = params
    t = np.arange(0.0, (periods + 1.6) / p.f, 1.0 / fs)
    v = p.V_amp * np.sin(2 * np.pi * p.f * t)
    qc = 0.0  # charge conducted through the gas (wall charge)
    Q = np.empty_like(t)
    for k, vk in enumerate(v):
        q = p.C_cell * (vk + qc / p.C_g)
        vg = vk - q / p.C_d
        if vg > p.U_b:
            q = p.C_d * (vk - p.U_b)
            qc = q - p.C_g * p.U_b
        elif vg < -p.U_b:
            q = p.C_d * (vk + p.U_b)
            qc = q + p.C_g * p.U_b
        Q[k] = q
    i = np.gradient(Q, t)
    keep = t >= 1.6 / p.f  # drop the start-up transient; start mid-cycle like a scope
    t, v, i = t[keep] - t[keep][0], v[keep], i[keep]
    return Waveform(t=t, v=v, i=i, meta={"model": "ideal", "params": p})


def cells(
    params: CircuitParams = CircuitParams(),
    n_cells: int = 126,
    spread: float = 0.2,
    residual: float = 0.4,
    periods: float = 2.3,
    fs_sim: float = 1e9,
    fs_scope: float = 1e8,
    bandwidth: float | None = 60e6,
    adc_v_lsb: float | None = 40.0,
    adc_i_lsb: float | None = 8e-3,
    noise_i: float = 2e-3,
    seed: int | None = 0,
) -> Waveform:
    """Multi-cell surface-DBD model with impulsive micro-discharges.

    Each cell k has C_d/N, C_g/N and a breakdown threshold drawn from
    U_b * (1 + spread * N(0,1)). When |u_g,k| exceeds it, the cell transfers
    charge so that u_g,k drops to ``residual`` x threshold (same sign), within
    one simulation step (the probe bandwidth then shapes the pulse).
    """
    rng = np.random.default_rng(seed)
    p = params
    N = n_cells
    Cd, Cg = p.C_d / N, p.C_g / N
    Ccell = Cd * Cg / (Cd + Cg)
    Ub = p.U_b * (1 + spread * rng.standard_normal(N))
    t = np.arange(0.0, (periods + 1.6) / p.f, 1.0 / fs_sim)
    v = p.V_amp * np.sin(2 * np.pi * p.f * t)
    qc = np.zeros(N)
    Qtot = np.empty_like(t)
    for k, vk in enumerate(v):
        q = Ccell * (vk + qc / Cg)
        vg = (q - qc) / Cg
        fire = np.abs(vg) > Ub
        if fire.any():
            s = np.sign(vg[fire])
            target = s * residual * Ub[fire]
            # u_g = v - q/Cd and q = Ccell (v + qc/Cg)  ->  solve qc for u_g = target
            # u_g = v (1 - Ccell/Cd) - Ccell qc /(Cd Cg)
            qc[fire] = (vk * (1 - Ccell / Cd) - target) * Cd * Cg / Ccell
            q = Ccell * (vk + qc / Cg)
        Qtot[k] = q.sum()
    i = np.gradient(Qtot, t)
    # measurement chain
    if bandwidth:
        tau = 1.0 / (2 * np.pi * bandwidth)
        a = np.exp(-1.0 / (fs_sim * tau))
        y = np.empty_like(i)
        acc = i[0]
        for k, x in enumerate(i):
            acc = a * acc + (1 - a) * x
            y[k] = acc
        i = y
    step = max(int(round(fs_sim / fs_scope)), 1)
    keep = t >= 1.6 / p.f
    t, v, i = t[keep][::step], v[keep][::step], i[keep][::step]
    t = t - t[0]
    i = i + noise_i * rng.standard_normal(len(i))
    if adc_v_lsb:
        v = np.round(v / adc_v_lsb) * adc_v_lsb
    if adc_i_lsb:
        i = np.round(i / adc_i_lsb) * adc_i_lsb
    meta = {"model": "cells", "params": p, "n_cells": N, "spread": spread, "residual": residual}
    return Waveform(t=t, v=v, i=i, meta=meta)
