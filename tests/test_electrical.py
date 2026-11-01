import numpy as np
import pytest

from dbd_diagnostics import electrical, waveform


def _sine(f=46.5e3, amp=2000.0, periods=3, fs=1e8, phase=0.3):
    t = np.arange(0, periods / f, 1 / fs)
    return t, amp * np.sin(2 * np.pi * f * t + phase)


def test_rms_of_sine():
    t, v = _sine()
    w = waveform.integer_period_window(t, v)
    assert electrical.rms(t[w.slice], v[w.slice]) == pytest.approx(2000 / np.sqrt(2), rel=2e-3)


def test_frequency():
    t, v = _sine(f=46.5e3)
    assert waveform.estimate_frequency(t, v) == pytest.approx(46.5e3, rel=1e-3)


def test_energy_of_resistor_and_capacitor():
    f, R, C = 46.5e3, 1e5, 30e-12
    t = np.arange(0, 1 / f, 1e-9)
    v = 2000 * np.sin(2 * np.pi * f * t)
    i_r = v / R
    i_c = C * np.gradient(v, t)
    # resistor: E = Vrms^2/R * T ; capacitor: no net energy
    assert electrical.energy_per_cycle(t, v, i_r) == pytest.approx((2000**2 / 2) / R / f, rel=1e-3)
    assert abs(electrical.energy_per_cycle(t, v, i_c)) < 1e-3 * (2000**2 / 2) / R / f
    assert electrical.mean_power(t, v, i_r) == pytest.approx((2000**2 / 2) / R, rel=1e-3)


def test_offset_removal_prevents_charge_drift():
    f = 46.5e3
    t = np.arange(0, 1 / f, 1e-8)
    i = 0.01 * np.sin(2 * np.pi * f * t) + 0.008  # one 8 mA LSB offset
    q_raw = electrical.charge(t, i, center=False)
    q = electrical.charge(t, electrical.remove_offset(t, i), center=False)
    assert abs(q_raw[-1]) > 1e-7  # ~170 nC drift per period without correction
    assert abs(q[-1]) < 1e-12
