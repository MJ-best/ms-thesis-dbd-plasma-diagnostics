"""The ideal Manley model has a known burning voltage; check all estimators."""
import numpy as np
import pytest

from dbd_diagnostics import electrical, lissajous, simulate, waveform


@pytest.fixture(scope="module")
def ideal_result():
    p = simulate.CircuitParams(C_d=34.2e-12, C_g=16.9e-12, U_b=900.0, V_amp=2240.0)
    wf = simulate.ideal(p)
    w = waveform.one_period_window(wf.t, wf.v)
    t, v, i = wf.t[w.slice], wf.v[w.slice], wf.i[w.slice]
    q = electrical.charge(t, electrical.remove_offset(t, i))
    return p, lissajous.analyse(t, v, q)


def test_capacitances(ideal_result):
    p, r = ideal_result
    assert r.C_dielectric == pytest.approx(p.C_d, rel=5e-3)
    assert r.C_total == pytest.approx(p.C_cell, rel=5e-3)
    assert r.C_gas == pytest.approx(p.C_g, rel=2e-2)


def test_burning_voltage_estimators(ideal_result):
    p, r = ideal_result
    for est in (r.U_b_onset, r.U_b_intercept, r.U_b_geometry, r.U_b_energy):
        assert est == pytest.approx(p.U_b, rel=1e-2)


def test_thesis_breakdown_formula_equals_peak_voltage(ideal_result):
    """Thesis Eq. (15) V_min + Q_max/C_dielectric reduces to V_max."""
    p, r = ideal_result
    assert r.V_breakdown_thesis == pytest.approx(p.V_amp, rel=1e-3)
    assert abs(r.V_breakdown_thesis - p.U_b) > 1000.0


def test_manley_energy(ideal_result):
    p, r = ideal_result
    assert r.E_cycle == pytest.approx(lissajous.manley_energy(p.C_d, p.C_g, p.U_b, p.V_amp), rel=1e-2)


def test_transferred_charge(ideal_result):
    p, r = ideal_result
    assert r.dQ == pytest.approx(p.C_d * (p.V_amp - r.V_on), rel=1e-2)


def test_gap_voltage_clamped(ideal_result):
    p, _ = ideal_result
    wf = simulate.ideal(p)
    w = waveform.one_period_window(wf.t, wf.v)
    t, v, i = wf.t[w.slice], wf.v[w.slice], wf.i[w.slice]
    q = electrical.charge(t, electrical.remove_offset(t, i))
    ug = lissajous.gap_voltage(v, q, p.C_d)
    assert np.max(np.abs(ug)) == pytest.approx(p.U_b, rel=1e-2)


def test_noisy_multicell_model_is_consistent():
    """With 126 stochastic cells, 8-bit ADC and 100 MS/s the method stays robust."""
    wf = simulate.cells(seed=1)
    w = waveform.one_period_window(wf.t, wf.v)
    t, v, i = wf.t[w.slice], wf.v[w.slice], wf.i[w.slice]
    q = electrical.charge(t, electrical.remove_offset(t, i))
    r = lissajous.analyse(t, v, q)
    assert 0.8 < r.C_dielectric / 34.2e-12 < 1.1
    assert r.V_breakdown_thesis == pytest.approx(r.V_max, rel=0.1)
    assert 400 < r.U_b < 950  # effective burning voltage of cells firing between 0.4 and 1.0 U_b
