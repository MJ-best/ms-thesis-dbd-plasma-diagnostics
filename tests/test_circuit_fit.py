"""Known-answer tests of the multi-cell equivalent-circuit fit."""
import pytest

from dbd_diagnostics import circuit_fit, simulate
from dbd_diagnostics.io import Waveform


def test_ideal_manley_is_recovered():
    p = simulate.CircuitParams(C_d=34.2e-12, C_g=16.9e-12, U_b=900.0, V_amp=2240.0)
    w = simulate.ideal(p, fs=1e9)
    w = Waveform(t=w.t[::10], v=w.v[::10], i=w.i[::10])  # 10 ns sampling like the scope
    f = circuit_fit.fit(w)
    assert f.C_d == pytest.approx(p.C_d, rel=1e-2)
    assert f.C_g == pytest.approx(p.C_g, rel=2e-2)
    assert f.U_b == pytest.approx(p.U_b, rel=1e-2)
    assert f.spread < 0.02
    assert f.r2 > 0.999


def test_threshold_spread_is_recovered():
    p = simulate.CircuitParams()
    w = simulate.cells(p, spread=0.1, residual=1.0, seed=1)  # clamped cells, scope chain emulated
    f = circuit_fit.fit(w)
    assert f.spread == pytest.approx(0.1, abs=0.02)
    assert f.C_d == pytest.approx(p.C_d, rel=3e-2)
    assert f.U_b == pytest.approx(p.U_b, rel=3e-2)
