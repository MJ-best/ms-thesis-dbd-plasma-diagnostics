import numpy as np
import pytest

from dbd_diagnostics import features, io, plasma, simulate


def test_paschen_100um_air():
    vb = plasma.paschen_voltage(760.0, 0.01, gamma=0.01)
    assert 800 < vb < 900


def test_number_density():
    assert plasma.number_density(101325, 273.15) == pytest.approx(2.687e19, rel=1e-3)


def test_mobility_and_thesis_reduced_field():
    assert plasma.electron_mobility() == pytest.approx(592.1, rel=1e-3)
    # thesis: E/N = 81.38e-16 V cm^2 needs U = 2.28 kV with N = 2.8e19 (not 2.1 kV)
    assert plasma.reduced_field(2100, 0.01, 2.8e19) == pytest.approx(75.0e-16, rel=1e-3)


def test_te_inside_and_outside_table():
    assert np.isnan(plasma.electron_temperature_from_field(2280, 0.01, 2.8e19))
    te_ex = plasma.electron_temperature_from_field(2280, 0.01, 2.8e19, extrapolate=True)
    assert 11 < te_ex < 14  # thesis: 13.2 eV from the extrapolated curve
    te = plasma.electron_temperature_from_field(900, 0.01, plasma.number_density())
    assert 5 < te < 8


def test_electron_density_formula():
    ne = plasma.electron_density(0.0143, 0.01, 4.5e-4, 2100, 592.12)
    assert ne == pytest.approx(1.6e12, rel=1e-2)  # thesis mean value


def test_extract_and_csv_roundtrip(tmp_path):
    wf = simulate.cells(seed=3, periods=3.3)
    path = tmp_path / "cap.csv"
    io.save_csv(wf, path)
    r = features.extract(io.load_csv(path))
    assert r["n_periods"] >= 2
    assert 45e3 < r["frequency"] < 48e3
    assert r["energy"] > 0 and r["power"] == pytest.approx(r["energy"] * r["frequency"], rel=1e-3)
    assert r["u_b"] < r["v_max"] < r["v_breakdown"] * 1.2


def test_extract_is_the_mean_of_the_periods():
    wf = simulate.cells(seed=3, periods=3.3)
    rows, r = features.extract_periods(wf), features.extract(wf)
    assert len(rows) == r["n_periods"]
    for k in ("v_rms", "energy", "c_dielectric", "u_b"):
        assert r[k] == pytest.approx(np.mean([row[k] for row in rows]))


def test_gwinstek_parser(tmp_path):
    header = ["Format,2.0A,", "Memory Length,4,Memory Length,4,", "Source,CH1,Source,CH2,",
              "Probe Ratio,1.000E+03,Probe Ratio,2.000E+00,", "Vertical Units,V,Vertical Units,A,",
              "Sampling Period,1.000E-08,Sampling Period,1.000E-08,", "Waveform Data,,Waveform Data,,"]
    data = [f"{k*1e-8:e},{100*k:f},{k*1e-8:e},{0.001*k:f}," for k in range(4)]
    p = tmp_path / "ALL0001.CSV"
    p.write_text("\n".join(header + data) + "\n")
    wf = io.load_csv(p)
    assert np.allclose(wf.v, [0, 100, 200, 300]) and np.allclose(wf.i, [0, 0.001, 0.002, 0.003])
    assert wf.meta["Vertical Units"] == ("V", "A")
