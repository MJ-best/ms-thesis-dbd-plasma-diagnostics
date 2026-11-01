import numpy as np
import pandas as pd
import pytest

from dbd_diagnostics import cli, features, io, ml, simulate


@pytest.fixture(scope="module")
def feature_table():
    rng = np.random.default_rng(0)
    rows = []
    for dev, dv in (("A", 0.0), ("B", 40.0)):
        for label in (0, 1):
            for _ in range(6):
                sag = 0.94 if label else 1.0
                p = simulate.CircuitParams(V_amp=(2240 + dv) * sag * (1 + 0.003 * rng.standard_normal()), U_b=900 * sag)
                r = features.extract(simulate.ideal(p))
                r.update(device=dev, label=label)
                rows.append(r)
    return pd.DataFrame(rows)


def test_train_and_predict_roundtrip(tmp_path, feature_table):
    feats = ["v_rms", "i_rms", "energy", "c_dielectric", "c_total", "q_max", "u_b"]
    csv = tmp_path / "features.csv"
    feature_table.to_csv(csv, index=False)
    model = tmp_path / "model.joblib"
    assert cli.main(["train", str(csv), "-o", str(model), "--model", "LR", "--columns", ",".join(feats)]) == 0
    bundle = ml.load_bundle(model)
    assert bundle["features"] == feats and bundle["n_train"] == len(feature_table)
    pred = ml.predict_bundle(bundle, feature_table)
    assert (pred["predicted_label"] == feature_table["label"]).mean() > 0.9
    cap = tmp_path / "cap.csv"
    io.save_csv(simulate.ideal(), cap)
    assert cli.main(["predict", str(model), str(cap), "-o", str(tmp_path / "pred.csv")]) == 0
    assert len(pd.read_csv(tmp_path / "pred.csv")) == 1


def test_plots():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from dbd_diagnostics import plots
    wf = simulate.cells(seed=2)
    assert plots.waveform_figure(wf) is not None
    assert plots.lissajous_figure(wf) is not None


def test_streamlit_app_runs():
    pytest.importorskip("streamlit")
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    app = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=180).run()
    assert not at.exception
    assert any(m.label == "Burning voltage U_b" for m in at.metric)
    at.number_input[0].set_value(5000.0).run()  # baseline far above -> alarm
    assert at.error and not at.exception
