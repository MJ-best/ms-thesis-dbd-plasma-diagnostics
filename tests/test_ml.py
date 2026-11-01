import numpy as np
import pandas as pd

from dbd_diagnostics import ml


def test_metrics_explicit_averaging():
    # thesis Table 6, LR on the train/test split: TN=15 FP=1 FN=2 TP=40
    y = np.r_[np.zeros(16), np.ones(42)].astype(int)
    p = np.r_[np.zeros(15), 1, np.zeros(2), np.ones(40)].astype(int)
    m = ml.evaluate(y, p)
    assert round(100 * m["accuracy"], 2) == 94.83
    assert round(100 * m["precision_macro"], 2) == 92.90  # thesis "PRE"
    assert round(100 * m["recall_macro"], 2) == 94.49  # thesis "REC"
    assert round(m["f1_binary"], 2) == 0.96  # thesis "F1" (binary, not macro)
    assert round(m["f1_macro"], 2) == 0.94


def test_group_holdout_runs():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                       "dev": np.repeat(["d1", "d2"], n // 2)})
    df["label"] = (df["a"] > 0).astype(int)
    models = {k: v for k, v in ml.thesis_models().items() if k in ("LR", "DT")}
    res = ml.results_table(ml.group_holdout(df, ["a", "b"], "label", "dev", models))
    assert set(res["test_group"]) == {"d1", "d2"}
    assert (res["accuracy"] > 0.8).all()


def test_baseline_normalize():
    df = pd.DataFrame({"x": [10.0, 10.0, 12.0, 20.0, 20.0, 24.0], "g": list("aaabbb")})
    ref = pd.Series([True, True, False, True, True, False])
    out = ml.baseline_normalize(df, ["x"], "g", ref)
    assert np.allclose(out["x"], [1, 1, 1.2, 1, 1, 1.2])
