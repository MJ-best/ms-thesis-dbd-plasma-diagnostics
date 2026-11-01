"""Surface-damage classification (thesis Chapter 5) with leakage-safe evaluation.

Differences from the original notebooks (see docs/thesis-review.md, Part C):

* scaling lives *inside* the sklearn Pipeline, so it is fitted on training data only;
* evaluation is done across devices (group hold-out), which is the setting that
  matters in practice; a random split of captures from the same device is
  optimistic because neighbouring captures are strongly correlated;
* metrics are reported with an explicit averaging convention (``*_binary`` for
  the positive class "degraded" = 1, ``*_macro`` for the class average) –
  the thesis table mixes macro precision/recall with binary F1;
* optional per-device baseline normalisation (divide each feature by its median
  over that device's first, known-healthy captures) removes device-to-device
  offsets, which is what made LR/SVM/kNN fail on the validation device.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#: The 14 input features of thesis Table 4 (column names used by features.py)
THESIS_FEATURES = [
    "frequency", "energy", "v_rms", "i_rms", "c_dielectric", "c_gas", "c_total",
    "v_min", "v_breakdown", "q_max", "n_discharges", "ne_mean", "ne_std", "te",
]


def thesis_models(random_state: int = 42) -> dict:
    """Classifiers with the hyper-parameters stated in thesis Sec. 5.4."""
    def pipe(est):
        return Pipeline([("scale", StandardScaler()), ("clf", est)])

    return {
        "LR": pipe(LogisticRegression(max_iter=5000)),
        "DT": pipe(DecisionTreeClassifier(max_depth=3, min_samples_split=4, random_state=random_state)),
        "RF": pipe(RandomForestClassifier(n_estimators=1000, random_state=random_state, n_jobs=-1)),
        "GB": pipe(GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000, random_state=random_state)),
        "SVM": pipe(SVC(kernel="linear", probability=True, random_state=random_state)),
        "kNN": pipe(KNeighborsClassifier(n_neighbors=3)),
    }


def evaluate(y_true, y_pred, y_score=None) -> dict:
    """All metrics with explicit averaging; positive class = 1 (degraded)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y_true)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_binary": precision_score(y_true, y_pred, zero_division=0),
        "recall_binary": recall_score(y_true, y_pred, zero_division=0),
        "f1_binary": f1_score(y_true, y_pred, zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["auc"] = roc_auc_score(y_true, y_score)
    return out


def baseline_normalize(df: pd.DataFrame, features: list[str], group: str, reference: pd.Series) -> pd.DataFrame:
    """Divide each feature by its per-group median over ``reference`` rows.

    ``reference`` must select captures known to be healthy *at deployment
    time* (e.g. the first hours of operation of each device), never rows chosen
    with knowledge of the label.
    """
    out = df.copy()
    for g, idx in df.groupby(group).groups.items():
        ref = df.loc[idx][reference.loc[idx]]
        if ref.empty:
            raise ValueError(f"group {g!r} has no reference rows")
        med = ref[features].median()
        out.loc[idx, features] = df.loc[idx, features] / med.replace(0, np.nan)
    return out


@dataclass
class GroupResult:
    model: str
    train_groups: tuple
    test_groups: tuple
    metrics: dict


def group_holdout(df: pd.DataFrame, features: list[str], label: str, group: str,
                  models: dict | None = None) -> list[GroupResult]:
    """Leave-one-group-out evaluation (train on all devices but one, test on it)."""
    models = models or thesis_models()
    results = []
    groups = sorted(df[group].unique())
    for g in groups:
        tr, te = df[df[group] != g], df[df[group] == g]
        for name, m in models.items():
            est = clone(m).fit(tr[features], tr[label])
            pred = est.predict(te[features])
            score = est.predict_proba(te[features])[:, 1] if hasattr(est, "predict_proba") else None
            results.append(GroupResult(name, tuple(x for x in groups if x != g), (g,),
                                       evaluate(te[label].to_numpy(), pred, score)))
    return results


def results_table(results: list[GroupResult]) -> pd.DataFrame:
    rows = [{"model": r.model, "test_group": ",".join(map(str, r.test_groups)), **r.metrics} for r in results]
    return pd.DataFrame(rows)
