"""Re-analysis of the original thesis captures (data not distributed here).

    python examples/reanalyse_thesis_data.py <m1_data dir> <m2_data dir>

Folder-to-label mapping used in the thesis (surface photographs):
  device m2 (training device):   5 h, 10 h -> 0 (normal);  15-40 h -> 1 (degraded)
  device m1 (validation device): data05, data10 -> 0;      data15-25 -> 1
Prints group statistics and the cross-device ML evaluation reported in
docs/thesis-review.md (Part D).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from dbd_diagnostics import features, io, ml

CORRECTED = ["frequency", "power", "v_rms", "i_rms", "c_dielectric", "c_total", "q_max", "u_b", "en_td",
             "te_corrected", "n_discharges", "ne_corrected_mean"]


def label_for(device, folder):
    """0 = normal (<= 10 h of operation), 1 = degraded (>= 15 h), as labelled in the thesis."""
    m = re.search(r"(\d+)\s*시간", folder) if device == "m2" else re.search(r"data(\d+)", folder)
    return 0 if int(m.group(1)) <= 10 else 1


def extract_device(root, device):
    rows = []
    for f in sorted(Path(root).rglob("ALL*.CSV")):
        wf = io.load_csv(f)
        if wf.dt > 5e-8:  # overview captures (0.2-0.4 ms/sample)
            continue
        r = features.extract(wf)
        r.update(device=device, folder=f.parent.name, label=label_for(device, f.parent.name))
        rows.append(r)
    return pd.DataFrame(rows)


def main(m1_dir, m2_dir):
    df = pd.concat([extract_device(m1_dir, "m1"), extract_device(m2_dir, "m2")], ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=ml.THESIS_FEATURES + CORRECTED)
    print(df.groupby(["device", "label"]).size())
    m2 = df[df.device == "m2"]
    print("\nfeature               normal mean±sd        degraded mean±sd      Welch p   Cohen d")
    for c in ml.THESIS_FEATURES + ["power", "v_max", "u_b", "te_corrected", "ne_corrected_mean"]:
        a, b = m2[m2.label == 0][c], m2[m2.label == 1][c]
        p = stats.ttest_ind(a, b, equal_var=False).pvalue
        d = (b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2)
        print(f"{c:18s} {a.mean():10.4g}±{a.std():<9.3g} {b.mean():10.4g}±{b.std():<9.3g} {p:9.2g} {d:+7.2f}")
    models = ml.thesis_models()
    rows = []
    for fs_name, feats in (("thesis14", ml.THESIS_FEATURES), ("corrected", CORRECTED)):
        tr, te = train_test_split(m2, test_size=0.2, stratify=m2.label, random_state=0)
        splits = {"random 80/20 within m2": (tr, te),
                  "train m2 -> test m1": (df[df.device == "m2"], df[df.device == "m1"]),
                  "train m1 -> test m2": (df[df.device == "m1"], df[df.device == "m2"])}
        for setting, (a, b) in splits.items():
            for name, m in models.items():
                est = clone(m).fit(a[feats], a.label)
                r = ml.evaluate(b.label.to_numpy(), est.predict(b[feats]), est.predict_proba(b[feats])[:, 1])
                rows.append({"features": fs_name, "setting": setting, "model": name, **r})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res[["features", "setting", "model", "tn", "fp", "fn", "tp", "accuracy", "f1_binary", "auc"]]
          .round(3).to_string(index=False))
    print("\nsingle-feature AUC (no model):")
    for c in ("v_rms", "v_max", "u_b", "i_rms", "n_discharges"):
        s = 1 if roc_auc_score(m2.label, m2[c]) >= 0.5 else -1
        m1 = df[df.device == "m1"]
        print(f"  {c:12s} m2 {roc_auc_score(m2.label, s * m2[c]):.3f}   m1 {roc_auc_score(m1.label, s * m1[c]):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
