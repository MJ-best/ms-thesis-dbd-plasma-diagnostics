"""Command-line interface.

    dbd-diag extract  <csv-or-dir> ... -o features.csv   # batch feature extraction
    dbd-diag simulate -o demo.csv [--ideal]               # synthetic capture
    dbd-diag analyse  <csv>                               # print one capture's results
    dbd-diag train    features.csv -o model.joblib [--group device]   # fit (and cross-validate) a model
    dbd-diag predict  model.joblib <csv-or-dir> ...       # classify captures with a saved model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import features, io, ml, simulate


def _iter_csv(paths):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(x for x in p.rglob("*") if x.suffix.lower() == ".csv")
        else:
            yield p


def cmd_extract(args) -> int:
    cfg = features.DeviceConfig(peak_method=args.peaks)
    rows = []
    for f in _iter_csv(args.inputs):
        try:
            wf = io.load_csv(f)
            if wf.dt > args.max_dt:
                print(f"skip {f} (sampling period {wf.dt:.2e} s > {args.max_dt:.0e} s)", file=sys.stderr)
                continue
            r = features.extract(wf, cfg)
            r["folder"] = f.parent.name
            rows.append(r)
        except Exception as exc:  # keep going: one bad file must not abort a batch
            print(f"error {f}: {exc}", file=sys.stderr)
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"wrote {len(df)} rows -> {args.output}")
    return 0


def cmd_simulate(args) -> int:
    wf = simulate.ideal() if args.ideal else simulate.cells(seed=args.seed)
    io.save_csv(wf, args.output)
    print(f"wrote {len(wf)} samples -> {args.output}")
    return 0


def cmd_analyse(args) -> int:
    r = features.extract(io.load_csv(args.input), features.DeviceConfig(peak_method=args.peaks))
    units = {"frequency": "Hz", "v_rms": "V", "i_rms": "A", "energy": "J/cycle", "power": "W",
             "c_total": "F", "c_dielectric": "F", "c_gas": "F", "q_max": "C", "v_max": "V",
             "v_min": "V", "u_b": "V", "v_breakdown": "V (thesis Eq.15, ~V_max)", "en_td": "Td",
             "te": "eV (thesis chain, extrapolated)", "te_corrected": "eV", "ne_mean": "cm^-3",
             "ne_corrected_mean": "cm^-3"}
    for k, v in r.items():
        print(f"{k:22s} {v!s:>24s} {units.get(k, '')}" if not isinstance(v, float)
              else f"{k:22s} {v:24.6g} {units.get(k, '')}")
    return 0


def cmd_train(args) -> int:
    df = pd.read_csv(args.features)
    feats = args.columns.split(",") if args.columns else ml.THESIS_FEATURES
    if args.group and args.group in df.columns and df[args.group].nunique() > 1:
        models = {args.model: ml.thesis_models()[args.model]}
        res = ml.results_table(ml.group_holdout(df.dropna(subset=feats), feats, args.label, args.group, models))
        cols = ["model", "test_group", "n", "tn", "fp", "fn", "tp", "accuracy", "f1_binary", "auc"]
        print("leave-one-group-out evaluation:")
        print(res[[c for c in cols if c in res]].round(3).to_string(index=False))
    bundle = ml.train_bundle(df, feats, args.label, args.model)
    ml.save_bundle(bundle, args.output)
    print(f"trained {args.model} on {bundle['n_train']} rows {bundle['class_counts']} -> {args.output}")
    return 0


def cmd_predict(args) -> int:
    bundle = ml.load_bundle(args.model)
    cfg = features.DeviceConfig(peak_method=args.peaks)
    rows = []
    for f in _iter_csv(args.inputs):
        try:
            r = features.extract(io.load_csv(f), cfg)
            r["file"] = str(f)
            rows.append(r)
        except Exception as exc:
            print(f"error {f}: {exc}", file=sys.stderr)
    df = pd.DataFrame(rows)
    pred = ml.predict_bundle(bundle, df)
    out = pd.concat([df[["file"]], pred], axis=1)
    print(out.to_string(index=False))
    if args.output:
        out.to_csv(args.output, index=False)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dbd-diag", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="batch feature extraction")
    e.add_argument("inputs", nargs="+")
    e.add_argument("-o", "--output", default="features.csv")
    e.add_argument("--peaks", choices=["robust", "thesis"], default="robust")
    e.add_argument("--max-dt", type=float, default=5e-8,
                   help="skip captures sampled slower than this (overview captures)")
    e.set_defaults(func=cmd_extract)
    s = sub.add_parser("simulate", help="write a synthetic DBD capture")
    s.add_argument("-o", "--output", default="synthetic_dbd.csv")
    s.add_argument("--ideal", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_simulate)
    a = sub.add_parser("analyse", help="analyse one capture")
    a.add_argument("input")
    a.add_argument("--peaks", choices=["robust", "thesis"], default="robust")
    a.set_defaults(func=cmd_analyse)
    t = sub.add_parser("train", help="train a classifier on a feature table")
    t.add_argument("features", help="CSV from `dbd-diag extract` with a label column added")
    t.add_argument("-o", "--output", default="model.joblib")
    t.add_argument("--label", default="label", help="0 = normal, 1 = degraded")
    t.add_argument("--group", default="device", help="column for leave-one-group-out evaluation")
    t.add_argument("--model", choices=["LR", "DT", "RF", "GB", "SVM", "kNN"], default="RF")
    t.add_argument("--columns", help="comma-separated feature columns (default: the 14 thesis features)")
    t.set_defaults(func=cmd_train)
    p = sub.add_parser("predict", help="classify captures with a saved model")
    p.add_argument("model")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--output")
    p.add_argument("--peaks", choices=["robust", "thesis"], default="robust")
    p.set_defaults(func=cmd_predict)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
