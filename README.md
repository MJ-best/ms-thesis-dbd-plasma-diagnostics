# [M.S. Thesis] Electrical diagnostics and ML prediction of dielectric surface damage in atmospheric-pressure DBD plasma

**English** | [한국어](README.ko.md)

[![tests](https://github.com/MJ-best/ms-thesis-dbd-plasma-diagnostics/actions/workflows/tests.yml/badge.svg)](https://github.com/MJ-best/ms-thesis-dbd-plasma-diagnostics/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![code: MIT](https://img.shields.io/badge/code-MIT-green)
![docs: CC BY 4.0](https://img.shields.io/badge/docs-CC%20BY%204.0-lightgrey)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MJ-best/ms-thesis-dbd-plasma-diagnostics/blob/main/notebooks/tutorial.ipynb)

This repository is the **open re-implementation of every computation in the M.S. thesis below**. It contains:

- the full thesis PDF and an erratum;
- a verification report on the thesis;
- a tutorial notebook (runs in Colab) and a web demo app;
- the Python package `dbd_diagnostics`, which consolidates the MATLAB, Python and notebook code written in 2020–2023 and previously spread over several repositories.

| | |
|---|---|
| Title | Electrical measurements and prediction of dielectric barrier surface damage in nonthermal atmospheric pressure dielectric barrier discharge plasma based on machine learning |
| Korean title | 저온 대기압 방전 플라즈마의 유전 장벽 표면 손상에 대한 전기적 측정과 머신러닝 분석 기반 손상 예측 |
| Author | Minjun Kim (김민준) |
| Degree | M.S. (Master of Science), Dept. of Electrical and Biological Physics, Graduate School, Kwangwoon University, Seoul; approved December 2023 |
| Advisor | Prof. Eun Ha Choi |
| Full text | [thesis/Kim_2023_MS_thesis_DBD_surface_damage_ML.pdf](thesis/Kim_2023_MS_thesis_DBD_surface_damage_ML.pdf) (in Korean) · [DBpia](https://www.dbpia.co.kr/journal/detail?nodeId=T16940279) |
| Erratum | [docs/erratum.md](docs/erratum.md) · [PDF (EN)](thesis/Kim_2023_MS_thesis_erratum_en.pdf) · [PDF (KO)](thesis/Kim_2023_MS_thesis_erratum_ko.pdf) |

---

## Contents

1. [Overview](#1-overview)
2. [Abstract](#2-abstract-as-published)
3. [Pipeline](#3-pipeline)
4. [Key results and verification](#4-key-results-and-verification)
5. [Installation and usage](#5-installation-and-usage)
6. [Repository layout](#6-repository-layout)
7. [Documentation](#7-documentation)
8. [Provenance: original repositories and drafts](#8-provenance-original-repositories-and-drafts)
9. [Data](#9-data)
10. [Citation](#10-citation)
11. [License](#11-license)

---

## 1. Overview

Atmospheric-pressure **micro-gap surface DBDs** are cold-plasma sources for cleaning, surface treatment, medicine and agriculture. They fail in three steps:

1. The dielectric that covers the electrodes erodes.
2. The exposed electrode makes the discharge unstable.
3. The discharge eventually stops.

The thesis proposes to diagnose and predict this damage **from the voltage and current waveforms alone**, without extra sensors.

**Device.**
- Coplanar electrodes: 126 pairs of 150 µm × 300 µm Ag pads facing each other across a 100 µm gap.
- Electrodes about 10 µm thick, under a screen-printed dielectric about 40 µm thick.
- Operated in ambient air at about 46.5 kHz and about 1.5 kV_rms.

**Measurement.**
- Tektronix P6015A HV probe and P6021A current probe.
- GW Instek GDS-2102A oscilloscope: 10 ns sampling, 5000 points per capture, 8-bit.

**Experiment.**
- Two devices (m1, m2) were run continuously until the discharge stopped.
- About 20 captures were recorded every 5 h.
- Damage was confirmed by optical microscopy, SEM and EDS: Ag rose from 0.57 % to 13.86 % at the damaged site.

**Analysis.**
- Fourteen features are computed per capture, including V_rms, I_rms, energy, Lissajous capacitances, transferred charge, onset and breakdown voltage, number of discharges, n_e and T_e.
- LR, DT, RF, GB, SVM and kNN classifiers label each surface as **normal** or **degraded**.

## 2. Abstract (as published)

> This study employed electrical measurements and artificial intelligence algorithms to diagnose and predict dielectric surface damage influencing the stable discharge of atmospheric pressure dielectric barrier discharge (DBD) plasma. Plasma discharge characteristics, including effective voltage, effective current, discharge energy per second, discharge initiation voltage, discharge voltage, number of discharges, electron density, electron temperature, capacitance, and discharge charge were calculated through electrical measurements of DBD plasma. Changes in plasma discharge characteristics before and after surface damage were observed. Surface damage led to a decrease in effective voltage, effective current, discharge initiation voltage, discharge voltage, and electron temperature, while increasing number of discharges, electron density, capacitance, and discharge charge. The damage to the dielectric on the electrode surface was confirmed through optical microscopy, scanning electron microscopy (SEM), and energy-dispersive X-ray spectroscopy (EDS) analysis. Machine learning models were trained to learn the variations in plasma discharge characteristics due to surface damage. Using logistic regression, decision tree, random forest, support vector machine, and k-nearest neighbors algorithms, the study classified and predicted surface damage based on plasma discharge characteristic parameters. This research introduces a method to predict and diagnose dielectric surface damage, referring to damage on the dielectric, which influences the lifespan of atmospheric pressure dielectric barrier discharge plasma devices in physics-related contexts.

Some statements in this abstract are corrected in the [verification report](docs/thesis-review.md): the "energy per second", the values of the discharge voltage and electron temperature, and the claimed changes in the number of discharges, n_e and I_rms. The V_rms change is about half as large once a control device is used. See [section 4](#4-key-results-and-verification).

## 3. Pipeline

```
oscilloscope CSV (t, v, i)
 │  io.load_csv            parses the GW Instek Format 2.0A header (probe ratio, V/div, sampling)
 ▼
period split + offsets      waveform / electrical: every full period, then averaged
 │
 ├─ electrical              V_rms, I_rms, energy per cycle E = ∫v·i dt, mean power P = E·f
 ├─ Lissajous (Q–V)          lissajous: C_total, C_dielectric, C_gas, transferred charge ΔQ,
 │                            discharge onset, burning voltage U_b, Manley energy
 ├─ discharge pulses         discharge: number of discharges, onset voltage (thesis + noise-referenced)
 └─ plasma parameters        plasma: Paschen, E/N, mobility, n_e, T_e (D/μ table)
 ▼
feature table (features.extract)   14 thesis features + corrected features
 ▼
machine learning (ml)      Pipeline(StandardScaler, model), cross-device evaluation,
                           binary vs macro metrics, per-device baseline normalisation
```

All equations are in [docs/methods.md](docs/methods.md).

![Ideal equivalent-circuit waveforms and Lissajous figure](docs/img/synthetic_waveform_lissajous.png)

*Ideal equivalent-circuit example from the built-in simulator. Thesis Eq. (15), V_min + Q_max/C_d (red), returns the peak applied voltage (2.24 kV). The burning voltage U_b across the gas (blue) is 0.90 kV.*

## 4. Key results and verification

The equations, numbers, statistics, ML protocol and references of the thesis were re-checked, and the raw captures of devices m1 and m2 were re-analysed with this package. The complete raw-data archive was then checked against a third device, m3, which was discharged alongside them under the same conditions, was not damaged, and serves as the healthy comparison device. Full details are in the **[verification report](docs/thesis-review.md)** and the **[raw-data verification](docs/raw-data-verification.md)**.

| | Thesis | Re-analysis / correction |
|---|---|---|
| Measured quantities | E 0.13 mJ, C_d 34.2 pF, Q_max 59.4 nC | 0.131 mJ, 34.5 pF, 59.5 nC — **reproduced** |
| Energy | "0.13 mJ per second" | **0.13 mJ per cycle**, i.e. a mean power of **≈ 6 W** |
| Breakdown voltage | V_min + Q/C_d = 2.1 kV | That formula equals V_max. The burning voltage is **U_b ≈ 0.81–0.86 kV**, consistent with Paschen (0.72–0.87 kV). |
| Electron temperature | 13 eV, at E/N = 814 Td (OriginLab extrapolation of data that end at 175 Td) | **≈ 6 eV**, at E/N ≈ 330 Td (just beyond the end of the source curve, 300 Td) |
| Changes after damage | Number of discharges ↑, n_e ↑, I_rms ↓ | Number of discharges and n_e are not significant on the thesis' own numbers (p = 0.89, 0.78). I_rms is not significant in the re-analysis (p = 0.27) and depends on the oscilloscope's A/div setting. Against the undamaged control m3, only the V_rms decrease (≈ −48 V, about half of the raw drop) holds for both damaged devices. |
| Classification | RF: 96.9 % on the validation device | Cross-device evaluation: DT/RF/GB 95–99 %. Random splits are optimistic (100 %). **V_rms alone gives AUC ≈ 1.0.** But the models call 95–100 % of the undamaged m3's captures in its low-voltage sessions "degraded"; referenced to m3, cross-device accuracy is 65–78 % (DT/RF/GB; 64–89 % over all six models). |
| Precursor (Sec. 5.7) | A ≥ 10 % drop of the normal probability signals impending damage | Not supported: Tables 7–8 show 0.886 → 0.909 and 0.816 → 0.818. |
| EDS (Table 3) | Ag 0.57 → 13.86 % | Reproduced exactly; the values are **at.%** from single 10,000× maps (normal = Ref sample). Over all maps, damaged sites give 1.3–13.9 at.% and intact dielectric 0.0–0.7 at.%. |

**Bottom line.** The damage is real and leaves a measurable electrical trace, but a weaker one than the thesis reports: about half of the V_rms drop is a supply variation shared with an undamaged device, so an alarm needs a control device or a same-session reference. Five points need correction:
- the interpretation of V_b, T_e and n_e;
- the precursor claim;
- the statistical wording;
- the evaluation protocol;
- the separation of damage from session and oscilloscope-setting effects.

## 5. Installation and usage

```bash
git clone https://github.com/MJ-best/ms-thesis-dbd-plasma-diagnostics.git
cd ms-thesis-dbd-plasma-diagnostics
pip install -e ".[dev]"     # numpy, scipy, pandas, scikit-learn (+ matplotlib, pytest)
pytest                      # 24 tests
```

Command-line tool:

```bash
dbd-diag simulate -o demo.csv
dbd-diag analyse demo.csv
dbd-diag extract captures/ -o features.csv
python examples/demo_synthetic.py
```

The four commands do the following, in order:
1. Write a synthetic 126-cell DBD capture through an emulated 100 MS/s, 8-bit scope.
2. Print every quantity for one capture.
3. Extract features from all captures in a folder, recursively, one row per capture.
4. Reproduce the figure above.

Python API:

```python
from dbd_diagnostics import io, features, ml

wf = io.load_csv("ALL0001.CSV")          # GW Instek CSV or a plain t,v,i CSV
r = features.extract(wf)                  # averaged over full periods
print(r["power"], r["c_dielectric"], r["u_b"], r["te_corrected"])

# cross-device evaluation (df: feature table, label 0 normal / 1 degraded, device id)
res = ml.group_holdout(df, ml.THESIS_FEATURES, label="label", group="device")
print(ml.results_table(res))
```

`python examples/reanalyse_thesis_data.py <m1_data> <m2_data>` reproduces section H of the verification report. `python examples/verify_raw_archive.py "<archive>/전류전압 측정" "<archive>/전압에 따른 QV 변화/I 기준으로 계산"` reproduces the control-device analysis of the [raw-data verification](docs/raw-data-verification.md).

Training and prediction on your own labelled data:

```bash
dbd-diag extract captures/ -o features.csv      # 1) extract features
# 2) add a `label` column (0 normal / 1 degraded) and a `device` column
dbd-diag train features.csv -o model.joblib     # 3) prints leave-one-device-out results, then fits on all rows
dbd-diag predict model.joblib new_captures/     # 4) classify new captures
```

### Tutorial notebook

[`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb) walks through the whole chain step by step: waveform, Lissajous figure, U_b, pulses, E/N and T_e, and cross-device ML. Use the **Open in Colab** badge at the top to run it in the browser without installing anything.

### Web demo app

```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

The app offers:
- Upload of oscilloscope CSVs, or generation of a synthetic capture with sliders for the equivalent-circuit parameters.
- Waveform, gas-gap voltage, Lissajous figure and a table of all quantities, downloadable as CSV.
- A **health check** against the same device's healthy baseline: ΔV_rms, ΔC_dielectric and ΔU_b, with an alarm on a V_rms drop.
- Classification with a model trained by `dbd-diag train`.

To host it, select this repository on [Streamlit Community Cloud](https://streamlit.io/cloud) and set the main file to `app/streamlit_app.py`. The dependencies are listed in `app/requirements.txt`.

## 6. Repository layout

```
thesis/                         full thesis PDF (CC BY-NC-ND 4.0), erratum PDFs (EN/KO)
app/                            Streamlit web demo
src/dbd_diagnostics/
  io.py          oscilloscope CSV reader
  waveform.py    period detection, integer-period windows
  electrical.py  RMS, energy per cycle, power, offset-free charge integration
  lissajous.py   Q-V analysis, equivalent circuit, burning voltage, Manley energy
  discharge.py   pulse detection (thesis + noise-referenced), number of discharges, onset voltage
  plasma.py      Townsend/Paschen, E/N, mobility, n_e, T_e
  features.py    one-call feature extraction (thesis + corrected features)
  ml.py          classifiers, explicit metrics, group hold-out, baseline normalisation
  simulate.py    synthetic waveforms (ideal Manley model, 126-cell stochastic model)
  plots.py       waveform and Lissajous figures
  cli.py         `dbd-diag` command (extract, analyse, simulate, train, predict)
  data/          D/mu table digitised from thesis Fig. 15
docs/            methods, verification report, erratum (EN/KO), legacy-code map, figures
notebooks/       step-by-step tutorial (Colab)
examples/        synthetic demo, re-analysis, raw-archive verification
tests/           24 tests: known-answer equivalent-circuit tests of every formula, CLI and app
```

## 7. Documentation

| Document | Content |
|---|---|
| [docs/methods.md](docs/methods.md) · [KO](docs/methods.ko.md) | Every equation, its assumptions and the thesis equation it corresponds to |
| [docs/thesis-review.md](docs/thesis-review.md) · [KO](docs/thesis-review.ko.md) | Verification report: Eqs. (1)–(32), numbers, measurement chain, statistics, ML, references, audit of the original code, re-analysis |
| [docs/raw-data-verification.md](docs/raw-data-verification.md) · [KO](docs/raw-data-verification.ko.md) | Verification against the complete raw-data archive: control device m3, session effects, oscilloscope settings, micrographs, EDS sources |
| [docs/erratum.md](docs/erratum.md) · [KO](docs/erratum.ko.md) | Thesis erratum: page, location, correction. PDFs are in `thesis/` |
| [notebooks/tutorial.ipynb](notebooks/tutorial.ipynb) | Step-by-step tutorial (runs in Colab) |
| [docs/legacy-code-map.md](docs/legacy-code-map.md) | How the old algorithms map to the new modules, and which bugs were fixed |

## 8. Provenance: original repositories and drafts

This repository consolidates the six repositories below. They are **private** and remain unchanged, with their full history. Reference commits are the default-branch heads at the time of consolidation (2026-11-01).

| Repository | Period | Commits | Reference commit | Content |
|---|---|---|---|---|
| [`plasma-lab`](https://github.com/MJ-best/plasma-lab) | 2020-09 | 5 | `master` @ `71c968d` | First Python port: CSV reader, piecewise-linear Lissajous fit |
| [`plasma-calc`](https://github.com/MJ-best/plasma-calc) | 2020-09 – 10 | 21 | `master` @ `a42f0df` | MATLAB analysis from first version to modular version, ozone scripts, early Python |
| [`Atmospheric_Plasma_Diagnosis`](https://github.com/MJ-best/Atmospheric_Plasma_Diagnosis) | 2020-12 – 2023-02 | 10 | `master` @ `980cc2c` | All MATLAB code (a superset of plasma-calc), 2021 Python automation |
| [`Plamsa_diagnosis_with_ML`](https://github.com/MJ-best/Plamsa_diagnosis_with_ML) | 2022-11 – 2023-06 | 44 | `main` @ `3e8a5ef` | 32 notebooks from the thesis period: features, ML, FFT research, app |
| [`Plasma_diagnosis_with_ML`](https://github.com/MJ-best/Plasma_diagnosis_with_ML) | 2023-03 | 1 | `master` @ `3252884` | Raw captures (about 650 CSVs, 181 MB) |
| [`oscilloscope-calc`](https://github.com/MJ-best/oscilloscope-calc) | 2025-10 – 2026-11 | 14 | `master` @ `a9b23b2` | Newest notebooks and a Flutter app (Dart port). Its calculator was later aligned with this package in PR #2 (`master` @ `9b76a80`) |

The intermediate reorganisation, which still includes a `legacy/` folder, is on branch `claude/thesis-automation-review-3wo35l` of `Atmospheric_Plasma_Diagnosis` (commit `09233c0`).

Where to find each draft:

| Topic | Repository | Path |
|---|---|---|
| First MATLAB analysis (manual `ginput` period, V_b from the intersection of the Lissajous corner lines) | plasma-calc | `MATLAB plasma code/20200718 플라즈마 계산코드/thanksgod1.m`, `thanksgod2.m` |
| MATLAB modularisation, first version | plasma-calc | `MATLAB plasma code/20200829 데이터 분석 모듈화/` |
| MATLAB final modular set (also in `Atmospheric_Plasma_Diagnosis/MATLAB_plasma_code/`) | plasma-calc | `MATLAB plasma code/20201014 새롬바이오 분석코드/csv_to_MATLAB_ginput.m`, `one_period_data_anlaysis.m`, `IV_QV_graph.m`, `save_data.m` |
| Non-sinusoidal energy | plasma-calc | `MATLAB plasma code/20201014 새롬바이오 분석코드/[non-sinusoidal_energy_calc] 비정현파플라즈마계산코드.zip` |
| Ozone analysis (outside the thesis) | plasma-calc | `MATLAB plasma code/20201021 오존데이터분석/` |
| First Python port, piecewise-linear Lissajous fit | plasma-lab | `plasma_calculation.py`, `csv_to_python.py` |
| Early Python (2020) and result plots | plasma-calc | `Python plasma code/plasma.py`, `modules/`, `result_all/` |
| 2021 Python automation | Atmospheric_Plasma_Diagnosis | `Python_plasma_code/Electrical_Diagnosis_20211007.py` (final), `Electrical_Diagnosis_20211006_ver14.ipynb` |
| E/N and T_e draft (unfinished) | Atmospheric_Plasma_Diagnosis | `Python_plasma_code/Electrical_Data_Analysis_20210913_ver10_(Minjun)_최종.ipynb` |
| 2022 three-class ML, LazyPredict | Plamsa_diagnosis_with_ML | `Plasma_Surface_modification_diagnostics_with_ML20221201`–`20221219.ipynb`, `plasma_features*.ipynb` |
| Feature calculation (Q–V regression charge, Peeters ζ_diel) | Plamsa_diagnosis_with_ML | `[feature_Calc]minjun_plasma_diagnosis_with_ML_20230226.ipynb`, `[full_version]…20230225.ipynb` |
| FFT and displacement-current research | Plamsa_diagnosis_with_ML | `[Research]minjun_plasma_diagnosis_with_ML_20230227.ipynb`, `…20230228.ipynb` |
| ML model comparison (AutoViz, DT pruning) | Plamsa_diagnosis_with_ML | `[ML_DL_models]minjun_plasma_diagnosis_with_ML_20230226.ipynb` |
| **Final physics notebook** (with F-tests) | oscilloscope-calc | `[Function]minjun_plasma_diagnosis_with_ML_20230613.ipynb` (earlier: Plamsa's `[Function]…20230425.ipynb`) |
| **Final ML training and Gradio app** | oscilloscope-calc | `[App]minjun_plasma_diagnosis_with_ML_20230425.ipynb`: cells 3–66 train and validate, cell 68 is the app. It is a superset of Plamsa's file of the same name. |
| Raw data | Plasma_diagnosis_with_ML | `plasma_data/m2_data/` (training device, 5–40 h), `m1_data/` (validation device, data05–data25), `m3_data/` (5–60 h, unused), `KaptonDBD/`, `uDBD_Air_1.5lpm/`, `uDBD_N60_1.5lpm/` |
| Old three-class feature table | Plasma_diagnosis_with_ML | `plasma_data/m2_data/20200724 M2 40시간/m2_final_data.csv` |
| Flutter/Dart app | oscilloscope-calc | `lib/domain/usecases/plasma_calculator.dart`, `lib/core/utils/peak_detection.dart`, `lib/data/repositories/csv_parser.dart` |

**Not in any repository.** These files were presumably kept only on the author's machine or Google Drive, and they are needed to reproduce Table 6 exactly:
- `20230426_m1_total_data.csv`, `20230426_m2_total_data.csv`
- `20230226_m2_{system,positive,negative}_data.csv`
- `m1_data_final.csv`, `m2_data_final.csv`
- `rf_clf.pkl`

The electron temperature of the thesis was not computed in code either: the D/μ curve of a figure in Raizer's book was digitised up to 175 Td (Engauge Digitizer) and fitted and extrapolated to 814 Td in OriginLab. The digitised points are kept by the author; the Origin project has not been found.

**Bugs fixed relative to the old code** include the peak-index offset, the V_min polarity mapping, the missing ×f in the Manley value, and using only the last period. The full list is in [docs/legacy-code-map.md](docs/legacy-code-map.md) and in section G of the verification report.

## 9. Data

The raw oscilloscope captures are not included. Every function can be run on synthetic waveforms from `dbd_diagnostics.simulate`. The original data are in the private repository `Plasma_diagnosis_with_ML`. The complete archive of the July 2020 experiment (devices M1–M3 with all sessions, voltage sweeps, micrographs, SEM/EDS reports and the 2020 MATLAB summaries) is kept by the author; its layout is described in [docs/raw-data-verification.md](docs/raw-data-verification.md).

## 10. Citation

Please cite the thesis if you use this code, these methods or the thesis content. GitHub's "Cite this repository" button (`CITATION.cff`) provides the same information.

```bibtex
@mastersthesis{kim2023dbd,
  author = {Kim, Minjun},
  title  = {Electrical measurements and prediction of dielectric barrier surface damage in
            nonthermal atmospheric pressure dielectric barrier discharge plasma based on machine learning},
  school = {Kwangwoon University, Department of Electrical and Biological Physics},
  type   = {M.S. thesis},
  year   = {2023},
  url    = {https://www.dbpia.co.kr/journal/detail?nodeId=T16940279}
}
```

## 11. License

| Part | License | You may |
|---|---|---|
| Code (`src/`, `tests/`, `examples/`, `app/`, `notebooks/`) | [MIT](LICENSE) | Use, modify and redistribute it, including commercially, if you keep the copyright notice. |
| Documentation and figures (`README*`, `docs/`) | [CC BY 4.0](LICENSE-docs.md) | Share and adapt them with attribution. |
| Thesis and erratum PDFs (`thesis/`) | [CC BY-NC-ND 4.0](thesis/README.md) | Share it with attribution, for non-commercial purposes, without modification. Third-party figures inside the thesis (e.g. the source data of Figs. 1 and 15) remain under their original rights. |

© 2020–2026 Minjun Kim (김민준). Kwangwoon University, Dept. of Electrical and Biological Physics. Advisor: Prof. Eun Ha Choi.
