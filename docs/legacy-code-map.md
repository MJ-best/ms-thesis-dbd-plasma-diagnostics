# Legacy code map / 기존 코드 대응표

This repository replaces six private working repositories (2020–2025). Their original contents are preserved in their own histories; nothing was deleted there. Exact paths of every draft and the reference commits are listed in README section 8.

이 저장소는 2020–2025년에 사용한 6개의 비공개 작업 저장소를 하나로 정리한 것입니다. 원래 저장소의 내용과 이력은 각 저장소에 그대로 남아 있으며, 삭제한 것은 없습니다. 초안별 정확한 경로와 기준 커밋은 README 8절에 있습니다.

## Repositories / 저장소

| Repository | Period | Content | Status |
|---|---|---|---|
| `plasma-lab` | 2020-09 | First Python port: CSV reader and piecewise-linear Lissajous fit (V_fire, C_t, C_d) | Superseded by `lissajous.py` |
| `plasma-calc` | 2020-09 – 10 | MATLAB V–I/Q–V analysis (period chosen by hand with `ginput`), ozone-data scripts, the `plasma-lab` modules again | Superseded; the ozone scripts are outside the thesis |
| `Atmospheric_Plasma_Diagnosis` | 2020-12 – 2023-02 | Superset of the `plasma-calc` MATLAB code; Python `Electrical_Diagnosis_20211007.py`; 2021 notebooks | Superseded; interim reorganisation with `legacy/` on branch `claude/thesis-automation-review-3wo35l` |
| `Plamsa_diagnosis_with_ML` | 2022-11 – 2023-06 | 32 dated notebooks: feature calculation, ML models, FFT/displacement-current research, Gradio app | Superseded by `features.py`, `ml.py`, `cli.py` |
| `Plasma_diagnosis_with_ML` | 2023-03 | Raw oscilloscope CSVs (m1, m2, m3, Kapton, µDBD), about 181 MB | Data only; not redistributed here |
| `oscilloscope-calc` | 2025-10 | Flutter/Dart port of the 2023 notebook, plus the newest notebooks (`[Function]…20230613`, `[App]…20230425`) | Separate app. Its Dart calculator was aligned with this package in PR #2 (merged into `master` as `9b76a80`): U_b, transferred charge, Manley power in W, n_e; cross-checked by `test/plasma_calculator_physics_test.dart` |

## Algorithm mapping / 알고리즘 대응

| Quantity | Last legacy implementation | New module | Changes |
|---|---|---|---|
| CSV reading | `pd.read_csv(...).drop(range(17))` in the notebooks | `io.load_csv` | Parses the header; works with either channel order |
| Period selection | `np.where(V == 0)`; only the last period is used | `waveform`, `features._periods` | Schmitt-trigger crossings; averages all full periods |
| Offsets | Current mean removed, voltage not | `electrical.remove_offset` | Removes both |
| V_rms, I_rms | Left Riemann sum | `electrical.rms` | Trapezoid rule |
| "Energy" | Σ v·i·dt, which is J/cycle but labelled per second in the thesis | `energy_per_cycle`, `mean_power` | Explicit J/cycle and W |
| Q(t) | Cumulative sum; constant taken at the zero crossing | `electrical.charge` | Centred, offset-free |
| C_total, C_dielectric | `curve_fit` on the T-windows taken from the pulses | `lissajous.analyse` | Two-segment least squares on each branch, with SE and R² |
| Q_max | `max|Q[T0:T2]| + max|Q[T2:T4]|`, which depends on the offset | `LissajousResult.dQ` | Q(V_max) − Q_on |
| V_breakdown | `1.4*(Cdiel/Cgas)**2/np.log(Freq)` (not a voltage); the thesis Eq. 15 appears in no code | `U_b_*`, `thesis_breakdown_voltage` | Burning voltage from the equivalent circuit |
| Pulse detection | `find_peak_and_width` on a slice; **indices used without the slice offset** | `discharge.thesis_peaks` (fixed), `robust_peaks` | Offset fixed; noise-referenced alternative |
| V_min | `max(V[peaks][V<0])`, **polarity swapped** when `sig_sine` is true | `discharge.pulse_statistics` | Taken per half cycle, by polarity |
| Manley | `4*Cd*Vmin*(Vmax-Vmin)/(1+Cg/Cd)`, J per half cycle mislabelled as power | `lissajous.manley_energy` | Correct per-cycle formula |
| n_e | Applied voltage, area π(0.01 cm)², offset bug | `plasma.electron_density`, `features` | Uses U_b and i_dis; area is configurable |
| T_e | Not implemented (hard-coded D/μ stub); the thesis values came from an OriginLab fit and extrapolation of the D/μ curve | `plasma.electron_temperature_from_field` | Digitised D/μ table; extrapolation off by default |
| ML | No scaler, random split, 10,000 trees, macro/binary metrics mixed | `ml` | Scaler inside the Pipeline, grouped evaluation, explicit metrics |
| App | Gradio (`gr.inputs`, hard-coded paths) | `cli.py` (`dbd-diag`) | Command-line interface; can be wrapped in a GUI if needed |
