# Thesis verification report

**English** | [한국어](thesis-review.ko.md)

**Subject.** Minjun Kim, *Electrical measurements and prediction of dielectric barrier surface damage in nonthermal atmospheric pressure dielectric barrier discharge plasma based on machine learning*, M.S. thesis, Dept. of Electrical and Biological Physics, Kwangwoon University, 2023 ([DBpia](https://www.dbpia.co.kr/journal/detail?nodeId=T16940279)).

**Scope.** This review covers:
- Equations (1)–(32).
- Internal consistency of all numbers in the text, tables and figures.
- The measurement chain.
- The statistical and ML evaluation.
- The references.
- Whether the six original code repositories match the thesis.
- A re-analysis of the original oscilloscope captures (devices m1, m2).
- A check against the complete raw-data archive, using a third device (m3) that ran in parallel without visible damage as a control (section J).

**Methods.**
1. Re-derivation of each equation.
2. Recomputation from the numbers printed in the thesis.
3. Equivalent-circuit simulations with a known ground truth (`tests/test_lissajous.py`).
4. Re-analysis of the raw data with this package (`examples/reanalyse_thesis_data.py`).
5. Reverse-engineering of the confusion matrices behind Table 6.
6. Comparison with the undamaged control device m3 measured in the same sessions (`examples/verify_raw_archive.py`).

**Severity levels.**
- **[Major]**: affects a conclusion or a headline number.
- **[Moderate]**: the method or wording must change, but the conclusion stands.
- **[Minor]**: notation, typos or reference formatting.

---

## 0. Summary

| # | Finding | Severity |
|---|---|---|
| 1 | Under the thesis' own equivalent circuit, Q_max = C_d (V_max − V_min). So the "breakdown voltage" of Eq. (15), V_b = V_min + Q_max/C_dielectric, is **identically V_max**, the peak applied voltage. The actual gas burning voltage is U_b ≈ **0.8–1.1 kV**, consistent with Paschen (0.72–0.87 kV for a 100 µm air gap). The thesis reports 2.1 kV. E/N, T_e and n_e inherit this error. | Major |
| 2 | The thesis T_e of 13 eV is read at E/N = 81.38×10⁻¹⁶ V cm² (= **814 Td**), in the **extrapolated** part of Fig. 15; the data end at ≈ 390 Td. Using U_b instead gives E/N ≈ 340 Td and T_e ≈ **6 eV**. The thesis T_e is also a deterministic function of V_b (r = 0.9995), so it is not an independent observation. | Major |
| 3 | The "discharge energy per second, 0.13 mJ" is the **energy per cycle**. The mean power is 0.13 mJ × 46.5 kHz ≈ **6.0 W** (power factor ≈ 0.19). Eq. (11), with its 1/T factor, defines power in W, which does not match the reported unit. | Major |
| 4 | Sec. 5.7 concludes that a ≥ 10 % drop of the "normal" probability is a precursor of damage. Tables 7–8 do not support this. Table 7 shows 0.886 → 0.909 (an **increase**); Table 8 shows 0.816 → 0.818 (**no change**). | Major |
| 5 | The abstract and conclusion claim that the number of discharges and n_e increase and that I_rms decreases, but give no statistical test. Testing the thesis' own mean ± SD gives p = 0.89 for the number of discharges and p = 0.78 for n_e, i.e. **not significant**. | Major |
| 6 | Data splitting has two problems. First, both half-cycles of one capture appear to enter as separate rows (inferred from the row arithmetic). Second, the label is constant per session, yet rows were split at random. Test scores are therefore optimistic: in the re-analysis, a random split gives 100 % for 5 of 6 models. | Major |
| 7 | In Table 6, precision and recall are **macro** averages, while F1 is the **binary** F1 of the degraded class. All 12 entries can be reproduced exactly from integer confusion matrices. This mixing contradicts Eqs. (30)–(32). | Moderate |
| 8 | Most of the classification power comes from **V_rms alone**: its single-feature AUC is 0.996 / 1.000. The classifier detects the sag of the applied voltage after damage. About half of that sag is common to an undamaged device measured in the same sessions (#11). | Moderate |
| 9 | Eq. (14) contains an arithmetic slip: 55.6 nC / 30.5 pF = **1.82 kV**, not 1.5 kV. The representative C_d also differs in three places (30.5 / 34.9 / 34.2 pF). | Moderate |
| 10 | The probe models are swapped. The P6015A is the 1000× HV probe and the P6021A is the current probe, as the CSV headers confirm. | Minor |
| 11 | ★ The undamaged device m3, measured in the same sessions, shows the same V_rms drop (−133 V at 15 h). Within that session the applied voltage jumps between discrete levels, i.e. the supply switches mode. Against m3, the damage-specific V_rms change is **≈ −48 V**, about half of the naive drop. Classifiers trained on m2 call **95–100 %** of m3's captures in its low-voltage sessions "degraded". | Major |
| 12 | Within one session, the oscilloscope settings change the features: doubling the current A/div adds 3.5 mA to I_rms, and 20 ns sampling (instead of 10 ns) removes 6.6 pulses and adds 3.0 pF to C_d. I_rms, the pulse count and n_e are therefore confounded with the settings, which differ between the groups. | Major |
| 13 | Table 3 gives **atomic %** from single 10,000× maps; its "Normal" column is the Ref sample. The 100 → 250 mA current increase of Fig. 7 is not reproduced when captures at the same current scale are compared. | Minor |

**What holds.**
- Re-analysing the raw captures reproduces the thesis' measured quantities:

  | Quantity | Thesis | Re-analysis |
  |---|---|---|
  | Energy per cycle | 0.13 mJ | 0.131 mJ |
  | C_d | 34.2 pF | 34.5 pF |
  | Q_max | 59.4 nC | 59.5 nC |
  | V_rms (normal) | 1.5 kV | 1489 V |

- With strict cross-device evaluation (train on one device, test on the other), DT, GB and RF still detect damage with 95–99 % accuracy. Part of this is a session effect common to all devices: after referencing the features to the control m3, the accuracy is 65–78 % (DT/RF/GB; 64–89 % over all six models) (section J).
- The damage itself is real (micrographs, EDS), and both damaged devices lose about 48 V more V_rms than the control.
- The central claim, that electrical diagnostics plus ML can detect dielectric surface damage, stands.
- What must change is the physical interpretation of V_b, T_e and n_e, the precursor claim, the statistical wording, the evaluation protocol, and the separation of damage from session and measurement-setting effects.

---

## A. Theory (Chapter 1)

**A1. Name of Eq. (7) [Minor].**
- α = A p exp(−B p d/V) is **Townsend's semi-empirical formula**, not the Saha ionisation equation.
- The Saha equation relates ionisation degree to temperature in thermal equilibrium.
- Cite Raizer [49] for the air constants.

**A2. Eqs. (1)–(8) [OK].**
- The self-sustainment condition γ(e^{αd} − 1) = 1 is correct.
- The Paschen law of Eq. (8) is correct.

**A3. "100 µm gap, breakdown voltage of several kV at 1 atm" (Sec. 1.1.4) [Moderate].**
- With A = 15 cm⁻¹ Torr⁻¹, B = 365 V cm⁻¹ Torr⁻¹ and pd = 7.6 Torr cm, Eq. (8) gives **0.72 kV (γ = 0.1) to 0.87 kV (γ = 0.01)**.
- The applied voltage is around 2 kV because of the **capacitive divider** formed by the dielectric and because of the wall charge.
- This sentence is later used to make V_b = 2.1 kV look plausible, so it should be corrected.

**A4. Region below the Paschen curve [Minor].**
- "Too many collisions" explains only the right branch (pd > pd_min).
- On the left branch there are too few collisions.

**A5. Units and typical values [Minor].**
- Write K instead of °K.
- Cold-plasma T_e is typically 1–10 eV (≈ 10⁴–10⁵ K), so "> 2000 K" is too loose.
- DBD filaments have n_e ≈ 10¹⁴–10¹⁵ cm⁻³ (Kogelschatz [61]).

**A6. Plasma definition [Minor].**
- Langmuir coined the term "plasma".
- The quoted definition ("quasi-neutral gas exhibiting collective behaviour") is Chen's [1].

---

## B. Measurement (Chapter 2)

**B1. Probes [Minor].**
- The text says "HV probe P6021A, current probe P6015A".
- In fact, the Tektronix **P6015A** is a 1000× HV probe (20 kV DC, 75 MHz), and the **P6021A** is an AC current probe (120 Hz–60 MHz, 2 or 10 mA/mV).
- The CSV headers agree: CH1 has `Probe Ratio 1.000E+03` with unit V, and CH2 has `Probe Ratio 2.000E+00` with unit A.

**B2. Limits of the measurement chain [Moderate].**
- The CSV headers show:
  - sampling every 10 ns (100 MS/s; some captures 20 ns);
  - 5000 points per capture (≈ 2.3 periods);
  - an 8-bit ADC, giving 40 V/LSB for voltage;
  - 1.6–8 mA/LSB for current, because the current scale was 0.04, 0.1 or 0.2 A/div depending on the session;
  - a current-probe bandwidth of 60 MHz (≈ 6 ns rise time).
- Nanosecond micro-discharges are therefore under-sampled. Pulse counts are lower bounds and peak currents are attenuated.
- V_min is quantised to ±40 V.
- These limits should be stated in the thesis.

**B3. [Minor]** Table 1 gives 19.8 mA, while the mean of the normal condition is 20.8 mA. If they come from different times, say so.

---

## C. Electrical characteristics (Chapter 3)

**C1. Eq. (11) and "energy per second" [Major].**
- P = (1/T)∫₀ᵀ v i dt is the **mean power** in W.
- The reported value of 0.13 mJ (unit J in Fig. 25a) is the **energy per cycle**, E = ∫₀ᵀ v i dt. The original code computes exactly that: `Energy_int += V*I*dt`, with no 1/T.
- The corresponding power is E·f ≈ **6.0 W**, which gives a power factor of ≈ 0.19 against V_rms·I_rms ≈ 31 VA.
- The re-analysis (m2, 40 normal captures) gives E = 0.131 ± 0.029 mJ and P = 6.08 ± 1.28 W.

**C2. Eq. (23) [Moderate].**
- Q = ∫₀ᵀ i dt is a single number. The Lissajous figure needs the running integral Q(t) = ∫₀ᵗ i dt′ + Q₀.
- Removing the current offset is mandatory. A one-LSB offset (8 mA) drifts by **172 nC per period**, about 3× the discharge charge.

**C3. Inconsistent V_min values [Minor].**
- Sec. 3.1.1 gives 300–500 V, the example in Sec. 3.1.3 uses 560 V, and the reported mean is 504.5 ± 168.8 V.
- The caption of Fig. 14 says T₁–T₂, but the plot shows the positive half-cycle (T₃–T₄).

**C4. Arithmetic in Eqs. (14)–(15) [Moderate].**
- 55.6 nC / 30.5 pF = **1.82 kV**, not 1.5 kV.
- Eq. (15) then gives **2.38 kV**, not 2.1 kV.
- The representative C_dielectric appears as 30.5, 34.9 and 34.2 pF in different places.

**C5. ★ Physical meaning of the "breakdown voltage", Eqs. (13)–(15) [Major].**

- **The identity.**
  - In the thesis' circuit (Figs. 17–18), the Q–V slope during discharge is C_dielectric.
  - Q_max (Eq. 25) is the charge moved from V_min to V_max.
  - Hence Q_max = C_d (V_max − V_min), and V_min + Q_max/C_d = **V_max**.

- **Numerical evidence.**
  - The mean V_b of 2.1 kV equals √2 × V_rms = √2 × 1.5 kV = 2.12 kV.
  - 504.5 V + 59.4 nC / 34.2 pF = 2.24 kV ≈ V_max.
  - Re-analysis: the thesis formula gives 2246 ± 81 V, against V_max = 2284 ± 46 V.

- **Cause.**
  - The wall voltage at ignition is set by the charge **remaining at the onset** (Q_on), which was deposited in the previous half-cycle.
  - The thesis instead uses the charge **transferred during** the current half-cycle (ΔQ).

- **Correct relations** (Manley [42]; Pipa & Brandenburg [55]):
  - The gap voltage is u_g(t) = v(t) − Q(t)/C_d.
  - During discharge, u_g = ±U_b (the burning voltage).
  - U_b can be obtained in four equivalent ways:
    1. U_b = V_on − Q_on/C_d (with centred Q);
    2. U_b = (1 − C_total/C_d)(V_max + V_on)/2;
    3. U_b = E_cycle/(2ΔQ);
    4. U_b = the voltage-axis intercept of the discharge line.

- **Checks.**

  | Check | Result |
  |---|---|
  | Thesis means, geometry formula | (1 − 11.3/34.2)(2.24 + 0.50)/2 = **0.92 kV** |
  | Thesis means, energy formula | 0.13 mJ / (2 × 59.4 nC) = **1.09 kV** |
  | Re-analysis (m2) | **860 ± 68 V** (normal) and **811 ± 75 V** (degraded); the energy method gives 1.09 / 1.06 kV |
  | Paschen, 100 µm air | 0.72–0.87 kV |
  | Simulation: C_d 34.2 pF, C_g 16.9 pF, U_b 900 V, V_max 2240 V | Thesis formula **2240 V**; corrected formula **900 V** |

  The simulated operating point also reproduces the thesis means: V_on = 450 V, ΔQ = 61 nC and E = 0.110 mJ.

- **Consequences.**
  - E/N, T_e and n_e must be recomputed.
  - U_b decreases after damage (p = 2×10⁻⁴), but the undamaged control m3 decreases by a similar amount over the same sessions (−38 V, p = 0.05), and U_b depends on the sampling setting (+35 V at 20 ns). U_b is therefore not established as a damage indicator (section J).

**C6. E/N, N and T_e [Major].**

1. **Gas density.**
   - N = 2.8×10¹⁹ cm⁻³ is called "Avogadro's number". It is the gas number density (Loschmidt number) N = p/(k_B T).
   - At 1 atm, N = 2.50×10¹⁹ cm⁻³ at 20 °C and 2.69×10¹⁹ cm⁻³ at 0 °C.
2. **Arithmetic.**
   - With V_b = 2.1 kV and N = 2.8×10¹⁹, E/N = **75.0**×10⁻¹⁶ V cm², not 81.38.
   - 81.38 would require V_b = 2.28 kV, or N = 2.58×10¹⁹.
3. **Units and extrapolation.**
   - 81.38×10⁻¹⁶ V cm² = 814 Td.
   - Digitising Fig. 15 shows that the data end at ≈ 39×10⁻¹⁶ V cm² (≈ 390 Td).
   - The extrapolated curve gives D/μ ≈ 13.3 V at 81.4×10⁻¹⁶ V cm², so the reported 13.2 eV is an **extrapolated value**.
   - Ref. [48] covers slow electrons only.
4. **Corrected values.**
   - With U_b ≈ 0.86 kV and N = 2.50×10¹⁹: E/N ≈ 340 Td (inside the data), D/μ ≈ 6 V, T_e ≈ **6 eV**.
   - Re-analysis: 6.2 ± 0.3 eV (normal) and 6.0 ± 0.4 eV (degraded).
5. **Interpretation.**
   - The Einstein relation assumes a Maxwellian distribution, so "characteristic energy D/μ" is the accurate term.
   - The thesis T_e depends only on V_b (r = 0.9995). "T_e decreases with damage" is therefore the V_max decrease restated in eV.

**C7. Electron density [Moderate].**
- **Voltage.** Eq. (20) uses V_b ≈ V_max. Using U_b raises n_e by about ×2.4.
- **Area.** 4.5×10⁻⁴ cm² is the top area of the cell pad. In a coplanar surface gap the current flows **parallel** to the surface, so the cross-section is the electrode edge (300 µm) × the discharge thickness.
- **Current.**
  - The probe measures the sum over 126 cell pairs, including displacement current.
  - Use the discharge current i_dis = (i − C_total dv/dt)/(1 − C_total/C_d) [55].
  - The reported mean n_e of 1.6×10¹² cm⁻³ corresponds to i ≈ 14 mA, i.e. 2 ADC LSB.
- **Mobility.** μ_e = 592 cm²/Vs is the low-field value; it is lower at hundreds of Td.
- **Original code.** It uses a different area, π(0.01 cm)², uses the applied voltage, and has an index-offset bug (G3).
- **Recommendation.** Report n_e as a cell-averaged relative index.

**C8. Table 2 statistics [Minor].**
- The C_total fitting error should read **7.63×10⁻¹⁴ F**. For a simple regression F = (slope/SE)²:
  - for C_d, 34.9 pF/√(6.74×10⁴) = 0.134 pF, which matches the table;
  - for C_total, 13.1 pF/√(2.94×10⁴) = 0.0764 pF.
- p = 1.11×10⁻¹⁶ = 2⁻⁵³ is the double-precision floor of 1 − CDF. Report it as p < 10⁻¹⁵.
- The fitted points are autocorrelated in time, so the F-test is weak evidence of a good fit. Report R², residuals and slope standard errors instead.
- C_gas = 20.97 pF checks out.

**C9. Interpreting C_dielectric [Moderate].**
- If the discharge does not cover the whole dielectric, the discharge-phase slope is an **effective ζ_diel ≤ C_d** (Peeters & van de Sanden [50]).
- The increase after damage (34.2 → 35.5 pF) more likely reflects a change in discharge coverage, or conduction to the exposed electrode, than a change of the dielectric itself.
- Where the electrode is exposed, the barrier assumption breaks down locally.

**C10. [Minor]** Write mean ± SD in a single unit, e.g. 34.2 ± 4.9 pF rather than "34.2 pF, SD 4.9×10⁻¹²".

**C11. Pulse counting [Moderate].**
- A median-prominence threshold is relative to each capture, which makes counts hard to compare across captures.
- Discarding "too large" peaks removes the strongest real discharges.
- Use a noise-referenced (MAD) threshold, as in `discharge.robust_peaks`.
- Because of B2, the counts are lower bounds in any case.

---

## D. Statistics and machine learning (Chapters 4–5)

**D1. Claimed increases and decreases need tests [Major].**

Welch t-tests on the thesis' own mean ± SD, assuming n = 80 / 209:

| Quantity | p | Cohen d | Verdict |
|---|---|---|---|
| Number of discharges | 0.89 | +0.02 | not significant |
| n_e | 0.78 | +0.03 | not significant |
| Energy | equal means | – | not significant |
| V_rms | 2×10⁻⁷⁶ | – | significant |
| I_rms | 5×10⁻⁴ | – | significant |
| V_min | 0.009 | – | significant |
| V_b | 1.5×10⁻⁵ | – | significant |
| T_e | 4×10⁻⁵ | – | significant |
| C_total | 8×10⁻⁶ | – | significant |
| C_gas | 0.001 | – | significant |
| C_d | 0.04 | – | borderline |
| Q_max | 0.06 | – | borderline |

Re-analysis (m2: 40 normal, 120 degraded captures):

| Quantity | p | Direction |
|---|---|---|
| I_rms | 0.27 | not significant |
| Energy | 0.8 | not significant |
| Number of discharges (MAD 5σ detector) | 6×10⁻⁴ | up |
| U_b | 2×10⁻⁴ | down |

Control check (section J): over the same sessions, the undamaged m3 also gains discharges (+2.7, p = 4×10⁻⁶) and loses U_b (−38 V, p = 0.05), and I_rms and the pulse count depend on the oscilloscope settings. None of the changes in this table is therefore established as an effect of damage. Against m3, only the V_rms decrease (≈ −48 V) is consistent in both damaged devices.

The abstract and conclusion should be revised, and effect sizes reported.

**D2. Table 6 metrics [Moderate].** All 12 entries are reproduced exactly by these confusion matrices (`tests/test_ml.py`). The test set has 16 normal and 42 degraded samples; the validation set has 80 and 81. PRE and REC turn out to be macro averages, while F1 is binary.

| Model | Test TN / FP / FN / TP | Validation TN / FP / FN / TP |
|---|---|---|
| LR | 15 / 1 / 2 / 40 | 42 / **38** / 0 / 81 |
| DT | 13 / 3 / 3 / 39 | 79 / 1 / 5 / 76 |
| RF | 13 / 3 / 1 / 41 | 79 / 1 / 4 / 77 |
| GB | 14 / 2 / 1 / 41 | 79 / 1 / 5 / 76 |
| SVM | 15 / 1 / 2 / 40 | 42 / **38** / 0 / 81 |
| kNN | 11 / 5 / 3 / 39 | 41 / **39** / 1 / 80 |

Report confusion matrices and confidence intervals. Examples (Wilson 95 % CI):
- RF validation: 156/161 = 96.9 % [92.9, 98.7] %.
- RF test: 54/58 = 93.1 % [83.6, 97.3] %.

**D3. Metric definitions in the text [Moderate].**
- Recall is defined as "of those predicted wrong, the fraction actually wrong", which is incorrect. Recall is the fraction of truly damaged samples that are predicted damaged.
- The FN definition repeats the FP definition. FN means "actually damaged (positive), predicted normal (negative)".
- Use "positive (damaged) / negative (normal)" instead of "correct / wrong".

**D4. LR/SVM on the validation set [Moderate].**
- Accuracy is 76.4 % while AUC is 0.99: 38 of the 80 normal samples are called degraded.
- The ranking survives, but the decision threshold **shifts between devices** (domain shift).
- Per-device baseline normalisation (`ml.baseline_normalize`) or threshold calibration fixes this. In the re-analysis, SVM m2 → m1 improves from 91.6 % to 98.9 %.

**D5. Splitting and leakage [Major].**
1. **Half-cycle rows.**
   - The row arithmetic points to one row per half-cycle: 289 = 80 + 209 and 161 = 80 + 81, Table 7 has 40 + 40 rows, and each folder holds 20 captures.
   - The original code also writes `*_positive_data.csv` and `*_negative_data.csv`.
   - So the two halves of one capture can land on both sides of a random split.
2. **Session correlation.**
   - Labels are constant per session, and each session is about 20 captures taken within minutes, so a random 80/20 split is optimistic.
   - In the re-analysis, a random split gives 100 % for 5 of 6 models.
   - Cross-device evaluation gives LR 81–95 %, SVM 86–92 % and kNN 69–75 %.
3. **Confounding with time.**
   - Label 0 always precedes label 1 in time.
   - The current-probe scale changed between sessions (0.04 / 0.1 / 0.2 A/div), so resolution and noise floor encode session identity. Section J quantifies this: within one session, doubling the A/div adds 3.5 mA to I_rms.
4. **Few devices.** With only two devices, generalisation claims are limited.
5. **Code mismatch.**
   - The final notebook in the repositories has no scaler (LR predicts all zeros), uses 10,000 trees and a random split.
   - The code and data files that produced Table 6 (`20230426_m*_total_data.csv`) are not in any repository.

**D6. V_rms dominates [Moderate].**
- Single-feature AUC is **0.996 (m2) and 1.000 (m1) for V_rms**, versus 0.909 and 0.935 for V_max.
- Figs. 28 and 29 show the same thing.
- The classifier detects "damage → load (capacitance and current) change → sag of the resonant supply's output voltage".
- This is a useful practical indicator. Separating it from changes in supply setting or slow drift needs a control experiment.
- The raw-data archive contains such a control: device m3 ran in parallel and shows no erosion. Its V_rms falls in the same sessions (−133 V at 15 h), so about half of the drop is a common-mode supply effect. The damage-specific part is ≈ −48 V, and classifiers trained on m2 call 95–100 % of m3's captures in its low-voltage sessions "degraded" (section J).

**D7. Random forest [Minor].**
- RF does not "select the best of 1000 trees"; it averages the votes or probabilities of all trees (Sec. 5.4 states this correctly).
- Fig. 29 shows a single tree.
- Show permutation importance or SHAP instead.

**D8. Precursor claim in Sec. 5.7 [Major].**
- Table 7: 0.886 → 0.909 (an increase). This is also in-sample, since the data were used for training.
- Table 8: 0.816 → 0.818 (no change).
- The "≥ 10 % drop" rule has no support in these data and should be presented as a hypothesis.

**D9. [Minor] Other points.**
- A DNN is announced in Sec. 5.2 and Fig. 26, but no results are given.
- The features are strongly redundant: C_gas = f(C_d, C_t); V_b ≈ V_max ∝ V_rms; T_e = g(V_b); n_e ∝ i/V_b.
- Typos: "Freqency", "min_samples_splite", "Degradaion"; "Fig. 32" should be Fig. 30.

---

## E. Surface analysis (Chapter 4) [OK]

- The EDS rise of Ag (0.57 → 13.86 %) and the fall of Si, Na and Al support electrode exposure.
- Worth adding:
  - EDS is semi-quantitative.
  - Zn and Pb are constituents of the glass-frit dielectric paste.

---

## F. References

| # | Issue |
|---|---|
| F1 | [10] and [25] are duplicates (Michel et al., J. Appl. Phys. 105, 073302 (2009)). |
| F2 | [26] and [27]: affiliation superscripts are fused into author names (Gimenob → Gimeno, Hansona → Hanson, Goldhawkc → Goldhawk, Simpsonc → Simpson, Lavoiea/b → Lavoie). |
| F3 | [28] is missing the journal: J. Appl. Phys. 124, 183301 (2018). |
| F4 | [13] is incomplete: no title, journal or year. |
| F5 | [1] Chen, "Plenum Press, 2012": check the edition (2nd ed. Plenum 1984; 3rd ed. Springer 2016). |
| F6 | [70] Elkan (cost-sensitive learning) is an odd source for precision, recall and F1. Consider Sokolova & Lapalme, Inf. Process. Manag. 45, 427 (2009). |
| F7 | Eq. (7) should cite Townsend, or Raizer [49]. |
| F8 | Content cross-check: [42] Manley ✓. [55] Pipa & Brandenburg provides the correct U_b relations, but they were not used. [50] Peeters (ζ_diel) relates to C9. [48] covers slow electrons only, so it cannot support values at 814 Td. |

> The network of this review session was restricted, so not every reference could be opened in full. Bibliographic details were checked against commonly available records.

---

## G. Original code vs. thesis (audit of six repositories)

| # | Topic | Finding |
|---|---|---|
| G1 | Eq. (15) | Not in the final Python code. The code instead computes `Vbd = 1.4*(Cdiel/Cgas)**2/np.log(Freq)`, which does not have units of voltage (values range from 0.005 to 2178). The 2020 MATLAB code intersected two Lissajous edge lines. |
| G2 | T_e | No E/N table or D/μ interpolation exists in any repository. |
| G3 | Index-offset bug | Peak indices relative to the slice `x[T0:T1]` are applied to the full-cycle arrays, so the second half-cycle's V, n_e and pulse energy are wrong. Example: M2 5 h, ALL0001 uses −80 V where the true value is 840 V. |
| G4 | V_min polarity | The `sig_sine` logic swaps the polarity pairing: 75 of 160 rows of `m2_final_data.csv` have V_min_p < 0. |
| G5 | Manley "power" | There is no ×f (the result is J/cycle), and the full-cycle factor 4 is applied per half-cycle. |
| G6 | Cycle handling | Only the last period is used, and the voltage DC offset is not removed. |
| G7 | Batch processing | A `try/except` around the loop aborts the whole folder on the first error. |
| G8 | Reproducibility | The markdown results table is stale, and the intermediate CSVs and `rf_clf.pkl` are missing. |

All of these are fixed in this package. See [legacy-code-map.md](legacy-code-map.md).

---

## H. Re-analysis of the original data with this package

Device m2: the 5 h and 10 h captures are normal (n = 40) and the 15–40 h captures are degraded (n = 120). All full periods of each capture are averaged.

| Quantity | Thesis (normal / degraded) | Re-analysis (normal / degraded) | p |
|---|---|---|---|
| Energy per cycle | 0.13 / 0.13 mJ | 0.131 / 0.132 mJ | 0.8 |
| Mean power | – | 6.08 / 6.14 W | 0.8 |
| V_rms | 1.5 / 1.4 kV | 1489 / 1396 V | 6×10⁻²⁷ |
| I_rms | 20.8 / 18.6 mA | 20.2 / 19.2 mA | 0.27 |
| C_dielectric (ζ) | 34.2 / 35.5 pF | 34.5 / 37.2 pF | 0.005 |
| C_total | 11.3 / 12.1 pF | 12.3 / 13.0 pF | 2×10⁻⁴ |
| C_gas | 17.3 / 18.7 pF | 19.4 / 20.3 pF | 0.03 |
| Q_max | 59.4 / 61.7 nC | 59.5 / 62.7 nC | 0.07 |
| V_max | – | 2284 / 2155 V | 6×10⁻¹⁶ |
| V_b (thesis Eq. 15) | 2.1 / 2.0 kV | 2246 / 2118 V (≈ V_max) | 4×10⁻⁹ |
| **U_b (burning voltage)** | – | **860 / 811 V** | 2×10⁻⁴ |
| E/N | 814 Td | 344 / 324 Td | – |
| T_e (thesis chain, extrapolated) | 13.0 / 12.5 eV | 12.3 / 11.7 eV | 4×10⁻⁹ |
| **T_e (U_b, inside the table)** | – | **6.2 / 6.0 eV** | 8×10⁻⁵ |
| Number of discharges | 50.1 / 50.5 (per ¼ period) | 9.7 / 13.4 (per half period, MAD 5σ) | 6×10⁻⁴ |

ML re-evaluation, using the thesis hyper-parameters with the scaler inside the Pipeline and the thesis 14-feature set:

| Setting | DT | RF | GB | LR | SVM | kNN |
|---|---|---|---|---|---|---|
| Random 80/20 within m2 | 100 % | 100 % | 100 % | 100 % | 100 % | 94 % |
| Train m2 → test m1 | 98.9 % | 98.9 % | 95.8 % | 94.7 % | 91.6 % | 74.7 % |
| Train m1 → test m2 | 98.8 % | 95.0 % | 98.8 % | 80.6 % | 86.2 % | 69.4 % |

V_rms alone gives AUC 0.996 on m2 and 1.000 on m1.

Section J repeats these comparisons against the undamaged control m3. The U_b and pulse-count changes also occur in m3, and the cross-device accuracy falls to 65–78 % (DT/RF/GB; 64–89 % over all six models) after referencing the features to m3.

---

## I. Recommended erratum and follow-up

1. Relabel Eq. (11) as energy per cycle, or report the power of 6 W. Rewrite Eq. (23) as the running integral Q(t).
2. Replace Eqs. (13)–(15) with u_g = v − Q/C_d and U_b = V_on − Q_on/C_d. Recompute E/N, T_e and n_e with N = 2.50×10¹⁹ cm⁻³.
3. Do not use T_e from the extrapolated region. Use a BOLSIG+/LXCat table instead.
4. Add statistical tests and effect sizes, and fix the claimed increases and decreases in the abstract and conclusion.
5. Split by device or by session, keep the scaler inside the Pipeline, and report confusion matrices with confidence intervals.
6. Present Sec. 5.7 as a hypothesis, and test the probability trend over time on its own.
7. Improve the measurement:
   - sample at ≥ 1 GS/s;
   - use a high-bandwidth current measurement (shunt or Rogowski);
   - keep V/div fixed;
   - measure Q–V with a monitor capacitor;
   - run a control experiment with fixed supply input.
8. Test more devices and use per-device baseline normalisation.
9. Run a control device in parallel and measure it in the same sessions (as m3 was). Report the false-positive rate on it, fix the oscilloscope settings, and log the supply input.

---

## J. Control device m3 and the complete raw-data archive

The full archive of the July 2020 experiment contains a third device, m3. It ran in parallel with m1 and m2 for 60 h, was measured in every session 10–33 min after m1, and shows no erosion in the micrographs. The thesis did not use it. With m3 as a control, the following holds. Details, tables and limitations are in **[raw-data-verification.md](raw-data-verification.md)**; `examples/verify_raw_archive.py` reproduces the numbers.

| # | Finding | Severity |
|---|---|---|
| J1 | The archive is the thesis dataset: section H is reproduced to the last digit, and the 2020 MATLAB summaries agree (23/26 sessions within 1 % for V_rms). The 2020 summaries label the energy "Energy per period", confirming C1. | Confirmed |
| J2 | The m3 V_rms drops with m1 and m2 (15 h: 1273 / 1312 / 1369 V after 1501 / 1496 / 1502 V at 10 h). Within the 15 h session the voltage of m2 and m3 jumps between ≈1150/1270, 1360 and 1460 V from capture to capture. Against m3, the damage-specific V_rms change is ≈ −48 V for both damaged devices. | Major |
| J3 | Classifiers trained on m2 call 95–100 % of m3's captures at 15, 25 and 40 h, and 100 % at 45–60 h, "degraded" (mean over 15–40 h: 57–86 %). | Major |
| J4 | Number of discharges (+2.7), U_b (−38 V) and V_min (−98 V) also change in m3. They are not established as damage indicators. | Major |
| J5 | Within a session: I_rms +3.5 mA per doubling of A/div; 20 ns sampling gives 6.6 fewer pulses, +3.0 pF C_d and +35 V U_b. The m2 degraded group has more 0.04 A/div captures (42 % vs 20 %). | Major |
| J6 | In the supply-voltage sweep at 20 h, the thesis V_b follows V_max with slope 0.87–1.08, while U_b moves with slope 0.23–0.33. Experimental confirmation of C5. | Confirmed |
| J7 | The m1 micrographs filed as "5h" (taken before the 10 h measurement) show a large melted blister at an electrode end, so the m1 "normal" sessions may contain a damaged cell. The m2 labels match the micrographs. | Moderate |
| J8 | Table 3 = atomic %; "Normal" is the Ref sample's `0-X10000-2` map and "Degradation" the m2 site-2-1 map `2-1-1X10000-1`. Over all maps, damaged sites give Ag 1.3–13.9 at.% and intact dielectric 0.0–0.7 at.%. | Minor |
| J9 | Fig. 7 (100 → 250 mA): at the same current scale, m2 gives 251 → 272 mA (p = 0.25) and m1 252 → 213 mA. | Moderate |
