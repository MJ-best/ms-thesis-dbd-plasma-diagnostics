# Thesis erratum

**English** | [한국어](erratum.ko.md)

**Thesis.** Minjun Kim, *Electrical measurements and prediction of dielectric barrier surface damage in nonthermal atmospheric pressure dielectric barrier discharge plasma based on machine learning*, M.S. thesis, Kwangwoon University, 2023.

**Date of this erratum.** 2026-11-01, revised 2026-11-04.

**Basis.**
- Each item refers to an entry of the [verification report](thesis-review.md), given by its ID (A1, C5, …).
- "Re-analysis" values come from device m2 (40 normal and 120 degraded captures).
- Items 59–65, and the revised items 2, 4 and 47, are based on the complete raw-data archive with the undamaged device M3 as a control ([raw-data verification](raw-data-verification.md); review section J).

**Page numbers.** These are the numbers printed on the thesis pages. The thesis is written in Korean, so the original wording is quoted in Korean only in the [Korean version](erratum.ko.md).

**Types of item.**

| Type | Meaning |
|---|---|
| **Correction** | Error of fact, arithmetic or definition |
| **Addition** | Missing interpretation or limitation |
| **Notation** | Units, typos, formatting |

> The thesis PDF itself is unchanged. Read it together with this erratum and the verification report.

## Abstract (pp. v–vi)

| # | Location | Correction | Type · ref |
|---|---|---|---|
| 1, 3 | KO and EN abstract | Replace "discharge energy per second" with "**discharge energy per cycle**". | Correction · C1 |
| 2, 4 | KO and EN abstract, sentence on changes due to damage | Replace with: "Surface damage lowered the effective voltage. Compared with an undamaged device measured at the same time points, the damage-specific decrease was about 3 % (≈ 48 V); about half of the raw decrease was a supply variation common to all devices. The capacitances and the discharge charge tended to increase, but not significantly against the control. The effective current, the number of discharges and the electron density depended on the oscilloscope settings and showed no significant damage-specific change." | Correction · D1, J |

## Chapter 1

| # | Page | Correction | Type · ref |
|---|---|---|---|
| 5 | 1 | Write "electron temperature of a few eV (≈ 10⁴–10⁵ K), gas temperature ≈ 300 K" instead of "> 2000 °K / 300 °K". | Notation · A5 |
| 6 | 1 | Add: "electron density in atmospheric DBD micro-discharge channels is 10¹⁴–10¹⁵ cm⁻³ [61]". The quoted 10¹⁰–10¹³ cm⁻³ applies to low-pressure plasmas. | Addition · A5 |
| 7 | 3 | Eq. (7) is **Townsend's semi-empirical formula**, not the Saha ionisation formula. | Correction · A1 |
| 8 | 5 | Below the Paschen curve there is no breakdown. To the right of the minimum, collisions are too frequent for electrons to gain the ionisation energy; to the left, there are too few collisions. | Correction · A4 |
| 9 | 6 | For a 100 µm air gap, Eq. (8) gives a breakdown voltage of ≈ 0.7–0.9 kV, not "several kV". The applied voltage must reach ≈ 2 kV peak because of the capacitive divider formed by the dielectric. | Correction · A3 |

## Chapter 2

| # | Page | Correction | Type · ref |
|---|---|---|---|
| 10–11 | 11 | The HV probe is the **P6015A**; the current probe is the **P6021A**. | Correction · B1 |
| 12 | 11–12 | Add the measurement limits: sampling 10 ns (some captures 20 ns), 5000 points (≈ 2.3 periods), 8 bit (40 V/LSB; 1.6–8 mA/LSB), current-probe bandwidth 60 MHz. As a result, pulse counts and peak currents are lower bounds. | Addition · B2 |
| 59 | 12 | Add to Sec. 2.1.3: a third device (M3) was operated in parallel for 60 h and measured at the same time points, and was not damaged. It serves as a control for supply and environmental variations. | Addition · J |
| 60 | 11–12 | Add: the current scale (0.04–0.4 A/div) and the sampling interval (10 / 20 ns) changed between and within time points. Within a time point, I_rms rises by 3.5 mA per doubling of A/div, and 20 ns sampling gives 6.6 fewer pulses. Compare I_rms, peak current, number of discharges and n_e only at fixed settings. | Addition · J |
| 61 | 13 | Fig. 7, "the discharge current rose from about 100 mA to 250 mA": state that Fig. 7 is a single example. At the same current scale (0.1 A/div) the peak currents are 251 → 272 mA (M2, not significant) and 252 → 213 mA (M1). | Correction · J |

## Chapter 3

| # | Page | Correction | Type · ref |
|---|---|---|---|
| 13 | 16 | V_min is ≈ 300–700 V (mean 504.5 ± 168.8 V), not "300–500 V". | Notation · C3 |
| 14, 16 | 16–17 | "Energy per second" → "**energy per cycle**": 0.13 ± 0.022 mJ (normal) and 0.13 ± 0.024 mJ (degraded). Add the mean power P = E·f ≈ **6.0 W**. | Correction · C1 |
| 62 | 16 | V_rms "1.5 kV → 1.4 kV": add that the undamaged device M3 dropped at the same time points (e.g. 1502 → 1369 V from 10 to 15 h). The damage-specific decrease is ≈ 48 V. | Addition · J |
| 15, 34 | 16, 32–33 | Write mean ± SD in a single unit: 1.50 ± 0.017 kV, 20.8 ± 4.0 mA, 34.2 ± 4.9 pF, 11.3 ± 1.2 pF, 17.3 ± 2.8 pF, 59.4 ± 9.0 nC, etc. | Notation · C10 |
| 17 | 17 | Eq. (11) → **E = ∫₀ᵀ v i dt** [J/cycle] and P = E/T [W]. | Correction · C1 |
| 18–20 | 19–21 | Replace Eqs. (13)–(15) with **text A** below. 55.6 nC / 30.5 pF = 1.82 kV, not 1.5 kV. The original Eq. (15) evaluates to V_max (2.1/2.0 kV). The burning voltage is U_b = 860 ± 68 V (normal) and 811 ± 75 V (degraded). | Correction · C4, C5 |
| 21 | 22 | Describe the pulse detector exactly: threshold = median prominence × 1.5, and pulses with IQR-outlier widths removed. Note that this threshold is relative to each capture. | Addition · C11 |
| 22 | 23 | Fig. 14 shows T₃–T₄, not T₁–T₂. | Notation · C3 |
| 23, 25 | 24–25 | In Eqs. (18) and (20), use **U_b** instead of V_breakdown. n_e then becomes ≈ 3.9×10¹² (normal) and 4.0×10¹² cm⁻³ (degraded), with no significant difference. Subtracting the displacement current from i_plasma is recommended. | Correction · C7 |
| 24 | 24 | The area 4.5×10⁻⁴ cm² is the pad top area, used as an approximation of the current cross-section. n_e is therefore a cell-averaged relative index. | Addition · C7 |
| 26 | 26 | Use N = p/(k_BT) = 2.50×10¹⁹ cm⁻³ (1 atm, 20 °C). This is the gas number density, not "Avogadro's number". | Correction · C6 |
| 27 | 26 | E/N ≈ 34×10⁻¹⁶ V cm² (≈ 340 Td) with U_b. Note that even the original inputs give 75.0, not 81.38. | Correction · C6 |
| 28 | 27 | At E/N ≈ 340 Td, just beyond the end of the source curve (Raizer, ≈ 300 Td), D/μ ≈ 6 V, so T_e ≈ **6 eV**. Re-analysis: 6.2 ± 0.3 eV (normal), 6.0 ± 0.4 eV (degraded). The original 13 eV is an OriginLab extrapolation to 814 Td of data digitised only up to 175 Td; depending on the fitted function, the value at 814 Td ranges from about 5 to 20 V. | Correction · C6 |
| 29 | 26 | Add: D/μ is the characteristic energy; it equals k_BT_e/e only for a Maxwellian distribution. | Addition · C6 |
| 30 | 28 | Eq. (23) → **Q(t) = ∫₀ᵗ i dt′ + Q₀**, after removing the current offset. | Correction · C2 |
| 31–33 | 32 | Table 2: the C_total fitting error is 7.63×10⁻¹⁴ F. Write p < 10⁻¹⁵ instead of 1.11×10⁻¹⁶. Add R² ≈ 0.98 and note the autocorrelation of the fitted points. | Correction · C8 |
| 35 | 32 | Add: the discharge-phase slope is an effective ζ_diel ≤ C_d when the discharge covers only part of the dielectric [50]. | Addition · C9 |

**Text A** (replaces Sec. 3.1.3, Eqs. 13–15)

- The gas-gap voltage is u_g(t) = v(t) − Q(t)/C_dielectric (13′). During discharge it stays approximately at the burning voltage U_b [42, 55].
- At ignition, the wall charge Q_on left over from the previous half cycle is still on the dielectric. It produces V_wall = −Q_on/C_dielectric (14′).
- Hence U_b = V_min − Q_on/C_dielectric = (1 − C_total/C_dielectric)·(V_max + V_min)/2 (15′).
- With the mean values of Table 2, U_b ≈ 0.92 kV, consistent with the Paschen value for a 100 µm air gap (0.72–0.87 kV). Re-analysis gives 860 ± 68 V (normal) and 811 ± 75 V (degraded).
- Why the original formula fails: using the transferred charge Q_max = C_dielectric (V_max − V_min) as the wall charge gives V_min + Q_max/C_dielectric = V_max.

## Chapters 4–5

| # | Page | Correction | Type · ref |
|---|---|---|---|
| 36 | 40 | "Degradaion" → "Degradation". | Notation |
| 63 | 36, 46 | Add to Sec. 4.2 and Fig. 27 / Table 5: M1 had a cell defective from fabrication. It shows as a melted blister at an electrode end in the micrographs taken after 5 h (before the 10 h measurement). It is not discharge damage, so the normal labels of M1 at 5 h and 10 h stand. | Addition · J |
| 64 | 38–40 | Figs. 23–24 "(a) normal discharge electrode" → "(a) new source, not discharged (Ref)". Table 3: "[%]" → "**[at.%]**"; add that each column is a single 10,000× map, and give the range over all maps (damaged sites 1.3–13.9 at.% Ag, intact dielectric 0.0–0.7 at.%). | Correction · J |
| 37 | 43 | "Freqency" → "Frequency"; the unit of the electron-density SD is cm⁻³. | Notation |
| 38 | 43 | Add: the features are not independent. C_gas follows from C_d and C_total; T_e and n_e follow from the breakdown voltage. | Addition · D9 |
| 39 | 44–45 | Remove the references to a DNN (no results are reported). | Notation · D9 |
| 40 | 45 | Add: the training/test set is a random split of time-ordered captures from one device, so test scores are optimistic. The cross-device validation set gives the primary performance. | Addition · D5 |
| 41 | 47 | "min_samples_splite" → "min_samples_split". | Notation |
| 42–44 | 49–50 | Precision = fraction of samples predicted *degraded* that are degraded. Recall = fraction of truly degraded samples that are predicted degraded. TP, TN, FP, FN refer to the classes degraded (positive) and normal (negative); **FN = degraded predicted as normal**. | Correction · D3 |
| 45 | 51 | Table 6: precision and recall are **macro averages**; F1 is the **binary F1** of the degraded class. | Correction · D2 |
| 46 | 51 | Add: on the validation set, LR and SVM reach AUC 0.99 but only 76.4 % accuracy. This is a threshold shift between devices, which per-device baseline normalisation removes. | Addition · D4 |
| 47 | 52 | Add: V_rms alone almost separates the classes (AUC ≈ 1.0); the classifier mainly detects the drop of the applied voltage. About half of that drop also occurs in the undamaged device M3 measured at the same time points. | Addition · D6, J |
| 65 | 51–52 | Add: classifiers trained on M2 call 95–100 % of the undamaged M3's captures "degraded" at time points with a low supply voltage. Report the false-positive rate on a control device together with the scores of Table 6. | Addition · J |
| 48 | 53 | A random forest **averages** the votes of all 1000 trees; it does not select the best tree. Fig. 29 shows one tree. | Correction · D7 |
| 49 | 55 | Tables 7–8 show no drop of the "normal" probability before damage (0.886 / 0.909; 0.816 / 0.818). The "≥ 10 % drop" precursor criterion is therefore not supported and should be stated as future work. | Correction · D8 |
| 50 | 56 | "Fig. 32 (a)/(b)" → "Fig. 30 (a)/(b)". | Notation |

## Chapter 6 (p. 57)

| # | Correction | Type · ref |
|---|---|---|
| 51–52 | Same changes as items 1–2 (energy per cycle; the list of significant and non-significant changes). | Correction · D1, J |

## References

| # | Correction |
|---|---|
| 53 | [25] duplicates [10] (Michel et al., J. Appl. Phys. 105, 073302 (2009)); merge them. |
| 54–55 | In [26] and [27], remove the affiliation letters from the author names: Gimeno, Hanson, Goldhawk, Simpson, Lavoie. |
| 56 | [28]: add the journal, *J. Appl. Phys.* 124, 183301 (2018). |
| 57 | [13]: add the title, journal, volume, pages and year. |
| 58 | [1]: check the edition (Plenum 2nd ed. 1984, or Springer 3rd ed. 2016). |
