# Verification against the complete raw-data archive (M1, M2, M3)

**English** | [한국어](raw-data-verification.ko.md)

**Subject.** Minjun Kim, *Electrical measurements and prediction of dielectric barrier surface damage in nonthermal atmospheric pressure dielectric barrier discharge plasma based on machine learning*, M.S. thesis, Kwangwoon University, 2023.

**Data.** The complete archive of the July 2020 experiment (`M!M2M3실험데이터종합`, about 5.3 GB). It is not included in this repository.

| Folder | Content |
|---|---|
| `전류전압 측정/1·2·3번소스` | Endurance sessions of M1 (0–25 h), M2 (0–40 h) and M3 (0–60 h). Each session has 20 detail captures (10 at 10 ns and 10 at 20 ns sampling) and one overview capture. |
| `전류전압 측정/4번소스` | 120-cell discharge, single-cell discharge, discharge with a clip contact (a separate device) |
| `전압에 따른 QV 변화` | Supply-input voltage sweep at 20 h (8.5 / 9 / 10 / 14 V) |
| `현미경사진`, `표면사진` | Optical micrographs and whole-device photographs taken at every session |
| `SEM, EDS 데이터` | SEM metadata and Bruker EDS reports (images) for M1, M2, M3 and Ref |
| `*.xlsx` | Per-session summaries computed in 2020 with the MATLAB code |

**Methods.**
1. Re-analysis with this package (`dbd_diagnostics`) of 600 endurance detail captures (500 from 5 h onward are analysed) and 70 sweep captures.
2. Timestamps and dark-burn areas of all 242 micrographs.
3. Every composition table in the EDS reports, compared with Table 3.
4. **M3**, which the thesis did not use, serves as an **undamaged control device**. In every session it was measured on the same day as M1 and M2, 10–33 min after M1.

Reproduce with `python examples/verify_raw_archive.py "<archive>/전류전압 측정" "<archive>/전압에 따른 QV 변화/I 기준으로 계산"`, which prints the numbers of sections 1–6. Values read from photographs and EDS reports are recorded in the tables of this document.

Severity levels follow the [verification report](thesis-review.md): **[Major]** affects a conclusion or a headline number; **[Moderate]** the method or wording must change; **[Minor]** notation or wording.

---

## 0. Summary

| # | Finding | Verdict |
|---|---|---|
| 1 | The archive is the thesis dataset. The re-analysis numbers of section H of the verification report are reproduced to the last digit, and the 2020 MATLAB summaries agree (23 of 26 sessions within 1 % for V_rms and 5 % for energy). The 2020 summaries label the energy **"Energy per period"**, which confirms from the original records that "discharge energy per second" is a labelling error (review C1). | Confirmed |
| 2 | The 15 h label boundary matches the M2 micrographs. But the M1 "5h" micrographs (taken 2020-07-20 10:30, before the 10 h measurement) show a **large melted blister at an electrode end**. The "normal (0)" part of the validation set may already contain a damaged cell. | Moderate |
| 3 | ★ The V_rms drop of the 15 h session (evening of 20 July) appears **equally in the undamaged M3** (−133 V against 10 h). Within that session V_rms jumps between ≈1150/1270, 1360 and 1460 V from one capture to the next, seconds apart. This is **switching of the supply's operating mode**, not gradual damage. A difference-in-differences against M3 gives a damage-specific V_rms change of **≈ −48 V** for both M1 and M2, about half of the naive change (−107 / −92 V). | Major |
| 4 | ★ The thesis classifiers trained on M2 call **95–100 %** of the captures of the undamaged M3 at 15, 25 and 40 h, and **100 %** at 45–60 h, "degraded" (mean false-positive rate over 15–40 h: 57–86 %). The classifiers respond to the **applied-voltage level** more than to surface damage. After dividing each feature by the same-session M3 value, cross-device accuracy falls from 94–99 % to **65–78 % (DT/RF/GB; 64–89 % over all six models)**. | Major |
| 5 | Within one session and one device, the oscilloscope settings change the values. Doubling the current A/div raises I_rms by **+3.5 mA** (p = 10⁻¹⁷). Going from 10 to 20 ns sampling raises I_rms by **+4.3 mA**, lowers the pulse count by **6.6**, and raises C_d by **3.0 pF** and U_b by **35 V**. The M2 degraded group has more 0.04 A/div captures (42 % vs 20 %). The claims "I_rms decreases, number of discharges and n_e increase" are confounded with the settings. | Major |
| 6 | In the supply-voltage sweep, the thesis V_b (Eq. 15) follows V_max with slope **0.87–1.08**, while the burning voltage U_b moves with slope 0.23–0.33 only. This is an independent **experimental confirmation** of review C5 (V_b ≡ V_max). | Confirmed |
| 7 | Table 3 (EDS) is traced. The "Normal" column is the `0-X10000-2` map of the **Ref sample**; the "Degradation" column is the `2-1-1X10000-1` map of the **M2 damaged site**. Both columns are **atomic %**. The 12 damaged-site maps give Ag 1.3–13.9 at.%; the 41 intact-dielectric maps give 0.0–0.7 at.% (one map 1.44). The conclusion is robust; the unit and the choice of map should be stated. | Minor |
| 8 | Fig. 7, "discharge current from ~100 mA to ~250 mA", is not reproduced when captures at the same current scale (0.1 A/div) are compared: M2 251 → 272 mA (p = 0.25), M1 252 → 213 mA (a decrease). | Moderate |
| 9 | Some statements of the earlier verification report must be revised. The U_b decrease (p = 2×10⁻⁴) and the increase in the number of discharges (p = 6×10⁻⁴) also occur in M3, in the same direction (−38 V, p = 0.05; +2.7, p = 4×10⁻⁶), so they are not supported as damage indicators. See section 9. | Report revision |

**What holds.**
- The surface damage itself is real. The burn marks in the M1 and M2 micrographs and the Ag increase in EDS agree.
- Both damaged devices lose about 48 V (≈ 3 %) more V_rms than the control. With V_rms referenced to M3, a single feature still gives AUC 0.99 (M1) and 0.83 (M2). A damage-specific signal exists.
- Its size is, however, much smaller than the session-to-session supply variation (M3 alone ranges from 1155 to 1502 V between sessions). Interpreting V_rms changes measured without a control device as damage produces many false alarms.

---

## 1. Archive and reproducibility

### 1.1 Session layout

- Each session has 20 detail captures. `ALL0001–0010` are sampled every 10 ns (about 2 periods), `ALL0011–0020` every 20 ns (about 4 periods). `ALL0021` is an overview capture at 0.2–0.4 ms per sample.
- The voltage channel is at 1000 V/div in every endurance session. Only M3 45 and 50 h used 500 V/div; there a few peak samples (0.3 %) touch the ADC rail (1940 V), but a sine fit without those samples gives the same V_rms (1122 V vs 1126 V). The voltage drop of those two sessions is real.
- The current channel changes between 0.04, 0.1, 0.2 and 0.4 A/div from session to session, and sometimes between the two blocks of one session (section 5).
- There are two 0 h sessions (17 and 18 July), but the thesis starts at 5 h. The M3 0 h session of 18 July is corrupt (Δt = 0), as its folder name says.
- In every session the order is M1 → M2 → M3, and all three devices were measured within 10–33 min (14 min at 15 h).

### 1.2 Reproduction of the earlier re-analysis (M2, normal 5–10 h n = 40, degraded 15–40 h n = 120)

| Quantity | Verification report, section H | This archive |
|---|---|---|
| Energy per cycle | 0.131 / 0.132 mJ | 0.1308 / 0.1321 mJ |
| V_rms | 1489 / 1396 V | 1489 / 1396 V |
| C_dielectric | 34.5 / 37.2 pF | 34.45 / 37.18 pF |
| V_max | 2284 / 2155 V | 2284 / 2155 V |
| U_b | 860 / 811 V | 860.1 / 810.5 V |
| T_e (from U_b) | 6.2 / 6.0 eV | 6.22 / 5.97 eV |

All values agree, so the data used in the earlier review (remote repository) and this archive are the same data.

**Table 1.** The thesis gives 46.5 kHz, 1.5 kV and 19.8 mA at discharge start. The re-analysis gives 46.9 kHz, 1481 V, 20.3 mA (M2, 5 h) and 46.95 kHz, 1484 V, 20.1 mA (M1, 5 h), in agreement.

### 1.3 Comparison with the 2020 MATLAB summaries

The 26 per-session `*종합데이터.xlsx` workbooks (each the mean of 10 detail captures) were compared with the re-analysis of the same captures.

| | Median difference | Sessions in agreement |
|---|---|---|
| V_rms | +0.11 % | 23/26 within 1 % |
| Energy per cycle | +0.25 % | 23/26 within 5 % |
| I_rms | +0.9 % | 21/26 within 4 % |

- The three mismatching sessions (M1 10 h, M2 15 h, one of two M3 60 h files) average a different block of captures. For example, the M2 15 h workbook value of 1375.6 V matches the `ALL0011–0020` block (1378.6 V).
- The 2020 workbooks name the energy row `Energy per period`. The original computation thus confirms review C1: the "discharge energy per second, 0.13 mJ" of Eq. (11) is an energy per cycle.
- `20200724_M4_40h 종합데이터.xlsx` in the M2 folder is a file-name typo (M2, 40 h).

---

## 2. Labels: micrographs and photographs

### 2.1 Timeline

File modification times (when the camera or microscope PC saved the file) and the measurement times in the CSV headers:

| Date | Electrical measurement | Micrographs |
|---|---|---|
| 18 July | 5 h: M1 14:22, M2 14:31, M3 14:40 | – |
| 20 July | – | "5h": M1 10:30, M2 10:37, M3 10:53 |
| 20 July | 10 h: M1 15:11, M2 15:20, M3 15:26 | 10 h: M1 16:45, M2 16:50, M3 16:58 |
| 20 July | 15 h: M1 19:58, M2 20:04, M3 20:10 | 15 h: M1 21:13, M2 21:17, M3 21:19 |
| 21 July | 20 h: M1 14:48, M2 15:07, M3 15:18 / 25 h: M1 20:26, M2 20:36, M3 20:42 | 20 h 16:40–16:55, 25 h 21:44–21:52 |

The "5h" micrographs were taken on the morning of 20 July, before the 5 → 10 h run.

### 2.2 Quantified burn marks

For every micrograph (640×480), the largest connected region with luminance < 90 was found; a region ≥ 150 px counts as a burn mark.

| Hours | M1 | M2 | M3 |
|---|---|---|---|
| 5 | **4/9** (max 1521 px) | 0/5 | 0/8 |
| 10 | 0/5 | 0/5 | 0/4 |
| 15 | 2/5 | 2/5 | 0/4 |
| 20 | 3/6 | 2/5 | 0/6 |
| 25 | 3/9 | 2/6 | 0/4 |
| 30 | – | 3/9 | 0/11 |
| 35 | – | 2/10 | 0/15 |
| 40 | – | 2/13 | 0/16 |
| 45–60 | – | – | 1/82 (one 159 px speck at 55 h) |

- **M2** [OK]: no burn marks at 5 and 10 h; dark erosion marks in every session from 15 h on. This matches the thesis label boundary (≤ 10 h normal, ≥ 15 h degraded).
- **M1** [Moderate]: dark erosion marks also start at 15 h. But `Snapshot000008` and `Snapshot000010` of the "5h" set show a **large melted blister** covering an electrode end. A similar melted region appears in 20 h `Snapshot000004`, and at 25 h the experimenter named a photograph `오른쪽 끝 결국 터짐` ("right end finally burst"), which suggests the same site was being tracked. Since these photographs predate the 10 h measurement, the M1 5 h and 10 h sessions that the thesis labels normal (0) — the 80 normal rows of the validation set — may already contain a damaged cell. Fig. 27 and Sec. 5.3 should say so, or the labelling rule should distinguish damage types (erosion vs melting/blistering).
- **M3**: no burn marks up to 50 h. This matches Sec. 4.2 of the thesis ("surface damage was observed in two samples"). M3 is the healthy comparison device, discharged alongside M1 and M2 under the same conditions; the experimenter confirmed that it was not damaged.

---

## 3. ★ M3 as a control: common-mode session effects [Major]

### 3.1 Session means of V_rms [V]

| Hours | M1 | M2 | M3 |
|---|---|---|---|
| 5 | 1484 | 1481 | 1484 |
| 10 | 1501 | 1496 | 1502 |
| **15** | **1273** | **1312** | **1369** |
| 20 | 1460 | 1432 | 1478 |
| 25 | 1423 | 1355 | 1454 |
| 30 | – | 1436 | 1475 |
| 35 | – | 1399 | 1473 |
| 40 | – | 1445 | 1443 |
| 45 / 50 / 55 / 60 | – | – | 1155 / 1156 / 1454 / 1176 |

- Before damage (5 and 10 h) the three devices agree to within a few volts.
- At 15 h all three drop **together**; the undamaged M3 drops by 133 V.
- M3 falls to 1155 V at 45 and 50 h and recovers to 1454 V at 55 h, although it is not damaged.

### 3.2 Voltage jumps within a session

V_rms [V] of the 15 h session in capture order (20 captures in about 4 min):

```
M2 15h  1457 1456 1149 1135 1459 1166 1147 1168 1153 1160 1359 1358 1469 1351 1358 1481 1364 1356 1340 1349
M3 15h  1464 1447 1264 1282 1453 1282 1294 1471 1481 1468 1464 1464 1461 1272 1281 1445 1276 1274 1274 1271
M1 15h  1274 1289 1273 1270 1277 1272 1274 1264 1272 1270 1275 1273 1280 1267 1270 1271 1273 1278 1272 1266
```

- The applied voltage jumps between discrete levels within seconds. This looks like mode switching of the resonant supply; gradual surface damage cannot explain it.
- The undamaged M3 jumps in the same way at the same time. On the evening of 20 July the supply (or its environment) varied in a way common to all three devices.

### 3.3 Difference-in-differences against M3

The change from "before" (5–10 h) to "after" (15 h to the last session), minus the M3 change over the same sessions. p (M3) is a Welch test of the M3 change itself; p (DiD) tests the session-mean differences (2 sessions before, 3–6 after).

| Device | Feature | Naive change | M3 change | p (M3) | DiD | p (DiD) |
|---|---|---|---|---|---|---|
| M1 | V_rms | −107 V | −59 V | 7×10⁻⁷ | **−48 V** | 0.18 |
| M2 | V_rms | −92 V | −44 V | 2×10⁻⁷ | **−48 V** | 0.018 |
| M1 | V_max | −156 V | −103 V | 2×10⁻⁶ | −54 V | 0.34 |
| M2 | V_max | −129 V | −65 V | 7×10⁻⁵ | −63 V | 0.16 |
| M1 | U_b | −148 V | −48 V | 0.024 | −100 V | 0.14 |
| M2 | U_b | −50 V | −38 V | 0.049 | −12 V | 0.80 |
| M1 | C_d | +1.5 pF | +1.3 pF | 0.43 | +0.2 pF | 0.98 |
| M2 | C_d | +2.7 pF | +0.1 pF | 0.93 | +2.6 pF | 0.56 |
| M2 | C_total | +0.66 pF | +0.27 pF | 0.45 | +0.40 pF | 0.50 |
| M2 | Q_max | +3.2 nC | −0.4 nC | 0.85 | +3.6 nC | 0.36 |
| M1 | Number of discharges | +0.5 | **+2.6** | 4×10⁻⁴ | −2.1 | 0.43 |
| M2 | Number of discharges | +3.7 | **+2.7** | 4×10⁻⁶ | +1.0 | 0.59 |

- **V_rms**: both damaged devices drop about 48 V more than the control. The same sign and size in both devices make a damage-specific effect plausible, but it is about half of the naive comparison.
- **Number of discharges** and **U_b** change significantly, in the same direction, in the undamaged M3. There is no evidence that they indicate damage.
- The increases of C_d, C_total and Q_max point the right way relative to the control but are not significant at the session level (low power).

---

## 4. ★ The classifiers call the control device "degraded" [Major]

### 4.1 False positives on M3

Share [%] of M3 captures classified "degraded" by models trained on M2 (5–10 h = 0, 15–40 h = 1) with the thesis' 14 features and hyper-parameters:

| Model | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LR | 5 | 25 | 100 | 80 | 100 | 80 | 55 | 100 | 100 | 100 | 100 | 100 |
| DT | 5 | 20 | 95 | 20 | 100 | 25 | 35 | 95 | 100 | 100 | 95 | 100 |
| **RF** | 5 | 20 | **95** | 20 | **100** | 30 | 35 | **95** | 100 | 100 | 100 | 100 |
| GB | 35 | 25 | 95 | 25 | 100 | 25 | 40 | 100 | 100 | 100 | 100 | 100 |
| SVM | 5 | 20 | 100 | 40 | 100 | 60 | 70 | 100 | 100 | 100 | 100 | 100 |
| kNN | 10 | 80 | 95 | 30 | 50 | 40 | 30 | 95 | 100 | 100 | 45 | 100 |

- Of the 15–40 h captures of M3, which show no burn marks, 57–86 % (RF 62.5 %) are classified "degraded". The false positives concentrate in sessions with low V_rms (15, 25, 40, 45–60 h).
- Even in the undamaged M3, V_rms alone separates the "≤ 10 h" and "≥ 15 h" sessions with AUC 0.90. Much of the classifiers' performance (V_rms alone: AUC 0.996) comes from this common-mode session effect.
- Training on M1 and M2 together gives nearly the same result.

### 4.2 Referencing to the control device

Each feature was divided by the same-session M3 mean, which roughly removes the common-mode session effect, and the cross-device accuracy was recomputed:

| Setting | DT | RF | GB | LR | SVM | kNN |
|---|---|---|---|---|---|---|
| Raw features, M2 → M1 | 99.0 % | 99.0 % | 94.0 % | 95.0 % | 92.0 % | 74.0 % |
| Referenced to M3, M2 → M1 | 65.0 % | 75.0 % | 65.0 % | 89.0 % | 84.0 % | 85.0 % |
| Raw features, M1 → M2 | 98.8 % | 96.2 % | 98.8 % | 79.4 % | 86.9 % | 66.9 % |
| Referenced to M3, M1 → M2 | 78.1 % | 68.8 % | 78.1 % | 71.9 % | 71.9 % | 63.8 % |

- A signal survives the referencing: referenced V_rms alone gives AUC 0.99 (M1) and 0.83 (M2).
- The M3 session means themselves contain mode jumps (section 3.2), so the referencing is a rough approximation. It does not change the conclusion that much of the "damage signal" was a session effect.

---

## 5. Dependence on measurement settings [Major for I_rms, number of discharges, n_e]

### 5.1 Comparisons within a session

Effect of the oscilloscope settings, from a regression with session fixed effects (500 captures):

| Feature | Per doubling of A/div | p | 20 ns vs 10 ns | p |
|---|---|---|---|---|
| I_rms | **+3.5 mA** | 10⁻¹⁷ | **+4.3 mA** | 5×10⁻⁵³ |
| Peak current | +59 mA | 4×10⁻¹⁰ | +37 mA | 9×10⁻¹⁰ |
| Number of discharges | +1.1 | 0.11 | **−6.6** | 3×10⁻⁴⁵ |
| n_e (thesis method) | +2.7×10¹² cm⁻³ | 2×10⁻⁸ | +5×10¹¹ | 0.07 |
| C_d | +0.3 pF | 0.54 | **+3.0 pF** | 2×10⁻²⁰ |
| U_b | +45 V | 3×10⁻⁹ | +35 V | 3×10⁻¹³ |
| V_rms | +19 V | 5×10⁻⁴ | −5 V | 0.13 |

Examples: in the M2 10 h session, I_rms is 16.1 mA at 0.04 A/div and 22.9 mA at 0.1 A/div. At M2 40 h it is 24.3 mA at 0.1 A/div and 36.4 mA at 0.2 A/div.

### 5.2 Causes

- The current channel does not clip: 0–1 % of captures repeat their extreme value three or more times.
- The noise floor scales with A/div: 2.4 / 5.9 / 11.9 mA at 0.04 / 0.1 / 0.2 A/div (1 LSB = 1.6 / 4 / 8 mA).
- Pulse detection depends directly on the time resolution (review B2).

### 5.3 Consequences for the thesis

Settings in the two M2 groups:

| Group | 0.04 A/div | 0.1 A/div | 0.2 A/div |
|---|---|---|---|
| Normal (5–10 h) | 8 | 32 | 0 |
| Degraded (15–40 h) | 50 | 60 | 10 |

- The degraded group has more low-sensitivity captures (0.04 A/div). This composition partly explains the "I_rms decrease" (20.8 → 18.6 mA).
- The number of discharges and the electron density change with the settings. Comparisons that do not fix the settings or include them as covariates cannot be read as damage effects.
- Because the settings differ between sessions, these features also act as **session identifiers** (quantitative confirmation of review D5-3).

---

## 6. Supply-voltage sweep: experimental confirmation of V_b ≡ V_max

Measured at 20 h with different supply-input voltages (8.5 V = the 20 h endurance session):

| Device | Supply [V] | V_rms | V_max | Thesis V_b (Eq. 15) | U_b | Thesis T_e [eV] |
|---|---|---|---|---|---|---|
| M1 | 8.5 / 9 / 10 / 14 | 1460 → 1654 | 2259 → 2681 | 2193 → 2642 | 857 → 956 | 12.1 → 14.0 |
| M2 | 8.5 / 10 / 14 | 1432 → 1562 | 2222 → 2530 | 2175 → 2447 | 849 → 966 | 12.0 → 13.1 |
| M3 | 8.5 / 10 / 14 | 1478 → 1577 | 2250 → 2525 | 2171 → 2428 | 870 → 935 | 12.0 → 13.1 |

| Device | dV_b/dV_max | r | dU_b/dV_max |
|---|---|---|---|
| M1 | 1.08 | 0.93 | 0.23 |
| M2 | 0.87 | 0.86 | 0.33 |
| M3 | 0.95 | 0.86 | 0.29 |

- The thesis "discharge voltage" V_b follows the applied peak voltage almost 1 : 1 (mean V_b − V_max = −57 to −94 V).
- The gas burning voltage U_b changes by only about a third as much.
- The thesis T_e is a function of V_b, so it moves with V_max.
- This confirms review C5 (Eq. 15 is identically V_max) and C6 (T_e restates V_b) under a different experimental condition.

---

## 7. Source of Table 3 (EDS) [Minor]

### 7.1 Source

| Table 3 column | Original report | Sample | Matching column |
|---|---|---|---|
| Normal | `Ref SEM, EDS/Ref EDS/0-X10000-2` (2020-07-27) | Ref sample, 10,000× map | **Atom [%]**, all 10 elements (C 7.70, O 30.27, Na 9.19, Mg 2.48, Al 20.04, Si 13.60, Ca 2.41, Zn 7.09, Ag 0.57, Pb 6.65) |
| Degradation | `M2 SEM, EDS/20200727 M2 EDS/2-1/2-1-1X10000-1` | M2 damaged site, 10,000× map | **Atom [%]**, all 10 elements (C 7.22, O 27.69, Na 6.49, Mg 2.52, Al 15.54, Si 9.88, Ca 2.03, Zn 8.87, Ag 13.86, Pb 5.90) |

- Table 3 gives the composition only as "[%]". It should say **atomic % (at.%)**. The same Ag 13.86 at.% is 31.05 wt.%.
- Pt (coating) is set to zero.

### 7.2 Distribution over all maps (Ag, at.%)

| Site | Maps | Ag [at.%] |
|---|---|---|
| Ref sample | 6 | 0.00, 0.00, 0.02, 0.05, 0.38, **0.57** |
| M3 after 60 h (31 July) | 13 | 12 maps 0.00–0.25, one map 1.44 |
| M2 intact sites 2-2, 2-3, 2-4 (27 July) | 10 | 0.00–0.56 |
| M1 dielectric over, between and away from the electrodes (23 July) | 12 | 0.01–0.68 |
| M1 exposed electrode, burnt part 1-1 (23 July) | 4 | 5.08, 7.21, 8.70, 11.94 |
| M1 exposed electrode, burst part 1-2 (23 July) | 4 | 1.28, 1.43, 2.87, 3.22 |
| M2 damaged site 2-1 (27 July) | 4 | 3.78, 4.28, 11.59, **13.86** |

- Damaged sites (12 maps, 1.3–13.9 at.%) and intact dielectric (41 maps, mostly ≤ 0.7 at.%) separate clearly. The conclusion that the damaged dielectric exposes the electrode is robust.
- Each Table 3 value is the maximum among the maps of its site (Normal 0.57 is the highest Ref map; Degradation 13.86 the highest 2-1 map). Ranges, or mean ± SD, would be more representative.
- The "(a) normal discharge electrode" of Sec. 4.3 and Figs. 23–24 is in fact the **Ref sample**. The thesis should state whether Ref is a discharged or an unused electrode.

---

## 8. The example of Fig. 7 [Moderate]

The thesis states that after surface damage "the discharge current rose sharply from about 100 mA to about 250 mA" and "the discharge onset voltage fell from about 500 V to 400 V". Comparing captures at the same current scale (0.1 A/div):

| Device | Peak current, normal → degraded | p | V_min, normal → degraded | p |
|---|---|---|---|---|
| M2 | 251 ± 68 → 272 ± 97 mA | 0.25 | 675 → 607 V | 0.09 |
| M1 | 252 ± 62 → 213 ± 39 mA | 0.002 (decrease) | 821 → 556 V | 4×10⁻⁵ |

- The normal peak current is already about 250 mA. The 2.5-fold increase is not reproduced; Fig. 7 is an unrepresentative single example (0.04 A/div captures give low peaks of about 150 mA).
- The decrease of the onset voltage is significant for M1 only.

---

## 9. Revisions to the earlier verification report (thesis-review.md)

These revisions have been applied to the [verification report](thesis-review.md) (section J and the locations below), the README and the [erratum](erratum.md) (items 59–65; revised items 2, 4 and 47).

| Location | Earlier statement | Revision |
|---|---|---|
| C5, consequences | "The decrease of U_b with damage (p = 2×10⁻⁴) is itself a meaningful new indicator." | U_b also falls in M3, by 38 V (p = 0.05). The M2 DiD is −12 V (p = 0.80), so there is no evidence that U_b indicates damage. U_b also depends on the sampling setting (+35 V). |
| D1, H | "Number of discharges (MAD 5σ) up, p = 6×10⁻⁴" | It also rises in M3 (+2.7, p = 4×10⁻⁶) and changes by 6.6 with the sampling interval. It cannot be read as a damage effect. |
| D6, README | "V_rms alone gives AUC ≈ 1.0; damage → load change → sag of the supply output" | About half of the drop is a common-mode session effect that the undamaged M3 also shows. The "control experiment" requested in D6 already exists in the archive (M3). The damage-specific part is ≈ −48 V. |
| D5-3 | "The current-probe scale changed between sessions, so resolution and noise floor encode session identity" (hypothesis) | Confirmed quantitatively (section 5). |
| H, README | Cross-device accuracy DT/RF/GB 95–99 % | 65–78 % (DT/RF/GB; 64–89 % over all six models) after correcting for the common-mode session effect. False-positive rate on the undamaged M3: 57–86 % (15–40 h). |
| README summary | "The V_rms decrease and C_total increase are significant." | V_rms still decreases relative to the control (≈ −48 V). The C_total increase is not significant against the control at the session level. |

---

## 10. Recommendations

**Thesis and erratum.**
1. State that about half of the V_rms change is a supply variation common to all three devices, and report the DiD against M3.
2. Report the **false-positive rate on the undamaged M3** alongside the classification scores. Include M3 as an undamaged device in training and evaluation, or use features normalised by a same-session control device.
3. Compare I_rms, peak current, number of discharges and n_e only between captures with the same settings (e.g. 0.1 A/div, 10 ns), or include the settings as covariates.
4. Give Table 3 in at.%, with per-map ranges, and identify the Ref sample.
5. Review the melted blister in the M1 5 h micrographs, and redefine or state the M1 labels of the validation set.
6. Replace Fig. 7 with representative values (mean ± SD at one scale), or mark it as a single example.

**Future experiments.**
1. Fix the oscilloscope settings (V/div, A/div, sampling) for all sessions.
2. Log the supply input voltage and current and the room temperature and humidity.
3. Always run a control device in parallel and measure it at the same time (as M3 was). Randomise the measurement order.
4. Log V_rms continuously with timestamps, so that mode changes within a session become visible.

---

## 11. Limitations

- **One healthy comparison device.** M3 was discharged alongside M1 and M2 under the same conditions and was not damaged (confirmed by the experimenter; consistent with the micrographs and with 12 of 13 EDS maps ≤ 0.25 at.%). Every change of M3, including the V_rms drop at 45–60 h, is therefore a supply or environmental change. With only one healthy device, however, the spread between two healthy devices in the same session is unknown.
- **Power.** The DiD works at the session level (2 sessions before, 3–6 after), so its power is low. "Not significant" does not mean "no effect".
- **EDS values.** Read directly from the tables in the Bruker report images (no raw spectra are available). Some M1 and M2 maps of 31 July (1-2, 1-3, 1-4, 1-s, 2-s) have no site name and are left out of section 7.2; their Ag is 0.3–3.0 at.%.
- **Identity of the M1 melted site.** That the 5 h and 25 h photographs show the same site is inferred from their appearance and file names.
- **Photograph times.** File modification times are used as capture times.
