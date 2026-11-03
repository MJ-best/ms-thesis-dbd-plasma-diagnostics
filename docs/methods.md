# Methods: from oscilloscope waveforms to plasma parameters and damage prediction

**English** | [한국어](methods.ko.md)

This page documents every computation implemented in `src/dbd_diagnostics`. For each step it states the equation, the assumptions, and how the step relates to the thesis. Equation numbers "(Th. n)" refer to the thesis.

## 0. Device and measurement

- **Discharge source**
  - Micro-gap coplanar surface DBD in ambient air.
  - 126 pairs of 150 µm × 300 µm electrode pads facing each other across a 100 µm gap.
  - Ag electrodes about 10 µm thick, with a screen-printed dielectric about 40 µm thick.
  - Driven at about 46.5 kHz and about 1.5 kV_rms.
- **Instruments**
  - Tektronix P6015A (1000×) high-voltage probe for v(t).
  - Tektronix P6021A current probe for i(t).
  - GW Instek GDS-2102A oscilloscope.
  - Each capture holds 5000 points at 10 ns or 20 ns per sample, with 8-bit resolution.
- **Reader**: `io.load_csv` reads the GW Instek "Format 2.0A" CSV, or a plain `t,v,i` CSV.

## 1. Windowing and offsets (`waveform`, `electrical`)

- **Integer periods**
  - Periods are found between successive rising zero crossings of v(t), using a Schmitt trigger with ±10 % amplitude hysteresis.
  - Every quantity is computed for each full period and then averaged.
- **Offset removal**
  - The voltage mean is removed.
  - The current offset is removed with $i \leftarrow i - \frac{1}{T}\int_0^T i\,dt$.
  - This step is essential. An offset of 1 LSB (8 mA) integrates to about 170 nC per period, about 3× the discharge charge.

## 2. Cycle-averaged electrical quantities

| Quantity | Equation | Note |
|---|---|---|
| RMS | $V_{rms}=\sqrt{\frac1T\int_0^T v^2dt}$, $I_{rms}=\sqrt{\frac1T\int_0^T i^2dt}$ | Th. 9–10 |
| Energy per cycle | $E=\int_0^T v\,i\,dt$ [J] | The thesis' "0.13 mJ" is this quantity |
| Mean power | $P=E/T=E f$ [W] | Th. 11 (≈ 6 W for the thesis device) |
| Charge | $Q(t)=\int_0^t i\,dt' + Q_0$, centred so that $\max Q=-\min Q$ | Th. 23 (running integral) |

## 3. Lissajous figure and equivalent circuit (`lissajous`)

**Equivalent circuit (Manley 1943).**
- The dielectric capacitance $C_d$ is in series with the gas-gap capacitance $C_g$.
- While the gap conducts, the gap voltage is clamped at the burning voltage $\pm U_b$.
- $Q$ is the charge on $C_d$.
- The gap voltage is

$$u_g(t)=v(t)-\frac{Q(t)}{C_d}.$$

- When there is no discharge, $dQ/dV=C_{cell}=C_dC_g/(C_d+C_g)$ (the thesis' $C_{total}$, Th. 24).
- During discharge, $Q=C_d\,(V-U_b)$, so $dQ/dV=C_d$ (the thesis' $C_{dielectric}$).

**Fitting.**
- One period is split at the voltage extrema into a rising branch (−V_max → +V_max) and a falling branch.
- Each branch is fitted by two straight lines with an optimal breakpoint (least squares).
- The first slope is $C_{total}$ and the second slope is $C_{dielectric}$.
- The breakpoint is the discharge initiation point $(V_{on}, Q_{on})$.
- Slope standard errors and $R^2$ are reported. For a simple regression, $F=(\text{slope}/SE)^2$.
- The gas capacitance is $C_g=C_dC_{total}/(C_d-C_{total})$ (Th. 24).

**Derived quantities.** All four expressions for $U_b$ are exact in the ideal model; see `tests/test_lissajous.py`.

| Quantity | Equation |
|---|---|
| Transferred charge per half cycle (thesis $Q_{max}$, Th. 25) | $\Delta Q = Q(V_{max})-Q_{on} = C_d (V_{max}-V_{on})$ |
| **Burning voltage** | $U_b = V_{on}-Q_{on}/C_d$ |
| | $\;\;\;= (1-C_{total}/C_d)\,(V_{max}+V_{on})/2$ |
| | $\;\;\;= E/(2\Delta Q)$ |
| | $\;\;\;=$ voltage-axis intercept of the discharge line |
| Minimum ignition amplitude | $U_{min}=U_b\,(C_d+C_g)/C_d$ |
| Manley energy | $E = 4C_dU_b\,(V_{max}-U_{min})$ |
| Discharge current (Pipa & Brandenburg 2019) | $i_{dis}=\dfrac{i-C_{total}\,dv/dt}{1-C_{total}/C_d}$ |

> **Thesis Eq. (15).** $V_{breakdown}=V_{min}+Q_{max}/C_{dielectric}$ reduces identically to $V_{max}$, because $Q_{max}=C_d(V_{max}-V_{min})$. It is kept only as `thesis_breakdown_voltage()` for reproducibility. Use $U_b$ for physics.

> **Partial surface discharging** (Peeters & van de Sanden 2015). When the discharge does not cover the whole dielectric, the discharge-phase slope is an effective $\zeta_{diel}\le C_d$, and the formulas above hold with $C_d\to\zeta_{diel}$. This is relevant for surface DBDs and for damaged electrodes.

## 4. Discharge pulses (`discharge`)

- **Half-cycle window.** Each half cycle is analysed from the zero crossing to the voltage extremum, where the discharge stops (thesis T₁–T₂ and T₃–T₄).
- **`thesis` detector.** This is the original algorithm, with the index offset fixed:
  1. Run `find_peaks(width=2)` on $v\,i$.
  2. Set the prominence threshold to the median prominence × 1.5.
  3. Run `find_peaks` again with that threshold.
  4. Remove peaks whose widths are IQR outliers.
- **`robust` detector** (the default).
  - Find peaks of $|i_{dis}|$ above $k\cdot\sigma_{MAD}$ (default $k=5$), keeping only peaks with the polarity of the half cycle.
  - Because the threshold refers to the noise level, counts can be compared across captures.
- **Outputs**
  - Number of discharges: mean count per half cycle.
  - $V_{min}$: mean $|v|$ at the first pulse of each half cycle (the thesis' discharge initiation voltage).
- **Limits.** At 10 ns sampling with a 60 MHz probe, ns-scale micro-discharges are under-sampled. Counts are lower bounds and peak currents are attenuated.

## 5. Plasma parameters (`plasma`)

| Quantity | Equation | Default / comment |
|---|---|---|
| Townsend α (Th. 7) | $\alpha=Ap\,e^{-Bp/E}$ | Air: A = 15 cm⁻¹Torr⁻¹, B = 365 V cm⁻¹Torr⁻¹ |
| Paschen (Th. 8) | $V_b=\dfrac{Bpd}{\ln(Apd)-\ln\ln(1+1/\gamma)}$ | 100 µm, 1 atm: 0.72–0.87 kV |
| Gas density | $N=p/(k_BT)$ | 2.50×10¹⁹ cm⁻³ at 293 K (the thesis used 2.8×10¹⁹) |
| Reduced field (Th. 22) | $E/N=U/(dN)$ | 1 Td = 10⁻¹⁷ V cm² |
| Mobility (Th. 19) | $\mu_e=(\mu_ep)/p$, $(\mu_ep)=0.45\times10^6$ cm² Torr V⁻¹ s⁻¹ | Low-field value, 592 cm²/Vs |
| Electron density (Th. 20) | $n_e=\dfrac{i\,d}{\mu_e\,e\,A\,U}$ | Use $U=U_b$ and $i=i_{dis}$; $A$ = channel cross-section |
| Electron temperature (Th. 21) | $k_BT_e/e = D_e/\mu_e\,(E/N)$ | Table digitised from thesis Fig. 15 (source: Raizer's curve, digitised to 175 Td; table points above 175 Td are the thesis' OriginLab extrapolation); NaN outside 1–390 Td unless `extrapolate=True` |

> The $D/\mu$ table (`data/dmu_air_thesis_fig15.csv`) is a digitisation of the thesis figure, which compiles Raja Rao & Govinda Raju 1971 and Raizer 1991. For quantitative work, replace it with a Boltzmann-solver table, e.g. BOLSIG+ with LXCat cross sections for air. Strictly, $D/\mu$ is the *characteristic energy*; it equals $k_BT_e/e$ only for a Maxwellian distribution.

## 6. Features (`features.extract`)

**Thesis features.** These are the 14 inputs of thesis Table 4: `frequency`, `energy`, `v_rms`, `i_rms`, `c_dielectric`, `c_gas`, `c_total`, `v_min`, `v_breakdown` (Th. 15, ≈ V_max), `q_max`, `n_discharges`, `ne_mean`, `ne_std`, `te` (thesis chain with N = 2.8×10¹⁹ and extrapolation).

**Corrected features.**

| Feature | Meaning |
|---|---|
| `power` | Mean power [W] |
| `u_b`, `u_b_energy`, `u_b_geometry` | Burning voltage by three estimators |
| `en_td` | E/N from $U_b$ [Td] |
| `te_corrected` | T_e from $U_b$, within the table (≤ 390 Td; above 175 Td it relies on the thesis extrapolation) |
| `ne_corrected_mean` | n_e from $i_{dis}$ and $U_b$ |
| `v_max` | Peak applied voltage |
| `c_*_r2` | Fit quality |

**Metadata.** `meta_vertical_scale` (the current V/div) is stored so that confounds that differ between time points can be checked.

## 7. Machine learning (`ml`)

- **Models** (thesis hyper-parameters), each run as `Pipeline(StandardScaler, model)` so that scaling is fitted on training data only:
  - LR
  - DT (`max_depth=3`, `min_samples_split=4`)
  - RF (1000 trees)
  - GB (`lr=0.01`, 1000 trees)
  - SVM (linear)
  - kNN (k = 3)
- **`evaluate`**
  - Reports the confusion matrix, accuracy, specificity and AUC.
  - Reports precision, recall and F1 both as `*_binary` (positive class = degraded) and as `*_macro`.
- **`group_holdout`**: leave-one-device-out (or leave-one-time-point-out) evaluation.
- **`baseline_normalize`**
  - Divides each feature by its median over a device's own early, known-healthy captures.
  - This removes device-to-device offsets, which caused the LR/SVM/kNN false positives on the thesis validation device.

## References

- T. C. Manley, *Trans. Electrochem. Soc.* 84, 83 (1943).
- A. V. Pipa, R. Brandenburg, *Atoms* 7, 14 (2019).
- F. J. J. Peeters, M. C. M. van de Sanden, *Plasma Sources Sci. Technol.* 24, 015016 (2015).
- A. V. Pipa et al., *Rev. Sci. Instrum.* 83, 075111 (2012).
- Yu. P. Raizer, *Gas Discharge Physics*, Springer (1991).
- C. Raja Rao, G. R. Govinda Raju, *J. Phys. D* 4, 769 (1971).
- U. Kogelschatz, *Plasma Chem. Plasma Process.* 23, 1 (2003).
