# 계산 방법: 오실로스코프 파형에서 플라즈마 변수와 손상 예측까지

[English](methods.md) | **한국어**

`src/dbd_diagnostics`에 구현된 모든 계산을 단계별로 정리했습니다. 각 단계마다 수식, 가정, 그리고 논문과의 관계를 함께 적었습니다. "(논문 n)"은 학위논문의 식 번호입니다.

## 0. 장치와 측정

- **방전원**: 대기 중에서 동작하는 마이크로 갭 동일평면 표면 DBD입니다. 구조는 다음과 같습니다.
  - 150 µm × 300 µm 크기의 전극 패드 126쌍이 100 µm 간격으로 마주 봅니다.
  - 전극은 Ag이며 두께는 약 10 µm입니다.
  - 유전체는 스크린 인쇄로 만들었고 두께는 약 40 µm입니다.
  - 구동 조건은 약 46.5 kHz, 약 1.5 kV_rms입니다.
- **측정 장비**:
  - 전압: Tektronix P6015A(1000×, 고전압 프로브)
  - 전류: Tektronix P6021A(전류 프로브)
  - 오실로스코프: GW Instek GDS-2102A
  - 캡처 하나는 5000점이며, 샘플 간격은 10 ns 또는 20 ns, 분해능은 8-bit입니다.
- **파일 읽기**: `io.load_csv`는 GW Instek "Format 2.0A" CSV와 일반 `t,v,i` CSV를 모두 읽습니다.

## 1. 주기 선택과 오프셋 (`waveform`, `electrical`)

- **주기 분할**: v(t)가 상승하며 0을 지나는 지점을 기준으로 주기를 나눕니다. 이때 진폭의 ±10% 히스테리시스를 둔 슈미트 트리거 방식을 씁니다.
- **평균**: 모든 완전 주기에 대해 각각 계산한 뒤 평균합니다. 원본 코드는 마지막 주기 하나만 사용했습니다.
- **오프셋 제거**:
  - 전압은 평균값을 빼서 제거합니다.
  - 전류는 다음 식으로 제거합니다.

    $$i \leftarrow i-\frac1T\int_0^T i\,dt$$

  - 전류 오프셋 제거는 반드시 필요합니다. 오프셋이 1 LSB(8 mA)만 있어도 주기마다 약 170 nC가 누적되는데, 이는 방전 전하의 약 3배입니다.

## 2. 주기 평균 전기량

| 물리량 | 식 | 비고 |
|---|---|---|
| 실효값 | $V_{rms}=\sqrt{\frac1T\int_0^T v^2dt}$, $I_{rms}$도 같은 방식 | 논문 9–10 |
| 1주기 에너지 | $E=\int_0^T v\,i\,dt$ [J] | 논문의 "0.13 mJ"가 이 값입니다 |
| 평균 전력 | $P=E f$ [W] | 논문 11 (이 장치에서 ≈ 6 W) |
| 전하 | $Q(t)=\int_0^t i\,dt'+Q_0$ | $\max Q=-\min Q$가 되도록 중심을 맞춥니다 (논문 23) |

## 3. 리사주 그래프와 등가회로 (`lissajous`)

**등가회로**

- 유전체 커패시턴스 $C_d$와 기체 갭 커패시턴스 $C_g$가 직렬로 연결된 모델입니다(Manley 1943).
- 방전 중에는 기체 갭 전압이 burning voltage $\pm U_b$로 고정됩니다.
- 기체 갭 전압은 다음과 같습니다.

  $$u_g(t)=v(t)-\frac{Q(t)}{C_d}$$

- 방전 전후 리사주 그래프의 기울기는 다음과 같습니다.

  | 구간 | 기울기 | 논문에서의 이름 |
  |---|---|---|
  | 방전이 없을 때 | $dQ/dV=C_{cell}=C_dC_g/(C_d+C_g)$ | $C_{total}$ (논문 24) |
  | 방전 중 | $Q=C_d(V-U_b)$이므로 $dQ/dV=C_d$ | $C_{dielectric}$ |

**적합 방법**

1. 한 주기를 전압 극값에서 나누어 상승 가지(−V_max → +V_max)와 하강 가지를 얻습니다.
2. 각 가지를 최적 분기점을 갖는 두 직선으로 최소제곱 적합합니다.
3. 첫 번째 기울기가 $C_{total}$, 두 번째 기울기가 $C_{dielectric}$이고, 분기점이 방전 개시점 $(V_{on}, Q_{on})$입니다.
4. 기울기의 표준오차와 $R^2$를 함께 보고합니다. 단순회귀에서는 $F=(\text{기울기}/SE)^2$입니다.
5. 기체 갭 커패시턴스는 $C_g=C_dC_{total}/(C_d-C_{total})$로 구합니다(논문 24).

**유도량**

| 물리량 | 식 |
|---|---|
| 반주기당 이동 전하 (논문 25의 $Q_{max}$) | $\Delta Q = C_d (V_{max}-V_{on})$ |
| Burning voltage | $U_b = V_{on}-Q_{on}/C_d = (1-C_{total}/C_d)(V_{max}+V_{on})/2 = E/(2\Delta Q)$ = 방전 직선의 V축 절편 |
| 최소 점화 진폭 | $U_{min}=U_b(C_d+C_g)/C_d$ |
| Manley 에너지 | $E=4C_dU_b(V_{max}-U_{min})$ |
| 방전전류 (Pipa & Brandenburg 2019) | $i_{dis}=\dfrac{i-C_{total}\,dv/dt}{1-C_{total}/C_d}$ |

이상적인 모델에서는 위의 네 가지 $U_b$ 표현이 모두 정확히 같은 값을 줍니다. 이는 `tests/test_lissajous.py`에서 확인합니다.

> **논문 식 (15)에 대하여**: $V_{breakdown}=V_{min}+Q_{max}/C_{dielectric}$에 $Q_{max}=C_d(V_{max}-V_{min})$을 대입하면 항등적으로 $V_{max}$가 됩니다. 따라서 이 식은 재현성 확인용 함수 `thesis_breakdown_voltage()`로만 남겨 두었습니다. 물리 해석에는 $U_b$를 사용하십시오.

> **부분 표면 방전** (Peeters & van de Sanden 2015): 방전이 유전체 전체를 덮지 않으면, 방전 구간의 기울기는 실제 $C_d$보다 작거나 같은 유효값 $\zeta_{diel}\le C_d$가 됩니다. 이 경우 위 식들에서 $C_d$를 $\zeta_{diel}$로 바꿔 쓰면 됩니다.

## 4. 방전 펄스 (`discharge`)

- **분석 구간**: 반주기마다 0점 통과부터 전압 극값까지를 분석합니다. 전압이 극값에 도달하면 방전이 멈추기 때문입니다(논문의 T₁–T₂, T₃–T₄ 구간).
- **`thesis` 검출기**: 원본 알고리즘을 그대로 따르되, 인덱스 오프셋 버그만 수정했습니다.
  1. $v\,i$에 `find_peaks(width=2)`를 적용합니다.
  2. 돌출도(prominence) 중앙값 × 1.5를 임계값으로 하여 다시 `find_peaks`를 적용합니다.
  3. 펄스 폭이 IQR 기준 이상치인 피크를 제거합니다.
- **`robust` 검출기**(기본값): $|i_{dis}|$가 $k\cdot\sigma_{MAD}$를 넘는 피크만 셉니다(기본 $k=5$). 극성은 반주기와 같은 피크만 남깁니다. 임계값이 잡음 수준을 기준으로 정해지므로 캡처끼리 비교할 수 있습니다.
- **출력값**:
  - 방전 횟수: 반주기당 펄스 수의 평균
  - $V_{min}$: 각 반주기 첫 펄스 시점의 $|v|$ 평균
- **측정 한계**: 샘플 간격 10 ns, 프로브 대역폭 60 MHz로는 ns 단위의 마이크로방전을 충분히 담을 수 없습니다. 따라서 방전 횟수는 하한값이고, 피크 전류는 실제보다 감쇠된 값입니다.

## 5. 플라즈마 변수 (`plasma`)

| 물리량 | 식 | 기본값과 설명 |
|---|---|---|
| Townsend α (논문 7) | $\alpha=Ap\,e^{-Bp/E}$ | 공기: A = 15 cm⁻¹Torr⁻¹, B = 365 V cm⁻¹Torr⁻¹ |
| Paschen (논문 8) | $V_b=\dfrac{Bpd}{\ln(Apd)-\ln\ln(1+1/\gamma)}$ | 100 µm, 1 atm에서 0.72–0.87 kV |
| 기체 수밀도 | $N=p/(k_BT)$ | 293 K에서 2.50×10¹⁹ cm⁻³ (논문은 2.8×10¹⁹ 사용) |
| 환산 전기장 (논문 22) | $E/N=U/(dN)$ | 1 Td = 10⁻¹⁷ V cm² |
| 이동도 (논문 19) | $\mu_e=(\mu_ep)/p$ | 저전계 값 592 cm²/Vs |
| 전자밀도 (논문 20) | $n_e=\dfrac{i\,d}{\mu_e e A U}$ | $U=U_b$, $i=i_{dis}$ 권장 |
| 전자온도 (논문 21) | $k_BT_e/e=D_e/\mu_e(E/N)$ | 논문 그림 15를 디지타이즈한 표 사용. 1–390 Td 범위 밖에서는 NaN을 반환하며, `extrapolate=True`일 때만 외삽합니다 |

> **D/μ 표에 대하여**: 이 표는 논문 그림 15(Raja Rao & Govinda Raju 1971, Raizer 1991)를 디지타이즈한 것입니다. 정량 분석에는 BOLSIG+와 LXCat 공기 단면적 데이터로 계산한 표를 쓰는 것을 권장합니다.
>
> 또한 D/μ는 엄밀히 말해 *특성 에너지(characteristic energy)* 이며, 전자 에너지 분포가 Maxwell 분포일 때만 $k_BT_e/e$와 같습니다.

## 6. 특성 추출 (`features.extract`)

- **논문 특성 14개**(논문 표 4): `frequency`, `energy`, `v_rms`, `i_rms`, `c_dielectric`, `c_gas`, `c_total`, `v_min`, `v_breakdown`(≈ V_max), `q_max`, `n_discharges`, `ne_mean`, `ne_std`, `te`
- **보정 특성**:

  | 이름 | 의미 |
  |---|---|
  | `power` | 평균 전력 |
  | `u_b`, `u_b_energy`, `u_b_geometry` | 세 가지 방법으로 구한 burning voltage |
  | `en_td` | 환산 전기장 [Td] |
  | `te_corrected` | $U_b$로 구한 전자온도 (표 범위 안에서만 계산) |
  | `ne_corrected_mean` | $i_{dis}$와 $U_b$로 구한 전자밀도 |
  | `v_max` | 인가전압 최댓값 |
  | `c_*_r2` | 커패시턴스 적합의 $R^2$ |

- **메타데이터**: 세션마다 다를 수 있는 교란 요인을 점검할 수 있도록 전류 채널의 V/div 값을 `meta_vertical_scale`로 저장합니다.

## 7. 머신러닝 (`ml`)

- **모델**: 논문의 하이퍼파라미터를 그대로 사용합니다.
  - LR
  - DT(max_depth=3, min_samples_split=4)
  - RF(트리 1000개)
  - GB(learning rate 0.01, 트리 1000개)
  - SVM(linear)
  - kNN(k = 3)
- **스케일링**: 모든 모델을 `Pipeline(StandardScaler, model)`로 묶어, 스케일러가 학습 데이터에만 맞춰지도록 했습니다.
- **`evaluate`**: 다음 지표를 계산합니다.
  - 혼동행렬
  - 정확도, 특이도, AUC
  - 정밀도·재현도·F1: binary(손상 = 양성)와 macro 평균을 각각 따로 표기
- **`group_holdout`**: 장치 단위(또는 세션 단위)로 하나를 빼고 평가합니다(leave-one-group-out).
- **`baseline_normalize`**: 각 장치에서 초기 정상 캡처들의 중앙값으로 특성을 나눕니다. 이렇게 하면 장치 간 기준값 차이가 제거됩니다. 논문의 검증 장치에서 LR·SVM·kNN이 정상을 손상으로 잘못 분류한 원인이 바로 이 차이였습니다.

## 참고문헌

- T. C. Manley, *Trans. Electrochem. Soc.* 84, 83 (1943)
- A. V. Pipa, R. Brandenburg, *Atoms* 7, 14 (2019)
- F. J. J. Peeters, M. C. M. van de Sanden, *Plasma Sources Sci. Technol.* 24, 015016 (2015)
- A. V. Pipa 외, *Rev. Sci. Instrum.* 83, 075111 (2012)
- Yu. P. Raizer, *Gas Discharge Physics*, Springer (1991)
- C. Raja Rao, G. R. Govinda Raju, *J. Phys. D* 4, 769 (1971)
- U. Kogelschatz, *Plasma Chem. Plasma Process.* 23, 1 (2003)
