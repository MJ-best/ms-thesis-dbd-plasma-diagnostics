"""Every number in docs/dbd-physics-primer.ko.md, computed step by step.

    python examples/physics_worked_numbers.py

Runs without the raw data: the measured inputs are session medians quoted in the primer
(device M1 of the thesis, and ATI #2 / ATI #3 of the aging test, which is kept separate).
"""
import numpy as np
from scipy.optimize import brentq

from dbd_diagnostics import plasma

e, eps0, me, amu = 1.602176634e-19, 8.8541878128e-12, 9.1093837e-31, 1.66053907e-27
A_AIR, B_AIR = 15.0, 365.0  # Townsend constants for air [1/(cm Torr)], [V/(cm Torr)]
P_TORR = 760.0

# Measured inputs (medians)
M1 = dict(f=47.08e3, V_rms=1482.0, V_max=2250.0, I_rms=19.7e-3, C_cell=11.16e-12, C_d=32.39e-12, U_b=842.0,
          E_meas=114.1e-6, P_meas=5.41)
N_PAIRS, PAD_AREA_M2, T_DIEL_M, GAP_CM = 126, 150e-6 * 300e-6, 40e-6, 0.01
ATI2 = dict(f=108.17e3, V_rms=1483.2, I_rms=53.1e-3, C_cell=49.0e-12)  # session 2020-01-13
ATI2_SESSIONS_C = [47.7, 46.4, 46.7, 47.4, 47.9, 48.3, 49.0, 45.7, 49.5, 46.1, 47.4, 48.2]  # C_cell [pF] per session
M_SESSION_SD = {"M1": (1.93, 11.74), "M2": (0.80, 12.72), "M3": (3.59, 15.39)}  # SD and mean of session C_cell [pF]
ATI3_ELASTICITY = -0.87  # slope of log V_rms on log C_cell over the ATI #3 sessions (examples/ati_aging_test.py)
M1_I_LF = 9.2e-3  # part of the M1 current below 1 MHz, 5 h session, 10 ns captures (examples/current_spectrum_check.py)


def step(title):
    print(f"\n== {title}")


step("1. gas state and electron energy gain")
N = plasma.number_density()  # 293.15 K, 1 atm, cm^-3
print(f"N = p/(k_B T) = {N:.3e} cm^-3")
for sigma in (1e-16, 1e-15):
    lam = 1 / (N * sigma)
    print(f"sigma = {sigma:.0e} cm^2: lambda = {lam * 1e4:.2f} um, field to gain 15.6 eV in one lambda = {15.6 / lam / 1e3:.0f} kV/cm")
print(f"energy fraction lost per elastic collision with N2, 2 m_e/M = {2 * me / (28 * amu):.1e}")

step("2. breakdown: Townsend and Paschen")
for g in (0.01, 0.1):
    print(f"gamma = {g}: alpha d = ln(1 + 1/gamma) = {np.log(1 + 1 / g):.2f};  V_b at d = 50/100/150/200 um:",
          " / ".join(f"{plasma.paschen_voltage(P_TORR, d * 1e-4, gamma=g):.0f}" for d in (50, 100, 150, 200)), "V")
for crit, name in ((np.log(101), "Townsend, gamma = 0.01"), (18.0, "Meek, alpha d = 18"), (20.0, "Meek, alpha d = 20")):
    a = crit / GAP_CM
    E = B_AIR * P_TORR / np.log(A_AIR * P_TORR / a)
    print(f"{name:24s} alpha = {a:5.0f} /cm  E = {E / 1e3:5.1f} kV/cm  V = {E * GAP_CM:4.0f} V  (d = 100 um)")
pd_ = np.logspace(-1, 3, 20000)
Vp = B_AIR * pd_ / (np.log(A_AIR * pd_) - np.log(np.log(1 + 1 / 0.01)))
Vp[Vp <= 0] = np.inf
k = np.argmin(Vp)
print(f"Paschen minimum (gamma 0.01): {Vp[k]:.0f} V at pd = {pd_[k]:.2f} Torr cm (d = {pd_[k] / P_TORR * 1e4:.0f} um at 1 atm)")
for g in (0.01, 0.1):
    x = brentq(lambda x: B_AIR * x / (np.log(A_AIR * x) - np.log(np.log(1 + 1 / g))) - M1["U_b"], 2, 100)
    print(f"gamma = {g}: V_b = U_b = {M1['U_b']:.0f} V at pd = {x:.2f} Torr cm -> d = {x / P_TORR * 1e4:.0f} um")

step("3. capacitances of the cell")
C_g = M1["C_d"] * M1["C_cell"] / (M1["C_d"] - M1["C_cell"])
print(f"C_g = C_d C_cell/(C_d - C_cell) = {C_g * 1e12:.1f} pF")
print(f"per pair: C_cell = {M1['C_cell'] / N_PAIRS * 1e15:.0f} fF, C_d = {M1['C_d'] / N_PAIRS * 1e15:.0f} fF")
C_pad = 2 * M1["C_d"] / N_PAIRS  # two pads in series per pair
print(f"eps_r needed if C_d were the parallel plate over one pad (40 um): {C_pad * T_DIEL_M / (eps0 * PAD_AREA_M2):.0f}")
print(f"same with the circuit-model C_d = 39 pF: {2 * 39e-12 / N_PAIRS * T_DIEL_M / (eps0 * PAD_AREA_M2):.0f}")
for er in (7, 10):
    print(f"eps_r = {er}: parallel-plate C_d (126 pairs) = {N_PAIRS * eps0 * er * PAD_AREA_M2 / T_DIEL_M / 2 * 1e12:.1f} pF")
w = 2 * np.pi * ATI2["f"]
print(f"ATI #2: omega C_cell V_rms = {w * ATI2['C_cell'] * ATI2['V_rms'] * 1e3:.1f} mA, measured I_rms = {ATI2['I_rms'] * 1e3:.1f} mA, "
      f"C_eff = I/(omega V) = {ATI2['I_rms'] / (w * ATI2['V_rms']) * 1e12:.1f} pF")
w1 = 2 * np.pi * M1["f"]
print(f"M1: omega C_cell V_rms = {w1 * M1['C_cell'] * M1['V_rms'] * 1e3:.1f} mA, measured I_rms = {M1['I_rms'] * 1e3:.1f} mA")

step("4. ignition and charge transfer (Manley)")
div = M1["C_d"] / (M1["C_d"] + C_g)
U_min = M1["U_b"] * (M1["C_d"] + C_g) / M1["C_d"]
V_on = 2 * U_min - M1["V_max"]
dQ = M1["C_d"] * (M1["V_max"] - V_on)
print(f"gap share before ignition u_g/v = C_d/(C_d + C_g) = {div:.3f}")
print(f"U_min = U_b (1 + C_g/C_d) = {U_min:.0f} V;  V_max = {M1['V_max']:.0f} V")
print(f"onset on the rising branch V_on = 2 U_min - V_max = {V_on:.0f} V")
print(f"charge per half period dQ = C_d (V_max - V_on) = {dQ * 1e9:.1f} nC  ({dQ / N_PAIRS * 1e9:.2f} nC per pair, "
      f"{dQ / N_PAIRS / e:.1e} electrons)")

step("5. energy and power")
E_m = 4 * M1["C_d"] * M1["U_b"] * (M1["V_max"] - U_min)
print(f"Manley E = 4 C_d U_b (V_max - U_min) = {E_m * 1e6:.1f} uJ;  2 U_b dQ = {2 * M1['U_b'] * dQ * 1e6:.1f} uJ")
print(f"measured E = {M1['E_meas'] * 1e6:.1f} uJ;  P = f E: Manley {M1['f'] * E_m:.2f} W, measured {M1['P_meas']:.2f} W")

step("6. self-limitation")
t_on = (np.pi / 2 - np.arcsin(V_on / M1["V_max"])) / w1
print(f"discharge phase V_on -> V_max lasts {t_on * 1e6:.2f} us; mean discharge current dQ/t_on = {dQ / t_on * 1e3:.1f} mA; "
      f"upper bound C_d omega V_max cos(.) = {M1['C_d'] * w1 * M1['V_max'] * np.cos(np.arcsin(V_on / M1['V_max'])) * 1e3:.1f} mA")
th = np.linspace(-np.pi / 2, np.pi / 2, 200001)  # rising half period of v = V_max sin(th)
C_th = np.where(th > np.arcsin(V_on / M1["V_max"]), M1["C_d"], M1["C_cell"])  # slope dQ/dv of the Manley cycle
I_manley = np.sqrt(np.mean((C_th * w1 * M1["V_max"] * np.cos(th)) ** 2))
print(f"Manley current i = (C_cell or C_d) dv/dt: rms {I_manley * 1e3:.1f} mA "
      f"(measured, M1 5 h: below 1 MHz {M1_I_LF * 1e3:.1f} mA, total {M1['I_rms'] * 1e3:.1f} mA; examples/current_spectrum_check.py)")

step("7. supply: voltage source or current source?")
eps = ATI3_ELASTICITY
x = np.sqrt(-eps / (1 + eps))
C0, f0, V0 = 48e-12, 107e3, 1500.0
print(f"elasticity d ln V / d ln C = {eps}:  omega C R_s = {x:.2f}, R_s = {x / (2 * np.pi * f0 * C0) / 1e3:.0f} kOhm, "
      f"open-circuit V_s = {V0 * np.sqrt(1 + x ** 2):.0f} V_rms (at C = 48 pF, V = 1500 V)")
print(f"C_cell 48 -> 98 pF: V ratio current source {48 / 98:.2f}, this model {(98 / 48) ** eps:.2f}, measured {583 / 1510:.2f}")

step("8. plasma parameters (assumption-sensitive)")
for d_um in (100, 150, 300):
    EN = plasma.reduced_field(M1["U_b"], d_um * 1e-4, N)
    print(f"d = {d_um} um: E/N = {EN / 1e-17:.0f} Td, T_e (thesis D/mu table) = "
          f"{plasma.electron_temperature_from_field(M1['U_b'], d_um * 1e-4, N):.2f} eV")
i_pair = dQ / t_on / N_PAIRS
ne = plasma.electron_density(i_pair, GAP_CM, 300e-4 * 100e-4, M1["U_b"], plasma.electron_mobility())
print(f"n_e = i d/(mu e A U), i = {i_pair * 1e3:.3f} mA per pair, A = 300 um x 100 um: {ne:.1e} cm^-3 (phase average)")

step("9. detection limit of a local change")
for dev, (sd, mean) in M_SESSION_SD.items():
    dmin = 1.96 * np.sqrt(2) * sd
    print(f"{dev}: session SD {sd:.2f} pF -> 95 % detectable change {dmin:.1f} pF = {dmin / (mean / N_PAIRS):.0f} pairs "
          f"(full loss of their C_cell); {dmin / (0.2 * mean / N_PAIRS):.0f} pairs if each changes by 20 %")
sd2 = np.std(ATI2_SESSIONS_C, ddof=1)
c_pair = np.mean(ATI2_SESSIONS_C) / (np.mean([v[1] for v in M_SESSION_SD.values()]) / N_PAIRS)
dmin = 1.96 * np.sqrt(2) * sd2
print(f"ATI #2: session SD {sd2:.2f} pF -> {dmin:.1f} pF = {dmin / (np.mean(ATI2_SESSIONS_C) / c_pair):.0f} pair-equivalents; "
      f"the device is {c_pair:.0f} pair-equivalents; ATI #3 rise +50 pF = {50 / (np.mean(ATI2_SESSIONS_C) / c_pair):.0f}")
