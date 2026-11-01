r"""Plasma-parameter estimates from electrical data (thesis Sec. 1.1, 3.1.5-3.1.6).

These are *order-of-magnitude, cell-averaged* estimates based on a uniform-field
picture of the 100 um micro-gap. Read docs/methods.md and docs/thesis-review.md
for the assumptions and their limits before quoting the numbers.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np

E_CHARGE = 1.602176634e-19  # [C]
K_B = 1.380649e-23  # [J/K]
TORR_PER_PA = 1.0 / 133.322368
TOWNSEND = 1e-17  # 1 Td = 1e-17 V cm^2


# --------------------------------------------------------------------------- #
# Townsend / Paschen (thesis Eqs. 1-8)
# --------------------------------------------------------------------------- #
def townsend_alpha(p_torr: float, E: float, A: float = 15.0, B: float = 365.0) -> float:
    """First Townsend coefficient alpha = A p exp(-B p / E)  [1/cm]  (Eq. 7).

    This is Townsend's semi-empirical formula (not the Saha equation).
    Defaults are the usual air constants A = 15 1/(cm Torr), B = 365 V/(cm Torr)
    (Raizer, *Gas Discharge Physics*, 1991). ``E`` in V/cm.
    """
    return A * p_torr * np.exp(-B * p_torr / E)


def paschen_voltage(p_torr: float, d_cm: float, A: float = 15.0, B: float = 365.0, gamma: float = 0.01) -> float:
    """Paschen breakdown voltage  V_b = B p d / [ln(A p d) - ln(ln(1 + 1/gamma))]  (Eq. 8).

    For air at 760 Torr and d = 100 um (pd = 7.6 Torr cm) this gives
    ~0.72-0.87 kV for gamma = 0.1-0.01.
    """
    pd = p_torr * d_cm
    den = np.log(A * pd) - np.log(np.log(1.0 + 1.0 / gamma))
    return float(B * pd / den) if den > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Gas state
# --------------------------------------------------------------------------- #
def number_density(p_pa: float = 101325.0, T_k: float = 293.15) -> float:
    """Ideal-gas number density N = p / (k_B T)  [cm^-3].

    1 atm: 2.69e19 (273 K), 2.50e19 (293 K), 2.46e19 (298 K).
    (The thesis used 2.8e19 cm^-3 and called it "Avogadro's number per cm^3";
    the correct name is the Loschmidt number / gas number density.)
    """
    return p_pa / (K_B * T_k) * 1e-6


def reduced_field(U: float, d_cm: float, N_cm3: float) -> float:
    """Reduced electric field E/N for a uniform gap  [V cm^2]  (Eq. 22).

    Divide by 1e-17 to get Townsend (Td); the thesis axis unit 1e-16 V cm^2 = 10 Td.
    """
    return U / (d_cm * N_cm3)


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #
MU_P_AIR = 0.45e6  # (mu_e p) for air [cm^2 Torr / (V s)], thesis Eq. 19


def electron_mobility(p_torr: float = 760.0, mu_p: float = MU_P_AIR) -> float:
    """Low-field electron mobility mu_e = (mu_e p)/p  [cm^2/(V s)]  (Eq. 19).

    592 cm^2/(V s) at 760 Torr. Note: mobility decreases at the high E/N of
    a micro-gap breakdown (hundreds of Td); this constant is an approximation.
    """
    return mu_p / p_torr


def electron_density(i_peak: float, d_cm: float, area_cm2: float, U: float, mu_e: float) -> float:
    """Electron density from Ohm's law  n_e = i d / (mu_e e A U)  [cm^-3]  (Eqs. 16-20).

    ``U`` must be the *gap* voltage (burning voltage), not the applied peak
    voltage, and ``i_peak`` should be the *discharge* current (displacement
    current removed). ``area_cm2`` is the cross-section of the current channel.
    """
    return i_peak * d_cm / (mu_e * E_CHARGE * area_cm2 * U)


# --------------------------------------------------------------------------- #
# Electron temperature via the characteristic energy D/mu (Eq. 21)
# --------------------------------------------------------------------------- #
def characteristic_energy(EN_vcm2: float, table_EN: np.ndarray, table_Dmu: np.ndarray,
                          extrapolate: bool = False) -> float:
    """Interpolate D_e/mu_e [V] at E/N [V cm^2] on a log-log table.

    Returns NaN outside the table unless ``extrapolate=True`` (the thesis
    operating point E/N ~ 8e-15 V cm^2 = 800 Td lies in the extrapolated region of
    its Fig. 15; values there are not supported by the cited data). Table points
    above 175 Td are already the thesis' OriginLab extrapolation of Raizer's curve.
    """
    x = np.log(np.asarray(table_EN, float))
    y = np.log(np.asarray(table_Dmu, float))
    xe = np.log(EN_vcm2)
    if not extrapolate and (xe < x.min() or xe > x.max()):
        return float("nan")
    if xe > x.max():
        s = (y[-1] - y[-2]) / (x[-1] - x[-2])
        return float(np.exp(y[-1] + s * (xe - x[-1])))
    if xe < x.min():
        s = (y[1] - y[0]) / (x[1] - x[0])
        return float(np.exp(y[0] + s * (xe - x[0])))
    return float(np.exp(np.interp(xe, x, y)))


def electron_temperature_eV(Dmu_V: float) -> float:
    """Einstein relation k_B T_e / e = D_e / mu_e  ->  T_e [eV] = D_e/mu_e [V]  (Eq. 21).

    Strictly valid for a Maxwellian EEDF only; for non-equilibrium discharges
    D/mu is the 'characteristic energy', commonly used as an effective T_e.
    """
    return float(Dmu_V)


@lru_cache(maxsize=1)
def dmu_table() -> tuple[np.ndarray, np.ndarray]:
    """(E/N [V cm^2], D/mu [V]) for air, digitised from thesis Fig. 15 (see the CSV header)."""
    text = resources.files("dbd_diagnostics").joinpath("data/dmu_air_thesis_fig15.csv").read_text()
    rows = [ln for ln in text.splitlines() if ln and not ln.startswith("#")][1:]
    arr = np.array([[float(x) for x in ln.split(",")] for ln in rows])
    return arr[:, 0], arr[:, 1]


def electron_temperature_from_field(U: float, d_cm: float, N_cm3: float, extrapolate: bool = False) -> float:
    """T_e [eV] = D/mu(E/N) with E/N = U/(d N)  (thesis Eqs. 21-22 + Fig. 15)."""
    en, dmu = dmu_table()
    return electron_temperature_eV(characteristic_energy(reduced_field(U, d_cm, N_cm3), en, dmu, extrapolate))
