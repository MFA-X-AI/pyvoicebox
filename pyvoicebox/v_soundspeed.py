"""V_SOUNDSPEED - Speed of sound, density and impedance of air."""

from __future__ import annotations
import numpy as np


def v_soundspeed(t=20, p=1, m=0.0289644, g=1.4) -> tuple[float, float, float]:
    """Calculate speed of sound, density, and acoustic impedance.

    Parameters
    ----------
    t : float, optional
        Air temperature in Celsius. Default 20.
    p : float, optional
        Air pressure in atm. Default 1.
    m : float, optional
        Average molecular weight of air in kg/mol. Default 0.0289644.
    g : float, optional
        Adiabatic constant for air. Default 1.4.

    Returns
    -------
    v : float
        Speed of sound in m/s.
    d : float
        Density of air in kg/m^3.
    z : float
        Characteristic impedance in Pa.s/m.
    """
    p_pa = p * 101325  # convert atm to pascal
    k = t + 273.15     # absolute temperature
    r = 8.3144         # J/(mol K) universal gas constant
    d = p_pa * m / (r * k)
    v = np.sqrt(g * r * k / m)
    z = v * d
    return v, d, z
