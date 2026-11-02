"""dbd-diagnostics: electrical diagnostics and ML-based surface-damage prediction
for atmospheric-pressure (surface) dielectric barrier discharges.

Reference implementation of the methods in
M. Kim, M.S. thesis, Kwangwoon University (2023), with the corrections documented
in docs/thesis-review.md.
"""
from . import discharge, electrical, features, io, lissajous, plasma, simulate, waveform

__all__ = ["discharge", "electrical", "features", "io", "lissajous", "plasma", "simulate", "waveform"]
__version__ = "1.0.0"
