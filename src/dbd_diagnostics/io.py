"""Oscilloscope CSV loading.

Supports the GW Instek GDS-2000 series "Format 2.0A" CSV (used in the thesis,
GDS-2102A; one file per capture, ``ALLxxxx.CSV`` with CH1 = voltage and
CH2 = current side by side) and plain 3-column CSV files ``time, voltage, current``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]


@dataclass
class Waveform:
    """A single voltage/current capture.

    Attributes
    ----------
    t : time [s]
    v : applied voltage [V]
    i : measured (total) current [A]
    meta : header information (probe ratios, scales, timestamp, ...)
    """

    t: np.ndarray
    v: np.ndarray
    i: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.t)))

    @property
    def fs(self) -> float:
        return 1.0 / self.dt

    def __len__(self) -> int:
        return len(self.t)


def _is_gwinstek(lines: list[str]) -> bool:
    return bool(lines) and lines[0].startswith("Format,")


def _parse_gwinstek(path: Path, lines: list[str]) -> Waveform:
    meta: dict = {}
    data_start = None
    for k, line in enumerate(lines):
        cells = [c.strip() for c in line.rstrip("\n").split(",")]
        if cells and cells[0] == "Waveform Data":
            data_start = k + 1
            break
        if len(cells) >= 2 and cells[0]:
            key = cells[0]
            ch1 = cells[1]
            ch2 = cells[3] if len(cells) >= 4 else None
            meta[key] = ch1 if ch2 in (None, "") else (ch1, ch2)
    if data_start is None:
        raise ValueError(f"{path}: 'Waveform Data' marker not found")
    arr = np.genfromtxt(lines[data_start:], delimiter=",")
    arr = np.atleast_2d(arr)
    # Format 2.0A with two channels: t1, ch1, t2, ch2, (trailing empty)
    if arr.shape[1] >= 4 and not np.all(np.isnan(arr[:, 3])):
        t, v, i = arr[:, 0], arr[:, 1], arr[:, 3]
    else:
        raise ValueError(
            f"{path}: expected a two-channel capture (CH1 voltage, CH2 current)"
        )
    units = meta.get("Vertical Units")
    if isinstance(units, tuple) and units[0] == "A" and units[1] == "V":
        v, i = i, v  # channels swapped
    meta["source_file"] = str(path)
    return Waveform(t=t, v=v, i=i, meta=meta)


def load_csv(path: PathLike) -> Waveform:
    """Load an oscilloscope capture.

    * GW Instek ``Format 2.0A`` files are detected automatically.
    * Otherwise the file must have three numeric columns ``t, v, i``
      (an optional single header row is skipped).
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    if _is_gwinstek(lines):
        return _parse_gwinstek(path, lines)
    arr = np.genfromtxt(lines, delimiter=",", skip_header=0)
    arr = np.atleast_2d(arr)
    if np.isnan(arr[0]).any():
        arr = arr[1:]
    if arr.shape[1] < 3:
        raise ValueError(f"{path}: need 3 columns (t, v, i)")
    return Waveform(t=arr[:, 0], v=arr[:, 1], i=arr[:, 2], meta={"source_file": str(path)})


def save_csv(wf: Waveform, path: PathLike) -> None:
    """Write a waveform as a plain ``time,voltage,current`` CSV."""
    header = "time_s,voltage_V,current_A"
    np.savetxt(path, np.column_stack([wf.t, wf.v, wf.i]), delimiter=",", header=header, comments="")
