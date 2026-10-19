"""Small shared helpers."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def to_jsonable(o):
    """Recursively convert numpy scalars/arrays and replace NaN/inf by None so the result is strict JSON."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write_json(path: Path, obj, indent: int = 1) -> None:
    path.write_text(json.dumps(to_jsonable(obj), indent=indent, default=str, allow_nan=False), encoding="utf-8")
