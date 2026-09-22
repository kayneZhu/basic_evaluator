"""Eval tests package.

A regular package so ``import tests`` from this checkout wins over
verl's ``tests/`` when this directory is first on ``sys.path``.
``python -m unittest tests.test_*`` from a parent directory must still
point ``-s``/``-t`` or ``PYTHONPATH`` at this repo; this file then
drops any ``third_party/verl`` entries that would shadow us.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EVAL_ROOT = Path(__file__).resolve().parent.parent


def _drop_verl_shadows() -> None:
    kept: list[str] = []
    for entry in sys.path:
        try:
            resolved = Path(entry).resolve()
        except OSError:
            kept.append(entry)
            continue
        parts = resolved.parts
        if "third_party" in parts and "verl" in parts:
            continue
        kept.append(entry)
    sys.path[:] = kept
    root = str(_EVAL_ROOT)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    mod = sys.modules.get("tests")
    if mod is not None:
        origin = getattr(mod, "__file__", None)
        if origin and "third_party" in Path(origin).parts:
            del sys.modules["tests"]
            for name in [k for k in sys.modules if k.startswith("tests.")]:
                del sys.modules[name]


_drop_verl_shadows()
