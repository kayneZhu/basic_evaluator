"""Refuse to collect anything under a verl checkout."""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    try:
        parts = Path(collection_path).resolve().parts
    except OSError:
        return False
    return "third_party" in parts


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    leaked = [item for item in items if "third_party" in Path(str(item.path)).parts]
    if leaked:
        listing = "\n".join(str(item.path) for item in leaked)
        raise pytest.UsageError(
            "collected tests under third_party/ "
            "(verl's suite would shadow ours):\n"
            f"{listing}"
        )
