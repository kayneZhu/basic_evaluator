"""
Sample-level jsonl: global ``sample_idx``, shard writer, resume, merge.

INTERFACE.md rules this implements:

- Canonical product is ``samples.jsonl`` under the benchmark directory.
- Workers write ``shards/{idx:03d}of{n:03d}.jsonl`` plus a sidecar
  ``.manifest.json`` declaring the owned half-open ``sample_idx`` range
  and optional ``problem_ids``.
- ``sample_idx`` is global. A shard that owns 64–127 writes
  ``sample_idx: 64``, never ``0``. Local remapping corrupts avg@8
  (indices 0–7 of the pooled draw) and ``p̂ = (k+½)/(K+1)`` (pooled K).
- Resume is keyed on ``(problem_id, sample_idx)``. Same key is not
  appended a second time.
- Merge = union, dedupe on that key, fail loudly on conflicting
  duplicates, require complete ``{0,…,K-1}`` per problem, then sort.
  Any partition of the same keys must yield a byte-identical file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


REQUIRED_FIELDS: Tuple[str, ...] = (
    "problem_id",
    "sample_idx",
    "response",
    "verified",
    "n_tokens",
)

# INTERFACE conflict fields. Extras are ignored for equality.
CONFLICT_FIELDS: Tuple[str, ...] = ("verified", "response", "n_tokens")

RecordKey = Tuple[str, int]


class ShardError(ValueError):
    """Base error for shard I/O and merge."""


class ShardManifestError(ShardError):
    """Illegal or inconsistent shard manifest."""


class ShardConflictError(ShardError):
    """Two records share a key but disagree on verified/response/n_tokens."""


class MergeIncompleteError(ShardError):
    """Merged pool is missing keys or has extras relative to problems × {0…K-1}."""


@dataclass(frozen=True)
class ShardManifest:
    shard_idx: int
    shard_count: int
    sample_idx_start: int
    sample_idx_end: int
    problem_ids: Optional[Tuple[str, ...]]

    def __post_init__(self) -> None:
        if self.shard_idx < 0 or self.shard_count <= 0:
            raise ShardManifestError(
                f"shard_idx={self.shard_idx} shard_count={self.shard_count} illegal"
            )
        if self.shard_idx >= self.shard_count:
            raise ShardManifestError(
                f"shard_idx {self.shard_idx} >= shard_count {self.shard_count}"
            )
        if self.sample_idx_start < 0 or self.sample_idx_end < self.sample_idx_start:
            raise ShardManifestError(
                f"illegal sample_idx range [{self.sample_idx_start}, {self.sample_idx_end})"
            )
        if self.problem_ids is not None and len(self.problem_ids) == 0:
            raise ShardManifestError("empty problem_ids is illegal")

    def owns(self, problem_id: str, sample_idx: int) -> bool:
        if not (self.sample_idx_start <= sample_idx < self.sample_idx_end):
            return False
        if self.problem_ids is None:
            return True
        return problem_id in self.problem_ids

    def to_json(self) -> Dict[str, Any]:
        return {
            "shard_idx": self.shard_idx,
            "shard_count": self.shard_count,
            "sample_idx_start": self.sample_idx_start,
            "sample_idx_end": self.sample_idx_end,
            "problem_ids": (
                None if self.problem_ids is None else list(self.problem_ids)
            ),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "ShardManifest":
        pids = data.get("problem_ids")
        if pids is not None:
            pids = tuple(pids)
        return cls(
            shard_idx=int(data["shard_idx"]),
            shard_count=int(data["shard_count"]),
            sample_idx_start=int(data["sample_idx_start"]),
            sample_idx_end=int(data["sample_idx_end"]),
            problem_ids=pids,
        )


def shard_stem(shard_idx: int, shard_count: int) -> str:
    return f"{shard_idx:03d}of{shard_count:03d}"


def shard_paths(benchmark_dir: Path, shard_idx: int, shard_count: int) -> Tuple[Path, Path]:
    shards = Path(benchmark_dir) / "shards"
    stem = shard_stem(shard_idx, shard_count)
    return shards / f"{stem}.jsonl", shards / f"{stem}.manifest.json"


def samples_jsonl_path(benchmark_dir: Path) -> Path:
    return Path(benchmark_dir) / "samples.jsonl"


def split_sample_axis(k: int, n_shards: int) -> List[Tuple[int, int]]:
    """
    Partition ``[0, K)`` into ``n_shards`` half-open global ranges.

    Ranges are contiguous and disjoint; their union is exactly ``[0, K)``.
    A shard writes the global indices, never a local 0..len-1 remap.
    """
    if k < 0:
        raise ValueError(f"K must be >= 0, got {k}")
    if n_shards <= 0:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    if n_shards > k > 0:
        raise ValueError(
            f"n_shards={n_shards} > K={k}: empty shards are not a legal partition"
        )
    if k == 0:
        return [(0, 0) for _ in range(n_shards)]
    base, rem = divmod(k, n_shards)
    ranges: List[Tuple[int, int]] = []
    start = 0
    for i in range(n_shards):
        size = base + (1 if i < rem else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def work_items(
    problem_ids: Sequence[str],
    sample_idx_start: int,
    sample_idx_end: int,
) -> List[RecordKey]:
    """Cartesian product with **global** ``sample_idx``. Never renumbers."""
    items: List[RecordKey] = []
    for pid in problem_ids:
        for idx in range(sample_idx_start, sample_idx_end):
            items.append((pid, idx))
    return items


def record_key(record: Mapping[str, Any]) -> RecordKey:
    return str(record["problem_id"]), int(record["sample_idx"])


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def canonical_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise ShardError(f"record missing required fields {missing}: {record!r}")
    if not isinstance(record["problem_id"], str):
        raise ShardError(f"problem_id must be str, got {type(record['problem_id'])}")
    if not _is_int(record["sample_idx"]):
        raise ShardError(f"sample_idx must be int, got {record['sample_idx']!r}")
    if record["sample_idx"] < 0:
        raise ShardError(f"sample_idx must be >= 0, got {record['sample_idx']}")
    if not isinstance(record["response"], str):
        raise ShardError(f"response must be str, got {type(record['response'])}")
    if not _is_bool(record["verified"]):
        raise ShardError(f"verified must be bool, got {record['verified']!r}")
    if not _is_int(record["n_tokens"]):
        raise ShardError(f"n_tokens must be int, got {record['n_tokens']!r}")
    out: Dict[str, Any] = {k: record[k] for k in REQUIRED_FIELDS}
    extras = [k for k in record.keys() if k not in REQUIRED_FIELDS]
    for k in sorted(extras):
        out[k] = record[k]
    return out


def dumps_record(record: Mapping[str, Any]) -> str:
    return json.dumps(
        canonical_record(record), ensure_ascii=False, separators=(",", ":")
    )


def loads_record(line: str) -> Dict[str, Any]:
    return canonical_record(json.loads(line))


def _conflict_payload(record: Mapping[str, Any]) -> Tuple[Any, ...]:
    rec = canonical_record(record)
    return tuple(rec[k] for k in CONFLICT_FIELDS)


def iter_jsonl_records(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield loads_record(line)
            except (json.JSONDecodeError, ShardError) as exc:
                raise ShardError(f"{path}:{lineno}: {exc}") from exc


def existing_keys(path: Path) -> Set[RecordKey]:
    return {record_key(r) for r in iter_jsonl_records(path)}


def records_by_key(path: Path) -> Dict[RecordKey, Dict[str, Any]]:
    out: Dict[RecordKey, Dict[str, Any]] = {}
    for rec in iter_jsonl_records(path):
        key = record_key(rec)
        if key in out and _conflict_payload(out[key]) != _conflict_payload(rec):
            raise ShardConflictError(
                f"conflicting duplicates inside {path} for {key}: "
                f"{out[key]!r} vs {rec!r}"
            )
        out[key] = rec
    return out


def write_manifest(path: Path, manifest: ShardManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_json(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_manifest(path: Path) -> ShardManifest:
    return ShardManifest.from_json(json.loads(path.read_text(encoding="utf-8")))


def validate_record_ownership(record: Mapping[str, Any], manifest: ShardManifest) -> None:
    rec = canonical_record(record)
    pid, idx = rec["problem_id"], rec["sample_idx"]
    if not manifest.owns(pid, idx):
        raise ShardError(
            f"record (problem_id={pid!r}, sample_idx={idx}) is outside shard "
            f"ownership {manifest.to_json()}; refusing local sample_idx remap"
        )


def write_shard(
    benchmark_dir: Path,
    manifest: ShardManifest,
    records: Sequence[Mapping[str, Any]],
    *,
    resume: bool = False,
) -> Path:
    """
    Write one shard's jsonl + manifest.

    ``sample_idx`` is written as given. If ``resume`` is true, skip keys
    already present; a second row for the same key is never appended. An
    incoming record that disagrees with an existing row fails loudly.
    """
    jsonl_path, manifest_path = shard_paths(
        benchmark_dir, manifest.shard_idx, manifest.shard_count
    )
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.is_file():
        existing_manifest = read_manifest(manifest_path)
        if existing_manifest != manifest:
            raise ShardManifestError(
                f"resume manifest mismatch for {manifest_path}: "
                f"{existing_manifest.to_json()} vs {manifest.to_json()}"
            )
    else:
        write_manifest(manifest_path, manifest)

    # Always key-skip. A second row for the same (problem_id, sample_idx)
    # is never appended, resume or not — INTERFACE resume rule.
    present = records_by_key(jsonl_path) if jsonl_path.is_file() else {}
    new_lines: List[str] = []
    for raw in records:
        validate_record_ownership(raw, manifest)
        rec = canonical_record(raw)
        key = record_key(rec)
        if key in present:
            if _conflict_payload(present[key]) != _conflict_payload(rec):
                raise ShardConflictError(
                    f"resume conflict for {key} in {jsonl_path}: "
                    f"{present[key]!r} vs {rec!r}"
                )
            continue
        present[key] = rec
        new_lines.append(dumps_record(rec))

    if new_lines:
        with jsonl_path.open("a", encoding="utf-8") as f:
            for line in new_lines:
                f.write(line + "\n")
    elif not jsonl_path.exists():
        jsonl_path.touch()
    return jsonl_path


def _read_fixed_n_shards(
    benchmark_dir: Path, shard_count: int
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    missing_files: List[str] = []
    for idx in range(shard_count):
        jsonl_path, manifest_path = shard_paths(benchmark_dir, idx, shard_count)
        if not jsonl_path.is_file():
            missing_files.append(str(jsonl_path))
            continue
        if manifest_path.is_file():
            manifest = read_manifest(manifest_path)
            if manifest.shard_count != shard_count or manifest.shard_idx != idx:
                raise ShardManifestError(
                    f"{manifest_path} does not match shard {idx} of {shard_count}: "
                    f"{manifest.to_json()}"
                )
            for rec in iter_jsonl_records(jsonl_path):
                validate_record_ownership(rec, manifest)
                records.append(rec)
        else:
            records.extend(iter_jsonl_records(jsonl_path))
    if missing_files:
        raise MergeIncompleteError(
            f"missing shard files for shard_count={shard_count}: {missing_files}"
        )
    return records


def merge_records(
    records: Iterable[Mapping[str, Any]],
    *,
    k: int,
    problem_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Union, dedupe on ``(problem_id, sample_idx)``, fail on conflict,
    require complete ``{0,…,K-1}`` per problem, sort.
    """
    if k < 0:
        raise ValueError(f"K must be >= 0, got {k}")
    if any(not pid for pid in problem_ids):
        raise ShardError("problem_ids must be non-empty strings")

    pooled: Dict[RecordKey, Dict[str, Any]] = {}
    for raw in records:
        rec = canonical_record(raw)
        key = record_key(rec)
        if key in pooled:
            if _conflict_payload(pooled[key]) != _conflict_payload(rec):
                raise ShardConflictError(
                    f"conflicting duplicates for {key}: "
                    f"{pooled[key]!r} vs {rec!r}"
                )
            continue
        pooled[key] = rec

    expected: Set[RecordKey] = set(work_items(problem_ids, 0, k))
    got = set(pooled.keys())
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing or extra:
        raise MergeIncompleteError(
            f"merge is not a complete problems × {{0..{k - 1}}} grid "
            f"({len(missing)} missing, {len(extra)} extra). "
            f"missing[:8]={missing[:8]} extra[:8]={extra[:8]}"
        )

    return sorted(pooled.values(), key=lambda r: (r["problem_id"], r["sample_idx"]))


def write_samples_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> bytes:
    """Write sorted canonical jsonl; return the exact file bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [dumps_record(r) for r in records]
    payload = "".join(line + "\n" for line in lines)
    path.write_text(payload, encoding="utf-8")
    return payload.encode("utf-8")


def merge_shards(
    benchmark_dir: Path,
    *,
    shard_count: int,
    k: int,
    problem_ids: Sequence[str],
) -> bytes:
    """
    Read every ``shards/{idx:03d}of{n:03d}.jsonl`` for a fixed n, merge,
    write ``samples.jsonl``. Downstream metrics must read only that file.
    """
    records = _read_fixed_n_shards(Path(benchmark_dir), shard_count)
    merged = merge_records(records, k=k, problem_ids=problem_ids)
    return write_samples_jsonl(samples_jsonl_path(Path(benchmark_dir)), merged)
