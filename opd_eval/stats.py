"""
§5.1 statistics on merged ``samples.jsonl`` only.

Estimators (INTERFACE.md / proposal §2.2 / §5.1):

- pass@k: Chen et al. (2021) unbiased estimator
  ``1 - C(n-c,k)/C(n,k)`` (numerically stable product form), for k=1..n.
  Per-problem ``(n, c)`` and the curve are the primary metric.
- raw_pass_bit / ``pass_at_k_bit``: legacy ``1[c ≥ 1]`` kept for learned /
  forgotten / McNemar which still operate on a single pass@K bit.
- avg@8: mean of ``verified`` over global ``sample_idx`` 0–7 of the **same**
  pass@K pool. Missing any of those eight indices is an error.
- ``p̂ = (k + ½) / (K + 1)`` (shrinkage). ``Δlog p̂`` is ckpt minus Base
  on the same ``problem_id``, natural log.
- learned / forgotten: Base 0→ckpt 1 / Base 1→ckpt 0 on the raw pass@K bits.
  On OR1-200 bins the counts are divided by the feasible denominators
  (learned / # Base-unsolvable in the bin; forgotten / # Base-solvable).
  Bins are an **input** (training-side pass@16). Do not read
  ``pass_count`` / ``pass_rate`` from ``ttn_test_200.jsonl``.
- Two-level bootstrap: resample problems, then resample that problem's K
  completions. Paired stats resample Base and ckpt completions independently
  after the shared problem draw. Default ``n_boot=10000``: §5.1 decides
  wins partly on whether a 95% CI excludes zero for effects as small as
  3.0pp, and 1000 resamples leaves visible Monte-Carlo noise in the CI
  endpoints while the extra cost is negligible on CPU.
- McNemar: paired pass@K bits vs Base. Chi-square without continuity
  correction, plus the exact two-sided binomial p-value on discordant pairs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .samples import canonical_record, iter_jsonl_records, record_key


class StatsError(ValueError):
    """Illegal sample pool or statistic input."""


VerifiedSeq = Tuple[bool, ...]
Pool = Dict[str, VerifiedSeq]

# Configurable; 10000 is the §5.1 default (was 1000). See module docstring.
DEFAULT_N_BOOT = 10000


def shrinkage_p_hat(k_successes: int, k_protocol: int) -> float:
    if k_protocol < 0:
        raise StatsError(f"K must be >= 0, got {k_protocol}")
    if k_successes < 0 or k_successes > k_protocol:
        raise StatsError(f"k={k_successes} outside [0, {k_protocol}]")
    return (k_successes + 0.5) / (k_protocol + 1)


def log_p_hat(k_successes: int, k_protocol: int) -> float:
    return math.log(shrinkage_p_hat(k_successes, k_protocol))


def delta_log_p_hat(k_base: int, k_ckpt: int, k_protocol: int) -> float:
    return log_p_hat(k_ckpt, k_protocol) - log_p_hat(k_base, k_protocol)


def pass_at_k_bit(k_successes: int) -> int:
    """Legacy raw bit ``1[c ≥ 1]``. Prefer :func:`unbiased_pass_at_k` for metrics."""
    return 1 if k_successes >= 1 else 0


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Chen et al. 2021: ``1 - C(n-c, k) / C(n, k)``, numerically stable.

    Uses the product form
    ``1 - ∏_{i=0}^{k-1} (n - c - i) / (n - i)`` so large ``n`` does not
    overflow ``math.comb``. Returns 0 when ``c == 0`` or ``k == 0``; 1 when
    ``n - c < k`` (every k-subset hits a success).
    """
    if n < 0 or c < 0 or k < 0:
        raise StatsError(f"n={n}, c={c}, k={k} must be non-negative")
    if c > n:
        raise StatsError(f"c={c} > n={n}")
    if k == 0:
        return 0.0
    if c == 0:
        return 0.0
    if k > n:
        raise StatsError(f"k={k} > n={n}")
    if n - c < k:
        return 1.0
    # Product over i=0..k-1 of (n-c-i)/(n-i).
    prob_miss = 1.0
    for i in range(k):
        prob_miss *= (n - c - i) / (n - i)
    return 1.0 - prob_miss


def unbiased_pass_at_k_curve(n: int, c: int) -> Dict[int, float]:
    """``{k: unbiased_pass_at_k(n, c, k)}`` for ``k = 1 .. n``."""
    if n < 0 or c < 0 or c > n:
        raise StatsError(f"illegal (n, c)=({n}, {c})")
    return {k: unbiased_pass_at_k(n, c, k) for k in range(1, n + 1)}


def _successes(verified: Sequence[bool]) -> int:
    return sum(1 for v in verified if v)


def load_samples(path: Path) -> List[Dict[str, Any]]:
    return [canonical_record(r) for r in iter_jsonl_records(Path(path))]


def pool_from_records(records: Iterable[Mapping[str, Any]], k_protocol: int) -> Pool:
    """
    Build ``problem_id → verified[sample_idx]`` of length ``K``.

    Requires a complete ``{0,…,K-1}`` grid. Extra indices are ignored only
    if the required range is present; gaps fail. Local remapping cannot
    hide here because keys are the stored ``sample_idx``.
    """
    if k_protocol <= 0:
        raise StatsError(f"K must be >= 1, got {k_protocol}")
    by_pid: Dict[str, Dict[int, bool]] = {}
    for raw in records:
        rec = canonical_record(raw)
        pid, idx = record_key(rec)
        by_pid.setdefault(pid, {})
        if idx in by_pid[pid] and by_pid[pid][idx] != rec["verified"]:
            raise StatsError(
                f"conflicting verified bits for ({pid!r}, {idx}): "
                f"{by_pid[pid][idx]} vs {rec['verified']}"
            )
        by_pid[pid][idx] = rec["verified"]

    pool: Pool = {}
    for pid, bits in by_pid.items():
        missing = [i for i in range(k_protocol) if i not in bits]
        if missing:
            raise StatsError(
                f"problem {pid!r} missing sample_idx {missing[:8]} "
                f"for K={k_protocol} (cannot compute pass@K / p̂ / avg@8)"
            )
        pool[pid] = tuple(bool(bits[i]) for i in range(k_protocol))
    if not pool:
        raise StatsError("empty sample pool")
    return pool


def load_pool(path: Path, k_protocol: int) -> Pool:
    return pool_from_records(load_samples(path), k_protocol)


def aligned_problem_ids(base: Pool, ckpt: Pool) -> List[str]:
    if set(base) != set(ckpt):
        raise StatsError(
            f"Base/ckpt problem_id sets differ: "
            f"only_base={sorted(set(base) - set(ckpt))[:8]} "
            f"only_ckpt={sorted(set(ckpt) - set(base))[:8]}"
        )
    for pid in base:
        if len(base[pid]) != len(ckpt[pid]):
            raise StatsError(
                f"{pid}: Base K={len(base[pid])} != ckpt K={len(ckpt[pid])}"
            )
    return sorted(base)


@dataclass(frozen=True)
class PerProblemStats:
    problem_id: str
    k: int
    k_protocol: int
    pass_bit: int
    p_hat: float
    log_p_hat: float
    avg8: Optional[float]
    pass_at_k: Optional[Dict[int, float]] = None

    @property
    def n(self) -> int:
        return self.k_protocol

    @property
    def c(self) -> int:
        return self.k


def per_problem_stats(pool: Pool, k_protocol: int) -> Dict[str, PerProblemStats]:
    out: Dict[str, PerProblemStats] = {}
    for pid, verified in pool.items():
        if len(verified) != k_protocol:
            raise StatsError(f"{pid}: len={len(verified)} != K={k_protocol}")
        k = _successes(verified)
        avg8 = None
        if k_protocol >= 8:
            avg8 = sum(1.0 if verified[i] else 0.0 for i in range(8)) / 8.0
        out[pid] = PerProblemStats(
            problem_id=pid,
            k=k,
            k_protocol=k_protocol,
            pass_bit=pass_at_k_bit(k),
            p_hat=shrinkage_p_hat(k, k_protocol),
            log_p_hat=log_p_hat(k, k_protocol),
            avg8=avg8,
            pass_at_k=unbiased_pass_at_k_curve(k_protocol, k),
        )
    return out


def mean_unbiased_pass_at_k(pool: Pool, k_protocol: int, k: int) -> float:
    """Mean of Chen unbiased pass@k over problems (each problem has n=k_protocol)."""
    stats = per_problem_stats(pool, k_protocol)
    return sum(s.pass_at_k[k] for s in stats.values()) / len(stats)  # type: ignore[index]


def mean_pass_at_k(pool: Pool, k_protocol: int) -> float:
    """Mean of legacy raw ``1[c≥1]`` bits. Prefer :func:`mean_unbiased_pass_at_k`."""
    stats = per_problem_stats(pool, k_protocol)
    return sum(s.pass_bit for s in stats.values()) / len(stats)


def per_problem_nc(pool: Pool) -> Dict[str, Dict[str, int]]:
    """Per-problem ``{\"n\": n, \"c\": c}`` for metrics writers."""
    out: Dict[str, Dict[str, int]] = {}
    for pid, verified in pool.items():
        out[pid] = {"n": len(verified), "c": _successes(verified)}
    return out


def avg_at_8(pool: Pool) -> float:
    """
    Mean of ``verified`` over sample_idx 0–7, then over problems.

    Equivalent to the global mean when every problem has those eight
    indices. Requires K ≥ 8; a shorter pool is not silently reused.
    """
    lengths = {len(v) for v in pool.values()}
    if lengths != {len(next(iter(pool.values())))}:
        raise StatsError("inconsistent K across problems")
    k_protocol = len(next(iter(pool.values())))
    if k_protocol < 8:
        raise StatsError(
            f"avg@8 needs sample_idx 0–7 of the pass@K pool; got K={k_protocol}"
        )
    stats = per_problem_stats(pool, k_protocol)
    return sum(s.avg8 for s in stats.values()) / len(stats)  # type: ignore[arg-type]


@dataclass(frozen=True)
class LearnedForgotten:
    learned: int
    forgotten: int
    n_base_unsolvable: int
    n_base_solvable: int
    n_problems: int

    @property
    def net(self) -> int:
        return self.learned - self.forgotten

    @property
    def learned_rate(self) -> Optional[float]:
        if self.n_base_unsolvable == 0:
            return None
        return self.learned / self.n_base_unsolvable

    @property
    def forgotten_rate(self) -> Optional[float]:
        if self.n_base_solvable == 0:
            return None
        return self.forgotten / self.n_base_solvable

    def as_dict(self) -> Dict[str, Any]:
        return {
            "learned": self.learned,
            "forgotten": self.forgotten,
            "net": self.net,
            "n_base_unsolvable": self.n_base_unsolvable,
            "n_base_solvable": self.n_base_solvable,
            "n_problems": self.n_problems,
            "learned_rate": self.learned_rate,
            "forgotten_rate": self.forgotten_rate,
        }


def learned_forgotten(
    base: Pool,
    ckpt: Pool,
    k_protocol: int,
    bins: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    ids = aligned_problem_ids(base, ckpt)
    base_s = per_problem_stats(base, k_protocol)
    ckpt_s = per_problem_stats(ckpt, k_protocol)

    def _for(subset: Sequence[str]) -> LearnedForgotten:
        learned = forgotten = unsolv = solv = 0
        for pid in subset:
            b = base_s[pid].pass_bit
            c = ckpt_s[pid].pass_bit
            if b == 0:
                unsolv += 1
            else:
                solv += 1
            if b == 0 and c == 1:
                learned += 1
            elif b == 1 and c == 0:
                forgotten += 1
        return LearnedForgotten(learned, forgotten, unsolv, solv, len(subset))

    overall = _for(ids)
    result: Dict[str, Any] = overall.as_dict()
    if bins is not None:
        unknown = [pid for pid in ids if pid not in bins]
        if unknown:
            raise StatsError(f"bin assignment missing for {unknown[:8]}")
        by_bin: Dict[str, Any] = {}
        for bin_id in sorted(set(bins[pid] for pid in ids)):
            subset = [pid for pid in ids if bins[pid] == bin_id]
            by_bin[bin_id] = _for(subset).as_dict()
        result["by_bin"] = by_bin
    return result


def per_problem_delta_log_p(
    base: Pool, ckpt: Pool, k_protocol: int
) -> Dict[str, float]:
    ids = aligned_problem_ids(base, ckpt)
    base_s = per_problem_stats(base, k_protocol)
    ckpt_s = per_problem_stats(ckpt, k_protocol)
    return {
        pid: delta_log_p_hat(base_s[pid].k, ckpt_s[pid].k, k_protocol)
        for pid in ids
    }


@dataclass(frozen=True)
class McNemarResult:
    learned: int
    forgotten: int
    chi2: Optional[float]
    p_chi2: Optional[float]
    p_exact: float
    n_discordant: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "learned": self.learned,
            "forgotten": self.forgotten,
            "n_discordant": self.n_discordant,
            "chi2": self.chi2,
            "p_chi2": self.p_chi2,
            "p_exact": self.p_exact,
        }


def _chi2_sf_1df(x: float) -> float:
    """Survival function of χ²(1): erfc(sqrt(x/2))."""
    if x < 0:
        raise StatsError(f"chi2 statistic must be >= 0, got {x}")
    return math.erfc(math.sqrt(x / 2.0))


def _binom_cdf_le(n: int, t: int) -> float:
    """P(X ≤ t) for X ~ Binomial(n, 1/2)."""
    if t < 0:
        return 0.0
    if t >= n:
        return 1.0
    return sum(math.comb(n, i) for i in range(t + 1)) / (2 ** n)


def mcnemar(base: Pool, ckpt: Pool, k_protocol: int) -> McNemarResult:
    lf = learned_forgotten(base, ckpt, k_protocol)
    b, c = lf["learned"], lf["forgotten"]
    n = b + c
    if n == 0:
        chi2: Optional[float] = None
        p_chi2: Optional[float] = None
        p_exact = 1.0
    else:
        chi2 = (b - c) ** 2 / n
        p_chi2 = _chi2_sf_1df(chi2)
        p_exact = min(1.0, 2.0 * _binom_cdf_le(n, min(b, c)))
    return McNemarResult(
        learned=b,
        forgotten=c,
        chi2=chi2,
        p_chi2=p_chi2,
        p_exact=p_exact,
        n_discordant=n,
    )


def percentile(values: Sequence[float], q: float) -> float:
    """Linear interpolation between nearest ranks (q in [0, 1])."""
    if not values:
        raise StatsError("percentile of empty sample")
    if not 0.0 <= q <= 1.0:
        raise StatsError(f"q must be in [0, 1], got {q}")
    xs = sorted(values)
    idx = (len(xs) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(xs[lo])
    w = idx - lo
    return float(xs[lo]) * (1.0 - w) + float(xs[hi]) * w


def percentile_ci(
    values: Sequence[float], alpha: float = 0.05
) -> Tuple[float, float]:
    return percentile(values, alpha / 2.0), percentile(values, 1.0 - alpha / 2.0)


def resample_verified(
    verified: Sequence[bool], rng: random.Random
) -> VerifiedSeq:
    k = len(verified)
    return tuple(verified[rng.randrange(k)] for _ in range(k))


def two_level_resample_pool(pool: Pool, rng: random.Random) -> Pool:
    """Resample problems, then resample each drawn problem's K completions."""
    pids = list(pool.keys())
    star: Pool = {}
    for draw_i in range(len(pids)):
        pid = pids[rng.randrange(len(pids))]
        # Distinct keys so a problem drawn twice is two independent copies.
        star[f"{pid}#{draw_i}"] = resample_verified(pool[pid], rng)
    return star


def two_level_resample_paired(
    base: Pool, ckpt: Pool, rng: random.Random
) -> Tuple[Pool, Pool]:
    ids = aligned_problem_ids(base, ckpt)
    base_star: Pool = {}
    ckpt_star: Pool = {}
    for draw_i in range(len(ids)):
        pid = ids[rng.randrange(len(ids))]
        key = f"{pid}#{draw_i}"
        base_star[key] = resample_verified(base[pid], rng)
        ckpt_star[key] = resample_verified(ckpt[pid], rng)
    return base_star, ckpt_star


def two_level_bootstrap(
    pool: Pool,
    statistic: Callable[[Pool], float],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    samples = [statistic(two_level_resample_pool(pool, rng)) for _ in range(n_boot)]
    lo, hi = percentile_ci(samples, alpha)
    return {
        "n_boot": n_boot,
        "seed": seed,
        "alpha": alpha,
        "mean": sum(samples) / len(samples),
        "ci_low": lo,
        "ci_high": hi,
        "samples": samples,
    }


def two_level_bootstrap_paired(
    base: Pool,
    ckpt: Pool,
    statistic: Callable[[Pool, Pool], float],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        b_star, c_star = two_level_resample_paired(base, ckpt, rng)
        samples.append(statistic(b_star, c_star))
    lo, hi = percentile_ci(samples, alpha)
    return {
        "n_boot": n_boot,
        "seed": seed,
        "alpha": alpha,
        "mean": sum(samples) / len(samples),
        "ci_low": lo,
        "ci_high": hi,
        "samples": samples,
    }


def mean_delta_log_p(base: Pool, ckpt: Pool) -> float:
    k_protocol = len(next(iter(base.values())))
    deltas = per_problem_delta_log_p(base, ckpt, k_protocol)
    return sum(deltas.values()) / len(deltas)


def net_migration(base: Pool, ckpt: Pool) -> float:
    k_protocol = len(next(iter(base.values())))
    return float(learned_forgotten(base, ckpt, k_protocol)["net"])
