import numpy as np
import pandas as pd
from model import predict_probability, act_to_sat

_RNG_SEED = 42
_POOL_SIZE = 10_000

_pool: pd.DataFrame | None = None


def _generate_pool() -> pd.DataFrame:
    rng = np.random.default_rng(_RNG_SEED)

    sat = np.clip(rng.normal(1450, 80, _POOL_SIZE), 400, 1600)
    gpa = np.clip(rng.normal(3.92, 0.08, _POOL_SIZE), 2.0, 4.0)

    ethnicities = ["White", "Asian", "Black", "Hispanic", "Other / Prefer not to say"]
    eth_probs = [0.40, 0.25, 0.15, 0.15, 0.05]
    ethnicity = rng.choice(ethnicities, size=_POOL_SIZE, p=eth_probs)

    legacy = rng.random(_POOL_SIZE) < 0.05
    athlete = rng.random(_POOL_SIZE) < 0.02

    probs = np.array([
        predict_probability(gpa[i], sat[i], ethnicity[i], legacy[i], athlete[i])
        for i in range(_POOL_SIZE)
    ])
    admitted = rng.random(_POOL_SIZE) < probs

    return pd.DataFrame({
        "sat": sat,
        "gpa": gpa,
        "ethnicity": ethnicity,
        "legacy": legacy,
        "athlete": athlete,
        "prob": probs,
        "admitted": admitted,
    })


def _get_pool() -> pd.DataFrame:
    global _pool
    if _pool is None:
        _pool = _generate_pool()
    return _pool


def pct_below_sat(sat_score: float) -> float:
    """Return the fraction of applicants with SAT <= sat_score (0–1)."""
    pool = _get_pool()
    return float((pool["sat"] <= sat_score).mean())


def pct_below_gpa(gpa: float) -> float:
    """Return the fraction of applicants with GPA <= gpa (0–1)."""
    pool = _get_pool()
    return float((pool["gpa"] <= gpa).mean())


def admitted_ethnicity_pct(ethnicity: str) -> float:
    """Return the fraction of admitted students who share this ethnicity (0–1)."""
    pool = _get_pool()
    admitted = pool[pool["admitted"]]
    if len(admitted) == 0:
        return 0.0
    return float((admitted["ethnicity"] == ethnicity).mean())


def similar_applicants_stats(
    sat_score: float, gpa: float, ethnicity: str, legacy: bool, athlete: bool,
) -> tuple[int, int]:
    pool = _get_pool()
    mask = (
        (pool["sat"] <= sat_score) &
        (pool["gpa"] <= gpa) &
        (pool["ethnicity"] == ethnicity)
    )
    if not legacy:
        mask &= ~pool["legacy"]
    if not athlete:
        mask &= ~pool["athlete"]
    similar = pool[mask]
    n_admitted_similar = int(similar["admitted"].sum())
    total_admitted = int(pool["admitted"].sum())
    n_similar_scaled = len(similar) * 5          # scale 10K pool → ~50K real applicants
    n_in_cohort = int(round((n_admitted_similar / total_admitted) * 2000)) if total_admitted else 0
    return n_similar_scaled, n_in_cohort
