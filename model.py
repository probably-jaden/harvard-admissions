import numpy as np


# Dummy coefficients based on NBER findings
INTERCEPT = -14.0
COEF_GPA = 1.8
COEF_SAT_NORM = 6.0
COEF_ETHNICITY = {
    "White": 0.0,
    "Asian": -0.3,
    "Black": 0.8,
    "Hispanic": 0.5,
    "Other / Prefer not to say": 0.0,
}
COEF_LEGACY = 1.4
COEF_ATHLETE = 2.5

SAT_MIN = 400
SAT_MAX = 1600
ACT_MIN = 1
ACT_MAX = 36


def act_to_sat(act_score: float) -> float:
    """Linear concordance conversion from ACT to SAT score."""
    return (act_score - ACT_MIN) * (SAT_MAX - SAT_MIN) / (ACT_MAX - ACT_MIN) + SAT_MIN


def normalize_sat(sat_score: float) -> float:
    """Normalize SAT score to [0, 1] range."""
    return (sat_score - SAT_MIN) / (SAT_MAX - SAT_MIN)


def sigmoid(x: float) -> float:
    """Standard logistic sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def predict_probability(
    gpa: float,
    sat_score: float,
    ethnicity: str,
    legacy: bool,
    athlete: bool,
) -> float:
    """
    Predict Harvard admissions probability using logistic regression
    with dummy coefficients.

    Args:
        gpa: Unweighted GPA on 0.0–4.0 scale
        sat_score: SAT score (400–1600); convert ACT scores first with act_to_sat()
        ethnicity: One of the keys in COEF_ETHNICITY
        legacy: True if applicant is child of Harvard alumnus/alumna
        athlete: True if applicant is on a coach's recruiting list

    Returns:
        Probability of admission as a float in [0, 1]
    """
    sat_norm = normalize_sat(sat_score)
    eth_coef = COEF_ETHNICITY.get(ethnicity, 0.0)

    logit = (
        INTERCEPT
        + COEF_GPA * gpa
        + COEF_SAT_NORM * sat_norm
        + eth_coef
        + COEF_LEGACY * int(legacy)
        + COEF_ATHLETE * int(athlete)
    )

    return float(sigmoid(logit))
