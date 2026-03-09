import streamlit as st
from model import predict_probability, act_to_sat
from data import pct_below_sat, pct_below_gpa, admitted_ethnicity_pct

st.set_page_config(
    page_title="Harvard Admissions Calculator",
    page_icon="🎓",
    layout="centered",
)

st.markdown(
    """
    <style>
        /* Global text color */
        body, .stApp { color: #333333; }

        /* Probability display */
        .prob-display {
            text-align: center;
            padding: 24px 0 8px 0;
        }
        .prob-number {
            font-size: 88px;
            font-weight: 700;
            color: #A51C30;
            line-height: 1;
        }
        .prob-label {
            font-size: 15px;
            color: #666666;
            margin-top: 6px;
        }

        /* Benchmark cards */
        .bench-card {
            background: #f8f8f8;
            border-left: 4px solid #A51C30;
            border-radius: 6px;
            padding: 14px 18px;
            margin-bottom: 12px;
        }
        .bench-stat {
            font-size: 26px;
            font-weight: 700;
            color: #A51C30;
        }
        .bench-label {
            font-size: 13px;
            color: #555555;
            margin-top: 2px;
        }

        /* Disclaimer */
        .disclaimer {
            font-size: 12px;
            color: #999999;
            text-align: center;
            padding: 12px 0 4px 0;
        }

        /* Tighten up slider labels */
        .stSlider label { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("# Harvard Admissions Calculator")
st.markdown("*Adjust your profile to see how your chances change.*")
st.divider()

# ── Inputs ───────────────────────────────────────────────────────────────────
gpa = st.slider("Your GPA (unweighted, 0–4.0)", 0.0, 4.0, 3.7, step=0.05)

test_type = st.radio("Test score type", ["SAT", "ACT"], horizontal=True)

if test_type == "SAT":
    sat_raw = st.slider("SAT Score", 400, 1600, 1450, step=10)
    sat_score = float(sat_raw)
else:
    act_raw = st.slider("ACT Score", 1, 36, 34)
    sat_score = act_to_sat(float(act_raw))

ethnicity = st.radio(
    "Ethnicity",
    ["White", "Asian", "Black", "Hispanic", "Other / Prefer not to say"],
)

legacy = st.checkbox("I am the child of a Harvard alumnus/alumna")
athlete = st.checkbox("I am on a coach's recruiting list")

st.divider()

# ── Compute ───────────────────────────────────────────────────────────────────
prob = predict_probability(gpa, sat_score, ethnicity, legacy, athlete)
prob_pct = prob * 100

sat_pct = pct_below_sat(sat_score) * 100
gpa_pct = pct_below_gpa(gpa) * 100
eth_admitted_pct = admitted_ethnicity_pct(ethnicity) * 100

# ── Probability Display ───────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="prob-display">
        <div class="prob-number">{prob_pct:.1f}%</div>
        <div class="prob-label">Estimated probability of admission</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── Benchmark Stats ───────────────────────────────────────────────────────────
if test_type == "SAT":
    score_label = f"SAT {int(sat_score)}"
else:
    score_label = f"ACT {act_raw} (≈ SAT {int(sat_score)})"

st.markdown(
    f"""
    <div class="bench-card">
        <div class="bench-stat">{sat_pct:.0f}%</div>
        <div class="bench-label">
            <strong>Applicants with the same or lower {test_type} score</strong><br>
            Your {score_label} is higher than {sat_pct:.0f}% of Harvard applicants in our model.
        </div>
    </div>

    <div class="bench-card">
        <div class="bench-stat">{gpa_pct:.0f}%</div>
        <div class="bench-label">
            <strong>Applicants with the same or lower GPA</strong><br>
            Your GPA of {gpa:.2f} is higher than {gpa_pct:.0f}% of Harvard applicants in our model.
        </div>
    </div>

    <div class="bench-card">
        <div class="bench-stat">{eth_admitted_pct:.0f}%</div>
        <div class="bench-label">
            <strong>Admitted students sharing your ethnicity</strong><br>
            {eth_admitted_pct:.0f}% of admitted students in our model identify as {ethnicity}.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="disclaimer">
        This calculator uses a simplified illustrative model for educational purposes.
        It is not affiliated with Harvard University.
    </div>
    """,
    unsafe_allow_html=True,
)
