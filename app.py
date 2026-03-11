import random
import numpy as np
import streamlit as st
from model import predict_probability, act_to_sat
from data import similar_applicants_stats

# ── Flavor messages by outcome category ──────────────────────────────────────
_ACCEPTED_LUCKY = [   # prob < 10 %
    "Wait… WHAT?! Harvard just pulled a total wildcard and said YES! 🎉",
    "Against all odds, the crimson gates swing open. Pack those Harvard hoodies! 🔴",
    "Plot twist of the century: YOU'RE IN! The admissions office is probably questioning themselves 😂",
    "Somehow, someway — Harvard said yes when literally no one expected it. Legendary. 🌟",
    "Legend has it the admissions officer gasped, shrugged, and stamped ACCEPTED 👀",
    "The admissions lottery smiled on you today. Buy a lotto ticket too, honestly 🍀",
]
_ACCEPTED_DECENT = [  # prob 10 – 50 %
    "Harvard saw your hustle and said YES! Now brace for the $90k/year price tag 💸",
    "A long shot that paid off! The crimson gates are officially open for you 🏛️",
    "You did it! Harvard said 'we'll take the chance' — and you delivered! 🎓",
    "The gamble paid off. Welcome to the club nobody can actually afford 😏",
    "Harvard took one look at your profile and said 'interesting, let's see what you've got' ✅",
    "Solid underdog story. Hollywood will be calling. 🎬",
]
_ACCEPTED_LIKELY = [  # prob > 50 %
    "Obviously. Harvard would've been foolish not to 👑",
    "No surprises here — the smart money was always on you 🎓",
    "Harvard saw your profile and said 'yes please' before you even finished clicking ✨",
    "With those stats? It'd be more shocking if they'd said no 🙄 (congrats though!)",
    "You walked in with a résumé and left with a crimson sweater. Expected, honestly 💅",
    "Checks out. Harvard just locked in their star admit for the year 🏆",
]
_REJECTED_UNLUCKY = [  # prob > 50 % but rejected
    "Ouch. You had better-than-even odds and Harvard still ghosted you. That's cruel 😭",
    "On any other day… but today just wasn't your day. Maybe Yale will appreciate you 😢",
    "You were statistically more likely to get in than not. Coin-flip life 🪙",
    "Harvard said 'it's not you, it's us.' It is definitely them 😤",
    "You brought a near-perfect profile and Harvard said… no? Absolutely rude 🙃",
    "Sometimes the universe just wants to humble you. Harvard was its instrument today 😔",
]
_REJECTED_MEDIUM = [  # prob 5 – 50 %
    "So close! If only you had that résumé candy of being the National Honor Society VP 📋",
    "Harvard said thanks but no thanks. Have you considered… community college? 😅",
    "Your essay about overcoming adversity wasn't quite adversity enough 📝",
    "You were in the conversation! Just… the wrong side of it 😬",
    "The admissions committee read your profile and whispered 'interesting… pass' 🤔",
    "So near yet so far. Try founding a nonprofit next time — they eat that up 🌍",
    "Harvard saw promise but decided to let someone else nurture it. Their loss tbh 🤷",
]
_REJECTED_LOW = [     # prob < 5 %
    "It was always a long shot, but respect for swinging big 🙌",
    "Harvard laughed, but so did everyone else who applied from a public school 😭 (jk, kind of)",
    "Don't worry — Zuckerberg dropped out anyway 💻",
    "You aimed for the moon and missed, but stars are pretty too ✨ (have you tried UMass?)",
    "The admissions office blinked at your file and gently placed it in the 'no' pile 🗂️",
    "A+ for effort, F for… well, you know. Harvard says hi from very far away 👋",
    "Applying was free and character-building! That's basically a Harvard education 😌",
]


def _pick_message(result: int, prob_pct: float) -> tuple[str, str]:
    """Return (message, alert_type) where alert_type is 'success' or 'error'."""
    if result == 1:
        if prob_pct < 10:
            body = random.choice(_ACCEPTED_LUCKY)
        elif prob_pct <= 50:
            body = random.choice(_ACCEPTED_DECENT)
        else:
            body = random.choice(_ACCEPTED_LIKELY)
        return f"🎉 **Accepted!** {body}", "success"
    else:
        if prob_pct > 50:
            body = random.choice(_REJECTED_UNLUCKY)
        elif prob_pct >= 5:
            body = random.choice(_REJECTED_MEDIUM)
        else:
            body = random.choice(_REJECTED_LOW)
        return f"❌ **Rejected!** {body}", "error"

st.set_page_config(
    page_title="Harvard Admissions Calculator",
    page_icon="🎓",
    layout="centered",
)

st.markdown(
    """
    <style>
        /* Probability display */
        .prob-display {
            text-align: center;
            padding: 24px 0 8px 0;
        }
        .prob-number {
            font-size: 110px;
            font-weight: 700;
            color: #C8102E;
            line-height: 1;
        }
        .prob-label {
            font-size: 15px;
            color: #888888;
            margin-top: 6px;
        }

        /* Benchmark cards — neutral bg works on both light and dark */
        .bench-card {
            background: rgba(128, 128, 128, 0.12);
            border-left: 4px solid #C8102E;
            border-radius: 6px;
            padding: 14px 18px;
            margin-bottom: 12px;
        }
        .bench-stat {
            font-size: 26px;
            font-weight: 700;
            color: #C8102E;
        }
        .bench-label {
            font-size: 13px;
            color: #aaaaaa;
            margin-top: 2px;
        }
        .bench-sublabel {
            font-size: 12px;
            color: #888888;
            font-weight: normal;
        }

        /* Disclaimer */
        .disclaimer {
            font-size: 12px;
            color: #888888;
            text-align: center;
            padding: 12px 0 4px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialization ─────────────────────────────────────────────
defaults = {
    'sat_val': 1300, 'act_val': 28, 'gpa_val': 3.5,
    'test_type_radio': 'SAT', 'ethnicity_radio': 'White',
    'legacy_check': False, 'athlete_check': False,
    'admission_result': None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Harvard Admissions Calculator")
st.markdown("*Adjust your profile to see how your chances change.*")
st.divider()

# ── Compute probability (before widgets) ─────────────────────────────────────
_test_type = st.session_state['test_type_radio']
_sat_score = float(st.session_state['sat_val']) if _test_type == 'SAT' \
             else act_to_sat(float(st.session_state['act_val']))
prob = predict_probability(
    gpa=float(st.session_state['gpa_val']),
    sat_score=_sat_score,
    ethnicity=st.session_state['ethnicity_radio'],
    legacy=bool(st.session_state['legacy_check']),
    athlete=bool(st.session_state['athlete_check']),
)
prob_pct = prob * 100

# ── Probability display ───────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="prob-display">
        <div class="prob-number">{prob_pct:.1f}%</div>
        <div class="prob-label">Estimated probability of admission</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Apply to Harvard button ───────────────────────────────────────────────────
col_btn, col_msg = st.columns([1, 3])
with col_btn:
    if st.button("Apply to Harvard"):
        result = int(np.random.binomial(1, prob))
        st.session_state['admission_result'] = (result, prob_pct)
with col_msg:
    if st.session_state['admission_result'] is not None:
        result, stored_prob = st.session_state['admission_result']
        msg, alert_type = _pick_message(result, stored_prob)
        if alert_type == "success":
            st.success(msg)
        else:
            st.error(msg)

st.divider()

# ── Button callbacks ──────────────────────────────────────────────────────────
def _sat_minus(): st.session_state['sat_val'] = max(400, st.session_state['sat_val'] - 30)
def _sat_plus():  st.session_state['sat_val'] = min(1600, st.session_state['sat_val'] + 30)
def _act_minus(): st.session_state['act_val'] = max(1, st.session_state['act_val'] - 1)
def _act_plus():  st.session_state['act_val'] = min(36, st.session_state['act_val'] + 1)
def _gpa_minus(): st.session_state['gpa_val'] = round(max(0.0, st.session_state['gpa_val'] - 0.04), 10)
def _gpa_plus():  st.session_state['gpa_val'] = round(min(4.0, st.session_state['gpa_val'] + 0.04), 10)

# ── Input widgets ─────────────────────────────────────────────────────────────
st.radio("Test score type", ["SAT", "ACT"], horizontal=True, key='test_type_radio')

if st.session_state['test_type_radio'] == 'SAT':
    c_minus, c_input, c_plus = st.columns([1, 2, 1])
    with c_minus:
        st.button("−", key="sat_minus", on_click=_sat_minus)
        st.caption("study for 10 less hours")
    with c_input:
        st.number_input("SAT Score", 400, 1600, step=30, key='sat_val')
    with c_plus:
        st.button("+", key="sat_plus", on_click=_sat_plus)
        st.caption("study for 10 more hours")
else:
    c_minus, c_input, c_plus = st.columns([1, 2, 1])
    with c_minus:
        st.button("−", key="act_minus", on_click=_act_minus)
        st.caption("study for 10 less hours")
    with c_input:
        st.number_input("ACT Score", 1, 36, step=1, key='act_val')
    with c_plus:
        st.button("+", key="act_plus", on_click=_act_plus)
        st.caption("study for 10 more hours")

c_minus, c_input, c_plus = st.columns([1, 2, 1])
with c_minus:
    st.button("−", key="gpa_minus", on_click=_gpa_minus)
    st.caption("go down a grade in 1 class")
with c_input:
    st.number_input("GPA (unweighted, 0–4.0)", 0.0, 4.0, step=0.15, key='gpa_val')
with c_plus:
    st.button("+", key="gpa_plus", on_click=_gpa_plus)
    st.caption("go up a grade in 1 class")

st.radio(
    "Ethnicity",
    ["White", "Asian", "Black", "Hispanic", "Other / Prefer not to say"],
    key='ethnicity_radio',
)

st.checkbox("I am the child of a Harvard alumnus/alumna", key='legacy_check')
st.checkbox("I am on a coach's recruiting list", key='athlete_check')

st.divider()

# ── Benchmark cards ───────────────────────────────────────────────────────────
n_similar_scaled, n_in_cohort = similar_applicants_stats(
    sat_score=_sat_score,
    gpa=float(st.session_state['gpa_val']),
    ethnicity=st.session_state['ethnicity_radio'],
    legacy=bool(st.session_state['legacy_check']),
    athlete=bool(st.session_state['athlete_check']),
)

st.markdown(
    f"""
    <div class="bench-card">
        <div class="bench-stat">{n_similar_scaled:,}</div>
        <div class="bench-label">
            <strong>Fellow applicants with a similar or weaker profile</strong>
        </div>
    </div>

    <div class="bench-card">
        <div class="bench-stat">{n_in_cohort}</div>
        <div class="bench-label">
            <strong>Spots in the class for applicants like you</strong>
            <span class="bench-sublabel"> &nbsp;out of 2,000 total seats</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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
