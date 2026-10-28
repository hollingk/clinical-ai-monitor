"""
dashboard.py: Clinical AI Coding Monitor
DataConnect Conference Workshop Starter

Run with:
    streamlit run dashboard.py

Workshop labs completed in this file:
  Lab 2 (Hour 2): Find the TODO (Lab 2) comment in the Live Feed tab and add a risk-level filter.
  Lab 3 (Hour 3): Find the TODO (Lab 3) comments and implement the action buttons.
  Lab 4 (Hour 4): Find the TODO (Lab 4) comment and add governance charts.

The risk scorer (risk_scorer.py) is implemented separately in Lab 1.
Until Lab 1 is complete, every decision will appear GREEN.
"""

import os
import random
import time
import streamlit as st
import pandas as pd
from datetime import datetime

from dotenv import load_dotenv
from db import (
    init_db,
    log_action,
    get_audit_log,
    get_db_metrics,
    get_daily_flag_rates,
    save_correction,
    get_correction_count,
    seed_sample_data,
)
from mock_stream import MOCK_DECISIONS, get_icd_vocabulary
from risk_scorer import score_risk, RED_FLAGS, FLAG_ICD_HALLUCINATION
from rogue_agent import get_coding_decision, agent_status

load_dotenv()
_LIVE_AGENT_ENABLED = os.getenv("USE_LIVE_AGENT", "false").lower() == "true"

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Clinical AI Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Let selectbox dropdowns grow wider than their (narrow) column so long ICD-10
# descriptions aren't cut off. Rows can't wrap: Streamlit virtualizes long
# option lists with a fixed row height, so wrapped text would overlap.
st.markdown(
    """
    <style>
    [data-testid="stSelectboxVirtualDropdown"] {
        width: max(var(--trigger-width), min(760px, 90vw)) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Bootstrap ───────────────────────────────────────────────────────────────

init_db()
ICD_VOCAB = get_icd_vocabulary()
ICD_OPTIONS = [
    f"{row['code']}: {row['description']}"
    for _, row in ICD_VOCAB.iterrows()
]


def _related_codes(code: str) -> list[str]:
    """Valid vocabulary codes in the same 3-character ICD-10 family as `code`."""
    family = code.strip().upper()[:3]
    return [opt for opt in ICD_OPTIONS if opt.startswith(family)]

# ─── Session State ────────────────────────────────────────────────────────────

_DEFAULTS = {
    "decision_index": 0,   # pointer into MOCK_DECISIONS (mock mode only)
    "live_decisions": [],  # all decisions seen this session, newest first
    "review_queue": [],    # YELLOW / RED decisions awaiting human action
    "use_live_agent": False,
    "live_note_input": "",
    "run_live": False,       # auto-stream simulated decisions (mock mode only)
    "sim_count": 0,          # simulated decisions generated so far
    "last_arrival": 0.0,     # time.time() of the last simulated arrival
    "reviewed_count": 0,     # decisions a human has acted on this session
}

# Long sessions in Run Live mode can pile up hundreds of decisions; only render
# this many cards per tab so the page stays responsive
RENDER_LIMIT = 30
for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _ingest_decision(decision: dict) -> None:
    """Score a decision and route it into the feed and/or review queue."""
    result = score_risk(decision)
    decision["risk_level"] = result.level
    decision["risk_flags"] = result.flags
    decision["scored_confidence"] = result.confidence
    decision["ingested_at"] = time.time()  # for measuring how long it waits

    st.session_state.live_decisions.insert(0, decision)

    if result.level in ("YELLOW", "RED"):
        existing_ids = {d["decision_id"] for d in st.session_state.review_queue}
        if decision["decision_id"] not in existing_ids:
            st.session_state.review_queue.append(decision)


def _pull_next_decision() -> None:
    """Pull the next mock decision, score it, and add it to the feed."""
    idx = st.session_state.decision_index
    if idx >= len(MOCK_DECISIONS):
        st.sidebar.warning("All 20 mock decisions have been loaded.")
        return
    decision = MOCK_DECISIONS[idx].copy()
    _ingest_decision(decision)
    st.session_state.decision_index += 1


def _simulate_decision() -> None:
    """
    Generate one more decision for Run Live mode by replaying a random mock
    decision under a new ID, so the stream never runs out.
    """
    st.session_state.sim_count += 1
    decision = random.choice(MOCK_DECISIONS).copy()
    decision["decision_id"] = f"SIM-{st.session_state.sim_count:04d}"
    decision["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _ingest_decision(decision)


def _pull_live_decision(note: str, patient_context: str = "") -> None:
    """Submit a note to the rogue agent, score the result, and add to feed."""
    with st.spinner("Agent processing note..."):
        decision = get_coding_decision(note=note, patient_context=patient_context)
    # Keep what was sent so the card can show it next to what leaked out
    decision["patient_context"] = patient_context
    _ingest_decision(decision)


def _remove_from_queue(decision_id: str) -> None:
    st.session_state.reviewed_count += 1
    st.session_state.review_queue = [
        d for d in st.session_state.review_queue
        if d["decision_id"] != decision_id
    ]


def _render_review_panel(decision: dict, key_prefix: str) -> None:
    """
    Render a decision's details plus the reviewer action panel.

    Used by both the Live Feed and the Review Queue. `key_prefix` keeps widget
    keys unique, since the same decision can appear in both tabs at once.
    """
    did = decision["decision_id"]
    col_info, col_actions = st.columns([2, 1.2])

    # ── Left: note details and risk flags ──────────────────────
    with col_info:
        st.markdown("**AI Recommendation**")
        st.markdown(
            f"`{decision['suggested_code']}`: {decision['code_description']}  \n"
            f"Confidence: **{decision['confidence']:.0%}**"
        )
        if FLAG_ICD_HALLUCINATION in decision.get("risk_flags", []):
            related = _related_codes(decision["suggested_code"])
            st.caption(
                f"`{decision['suggested_code']}` is not a valid ICD-10 code, "
                "so this recommendation can't be approved as-is."
            )
            if related:
                st.caption(
                    f"Valid codes in the `{decision['suggested_code'][:3]}` family:  \n"
                    + "  \n".join(f"• {opt}" for opt in related)
                )
            else:
                st.caption(
                    f"No valid codes in the `{decision['suggested_code'][:3]}` "
                    "family. Choose one under ✏️ Edit & Approve."
                )

        st.markdown("**Physician Note Excerpt**")
        st.text(decision["note_excerpt"])
        if decision.get("patient_context"):
            st.markdown("**Patient Details Sent to Agent**")
            st.text(decision["patient_context"])
        st.markdown("**AI Rationale**")
        st.text(decision["rationale"])

        if decision.get("risk_flags"):
            for flag in decision["risk_flags"]:
                if flag in RED_FLAGS:
                    st.error(f"🚩 {flag}", icon="🔴")
                else:
                    st.warning(f"⚠️ {flag}", icon="🟡")
        else:
            st.caption("No risk flags.")

    # ── Right: reviewer action panel ───────────────────────────
    with col_actions:
        st.markdown("**Reviewer Action**")

        # Only decisions still in the queue can be acted on
        if not any(d["decision_id"] == did for d in st.session_state.review_queue):
            if decision.get("risk_level") == "GREEN":
                st.caption("🟢 No flags, so no human review is required.")
            else:
                st.caption("✅ Already reviewed.")
            return

        reviewer = st.text_input(
            "Your name",
            key=f"{key_prefix}_reviewer_{did}",
            placeholder="Enter your name",
        )

        action_choice = st.radio(
            "Select action",
            ["✅ Approve", "✏️ Edit & Approve", "❌ Reject & Escalate"],
            key=f"{key_prefix}_action_{did}",
            label_visibility="collapsed",
        )

        # ── ✅ Approve ─────────────────────────────────────────
        if action_choice == "✅ Approve":
            st.caption("Confirm the AI's code is correct.")
            if st.button("Submit Approval", key=f"{key_prefix}_btn_approve_{did}", use_container_width=True):
                if not reviewer.strip():
                    st.error("Please enter your name.")
                else:
                    # TODO (Lab 3): Uncomment the lines below to wire up the audit log.
                    # log_action(
                    #     decision_id=did,
                    #     reviewer=reviewer,
                    #     action="approved",
                    #     ai_suggested_code=decision["suggested_code"],
                    #     risk_level=decision.get("risk_level"),
                    #     risk_flags=decision.get("risk_flags", []),
                    #     queued_at=decision.get("timestamp"),
                    # )
                    _remove_from_queue(did)
                    st.success(f"✅ Approved. (Audit log write: TODO Lab 3)")
                    st.rerun()

        # ── ✏️ Edit & Approve ──────────────────────────────────
        elif action_choice == "✏️ Edit & Approve":
            st.caption("Select the correct ICD-10 code.")
            corrected_selection = st.selectbox(
                "Correct code",
                options=ICD_OPTIONS,
                key=f"{key_prefix}_corrected_code_{did}",
            )
            # The selected value is truncated inside the input; show it in full.
            st.caption(corrected_selection)
            correction_note = st.text_area(
                "Note (optional)",
                key=f"{key_prefix}_correction_note_{did}",
                height=68,
                placeholder="Reason for correction...",
            )
            if st.button("Submit Correction", key=f"{key_prefix}_btn_edit_{did}", use_container_width=True):
                if not reviewer.strip():
                    st.error("Please enter your name.")
                else:
                    corrected_code = corrected_selection.split(": ")[0]

                    # TODO (Lab 3): Uncomment the lines below to wire up the audit log
                    # and corrections store.
                    # log_action(
                    #     decision_id=did,
                    #     reviewer=reviewer,
                    #     action="corrected",
                    #     ai_suggested_code=decision["suggested_code"],
                    #     corrected_code=corrected_code,
                    #     note=correction_note,
                    #     risk_level=decision.get("risk_level"),
                    #     risk_flags=decision.get("risk_flags", []),
                    #     queued_at=decision.get("timestamp"),
                    # )
                    # save_correction(
                    #     decision_id=did,
                    #     ai_suggested_code=decision["suggested_code"],
                    #     corrected_code=corrected_code,
                    #     note=correction_note,
                    # )
                    _remove_from_queue(did)
                    st.success(
                        f"✏️ Corrected to `{corrected_code}`. "
                        "(Audit log + corrections store write: TODO Lab 3)"
                    )
                    st.rerun()

        # ── ❌ Reject & Escalate ───────────────────────────────
        elif action_choice == "❌ Reject & Escalate":
            st.caption("Block this decision from submission.")
            escalation_reason = st.text_area(
                "Escalation reason",
                key=f"{key_prefix}_escalation_{did}",
                height=68,
                placeholder="Describe why this decision should not proceed...",
            )
            if st.button(
                "Reject & Escalate",
                key=f"{key_prefix}_btn_reject_{did}",
                use_container_width=True,
                type="primary",
            ):
                if not reviewer.strip():
                    st.error("Please enter your name.")
                elif not escalation_reason.strip():
                    st.error("Please provide an escalation reason.")
                else:
                    # TODO (Lab 3): Uncomment the lines below.
                    # log_action(
                    #     decision_id=did,
                    #     reviewer=reviewer,
                    #     action="rejected",
                    #     ai_suggested_code=decision["suggested_code"],
                    #     escalation_reason=escalation_reason,
                    #     risk_level=decision.get("risk_level"),
                    #     risk_flags=decision.get("risk_flags", []),
                    #     queued_at=decision.get("timestamp"),
                    # )
                    _remove_from_queue(did)
                    st.error(
                        f"❌ Decision rejected and escalated. "
                        "(Audit log write: TODO Lab 3)"
                    )
                    st.rerun()

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── Mode selector ──────────────────────────────────────────────────────
    st.header("⚙️ Decision Feed")

    if _LIVE_AGENT_ENABLED:
        _status = agent_status()
        _backend_label = {
            "anthropic": f"Anthropic ({_status['model']})",
            "openai": f"OpenAI ({_status['model']})",
            "gemini": f"Gemini ({_status['model']})",
            "mock": "Mock mode (no API key)",
        }.get(_status["backend"], _status["backend"])
        st.session_state.use_live_agent = st.toggle(
            "🤖 Use Live Agent",
            value=st.session_state.use_live_agent,
            help=f"Current backend: {_backend_label}",
        )
    else:
        st.session_state.use_live_agent = False

    using_live = st.session_state.use_live_agent

    # ── Mock mode controls ─────────────────────────────────────────────────
    if not using_live:
        if st.button("▶ Pull Next Decision", use_container_width=True, type="primary"):
            _pull_next_decision()
            st.rerun()

        if st.button("▶▶ Pull All Remaining", use_container_width=True):
            while st.session_state.decision_index < len(MOCK_DECISIONS):
                _pull_next_decision()
            st.rerun()

        st.divider()
        loaded = st.session_state.decision_index
        total = len(MOCK_DECISIONS)
        st.progress(loaded / total, text=f"{loaded} / {total} mock decisions loaded")

        # ── Run Live: stream decisions faster than a human can review ──────
        st.divider()
        st.subheader("⚡ Run Live")
        st.session_state.run_live = st.toggle(
            "Stream decisions automatically",
            value=st.session_state.run_live,
            help=(
                "Replays the mock decisions under new SIM- IDs on a timer, "
                "so you can watch the review backlog grow."
            ),
        )
        live_interval = st.select_slider(
            "New decision every",
            options=[0.5, 1, 2, 3, 5, 10],
            value=2,
            format_func=lambda s: f"{s}s",
            key="live_interval",
        )
        st.caption(f"≈ {60 / live_interval:.0f} decisions per minute")

    # ── Live agent controls ────────────────────────────────────────────────
    else:
        # Run Live is mock-only, so it never spends API calls
        st.session_state.run_live = False
        st.caption("Submit a physician note to the live rogue agent.")

        live_note = st.text_area(
            "Physician note",
            key="live_note_input",
            height=150,
            placeholder=(
                "e.g. 68yo male, BP 195/118, severe headache. "
                "No end-organ damage on workup."
            ),
        )
        live_context = st.text_input(
            "Patient record details (name, DOB, MRN)",
            placeholder="e.g. Margaret O'Brien, DOB 4-22-1971, MRN #882741",
            help=(
                "Optional. Simulates the patient details a real coding system pulls "
                "from the health record."
            ),
        )
        if st.button("▶ Submit to Agent", use_container_width=True, type="primary"):
            if live_note.strip():
                _pull_live_decision(live_note.strip(), live_context.strip())
                st.rerun()
            else:
                st.warning("Please enter a physician note.")

        st.divider()
        st.caption(f"🤖 Live decisions pulled: **{sum(1 for d in st.session_state.live_decisions if d.get('decision_id', '').startswith('LIVE-'))}**")

    # ── Shared stats ───────────────────────────────────────────────────────
    st.divider()
    flagged = sum(
        1 for d in st.session_state.live_decisions
        if d.get("risk_level") in ("YELLOW", "RED")
    )
    st.caption(f"🟡🔴 Flagged this session: **{flagged}**")
    st.caption(f"📥 In review queue: **{len(st.session_state.review_queue)}**")

    if not _LIVE_AGENT_ENABLED:
        st.divider()
        st.caption(
            "💡 To enable the live agent, set `USE_LIVE_AGENT=true` "
            "and an API key in your `.env` file."
        )

    st.divider()
    if st.button("🔄 Reset Session", use_container_width=True):
        for key in _DEFAULTS:
            st.session_state[key] = _DEFAULTS[key]
        st.rerun()


# ─── Header ──────────────────────────────────────────────────────────────────

st.title("🏥 Clinical AI Coding Monitor")
st.caption("Human-on-the-Loop Dashboard · ICD-10 Assignment Review System")
st.info(
    "**How this works:** an AI reads each clinician's note and predicts the ICD-10 "
    "code for it. **Every code shown here comes from the AI.** Clinicians write "
    "notes, not codes. Your job as the reviewer is to confirm, correct, or reject "
    "those AI-suggested codes before they go to a payer.",
    icon="ℹ️",
)

# ─── Top Metrics ─────────────────────────────────────────────────────────────

db_metrics = get_db_metrics()

_total_decisions = len(st.session_state.live_decisions)

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Decisions This Session",
    _total_decisions,
)
c2.metric(
    "Flagged This Session",
    flagged,
    delta=f"{flagged / max(_total_decisions, 1):.0%} flag rate" if _total_decisions else None,
    delta_color="inverse",
)
c3.metric(
    "Historical Override Rate",
    f"{db_metrics['override_rate']:.0%}",
    help="Corrections + rejections / total reviews (from audit log)",
)
c4.metric(
    "Awaiting Review",
    len(st.session_state.review_queue),
    delta_color="inverse",
)

# Retraining alert
correction_count = get_correction_count()
if correction_count >= 50:
    st.warning(
        f"⚠️ **Retraining threshold reached:** {correction_count} corrections accumulated. "
        "Notify the ML team to schedule a fine-tuning run.",
        icon="🔔",
    )

# ─── Run Live ────────────────────────────────────────────────────────────────

if st.session_state.run_live:
    _interval = st.session_state.live_interval

    @st.fragment(run_every=_interval)
    def _live_tick() -> None:
        """
        Fires on a timer. When a full interval has passed since the last
        arrival, add a decision and rerun the whole app so every tab updates.
        The elapsed-time check stops the full rerun itself from adding another.
        """
        now = time.time()
        if now - st.session_state.last_arrival >= _interval:
            st.session_state.last_arrival = now
            _simulate_decision()
            st.rerun()

    _live_tick()

    _queue = st.session_state.review_queue
    _now = time.time()
    _oldest_wait = _now - min(d.get("ingested_at", _now) for d in _queue) if _queue else 0
    _banner = st.error if len(_queue) >= 10 else st.warning
    _banner(
        f"**🔴 LIVE**: a new AI decision every **{_interval}s** "
        f"(≈ {60 / _interval:.0f}/min). &nbsp; "
        f"**{len(_queue)}** waiting for review · "
        f"oldest waiting **{_oldest_wait:.0f}s** · "
        f"**{st.session_state.reviewed_count}** reviewed by you",
        icon="⏱️",
    )

st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────────────

# ── TODO (Lab 2): Add a risk-level filter here ────────────────────────────────
#
# Replace the stub line below with a real st.selectbox (or st.radio) that
# lets a reviewer filter the Live Feed and Review Queue to show All, GREEN,
# YELLOW, or RED only.
#
# Example:
#   risk_filter = st.selectbox(
#       "Filter by risk level",
#       options=["All", "GREEN", "YELLOW", "RED"],
#       index=0,
#   )
#
# Hints:
#   - Store the result in a local variable. It does NOT need to go in
#     session state (no need to persist across reruns).
#   - It sits above the tabs, so both tabs below can use it to filter
#     their decision lists.
#   - The risk level on each decision is in decision["risk_level"],
#     set by _ingest_decision() when the decision was first scored.
#   - "All" should show everything; any other value should filter by match.
#
risk_filter = "All"  # stub: replace this line with a st.selectbox call

# ─────────────────────────────────────────────────────────────────────────────

tab_feed, tab_queue, tab_governance = st.tabs(
    ["📋 Live Feed", "🔍 Review Queue", f"📊 Governance"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: LIVE FEED
# ══════════════════════════════════════════════════════════════════════════════

with tab_feed:
    if not st.session_state.live_decisions:
        st.info(
            "No decisions loaded yet. "
            "Use **▶ Pull Next Decision** in the sidebar to start the feed."
        )

    decisions_to_show = [
        d for d in st.session_state.live_decisions
        if risk_filter == "All" or d.get("risk_level") == risk_filter
    ]

    if risk_filter != "All" and not decisions_to_show and st.session_state.live_decisions:
        st.caption(f"No {risk_filter} decisions in the feed yet.")

    if len(decisions_to_show) > RENDER_LIMIT:
        st.caption(f"Showing the newest {RENDER_LIMIT} of {len(decisions_to_show)} decisions.")

    for decision in decisions_to_show[:RENDER_LIMIT]:
        risk = decision.get("risk_level", "GREEN")
        badge = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(risk, "⚪")
        border_color = {"GREEN": "green", "YELLOW": "orange", "RED": "red"}.get(risk, "gray")

        with st.container(border=True):
            col_badge, col_main, col_meta = st.columns([0.5, 6, 2])

            with col_badge:
                st.markdown(f"<h2 style='margin:0'>{badge}</h2>", unsafe_allow_html=True)

            with col_main:
                st.markdown(
                    f"**{decision['decision_id']}** &nbsp;·&nbsp; "
                    f"`{decision['suggested_code']}`: {decision['code_description']}"
                )
                st.caption(decision["note_excerpt"])
                if decision.get("patient_context"):
                    st.caption(f"🪪 Patient details sent to agent: {decision['patient_context']}")

                # Color each flag by its own severity (same as the review queue),
                # not by the decision's overall level
                red_flags = [f for f in decision.get("risk_flags", []) if f in RED_FLAGS]
                yellow_flags = [f for f in decision.get("risk_flags", []) if f not in RED_FLAGS]
                if red_flags:
                    st.error(" &nbsp;·&nbsp; ".join(f"🚩 `{f}`" for f in red_flags), icon="🔴")
                if yellow_flags:
                    st.warning(" &nbsp;·&nbsp; ".join(f"⚠️ `{f}`" for f in yellow_flags), icon="🟡")

            with col_meta:
                st.caption(f"🕐 {decision['timestamp']}")
                st.caption(
                    f"Confidence: **{decision['confidence']:.0%}**"
                )
                if any(d["decision_id"] == decision["decision_id"] for d in st.session_state.review_queue):
                    st.caption("📥 Awaiting review")
                elif risk in ("YELLOW", "RED"):
                    st.caption("✅ Reviewed")

            label = "🔍 Details & review" if risk in ("YELLOW", "RED") else "🔍 Details"
            with st.expander(label, expanded=False):
                _render_review_panel(decision, key_prefix="feed")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: REVIEW QUEUE
# ══════════════════════════════════════════════════════════════════════════════

with tab_queue:
    queue = st.session_state.review_queue

    if not queue:
        if st.session_state.live_decisions:
            st.success("✅ Review queue is empty. All flagged decisions have been processed.")
        else:
            st.info("Pull some decisions from the sidebar. Flagged ones will appear here.")
    else:
        st.write(f"**{len(queue)} decision(s) pending human review**")

        queue_to_show = [
            d for d in queue
            if risk_filter == "All" or d.get("risk_level") == risk_filter
        ]
        if risk_filter == "GREEN":
            st.caption("GREEN decisions don't need human review, so none are queued.")
        elif risk_filter != "All" and not queue_to_show:
            st.caption(f"No {risk_filter} decisions awaiting review.")

        if len(queue_to_show) > RENDER_LIMIT:
            st.caption(
                f"Showing the oldest {RENDER_LIMIT}; "
                f"{len(queue_to_show) - RENDER_LIMIT} more are waiting behind them."
            )

        for decision in queue_to_show[:RENDER_LIMIT]:  # slicing copies, so removals during loop are safe
            risk = decision.get("risk_level", "YELLOW")
            badge = {"YELLOW": "🟡", "RED": "🔴"}.get(risk, "🟡")
            did = decision["decision_id"]

            with st.expander(
                f"{badge} **{did}** · `{decision['suggested_code']}`: {decision['code_description']}",
                expanded=False,
            ):
                _render_review_panel(decision, key_prefix="queue")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: GOVERNANCE
# ══════════════════════════════════════════════════════════════════════════════

with tab_governance:
    db_m = get_db_metrics()

    # Summary KPIs
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Total Reviewed (All Time)", db_m["total_reviewed"])
    g2.metric("Approved", db_m["approved"])
    g3.metric("Corrected", db_m["corrected"])
    g4.metric("Rejected", db_m["rejected"])

    st.divider()

    # ── TODO (Lab 4): Add governance charts below ─────────────────────────────
    #
    # 0. First, load 30 days of sample review history so the charts have
    #    something to show. Sample rows have decision IDs starting with HIST-;
    #    your own reviews from Lab 3 are kept alongside them.
    #    seed_sample_data()
    #    db_m = get_db_metrics()  # refresh the KPIs above to include the samples
    #
    # Suggested charts (all data is available from db.py):
    #
    # 1. Override rate trend (drift signal)
    #    daily_df = get_daily_flag_rates()
    #    if daily_df.empty:
    #        st.info("No reviews logged yet. Log some in Lab 3, or run step 0 above.")
    #    else:
    #        st.line_chart(daily_df.set_index("date")["override_rate"])
    #
    # 2. Action breakdown bar chart
    #    action_counts = pd.DataFrame({
    #        "Action": ["Approved", "Corrected", "Rejected"],
    #        "Count":  [db_m["approved"], db_m["corrected"], db_m["rejected"]],
    #    })
    #    st.bar_chart(action_counts.set_index("Action"))
    #
    # 3. Full audit log table
    #    audit_df = get_audit_log()
    #    st.dataframe(audit_df, use_container_width=True)
    #
    # 4. Reviewer consistency: pivot table of reviewer × action
    #    pivot = audit_df.pivot_table(
    #        index="reviewer", columns="action", values="id",
    #        aggfunc="count", fill_value=0
    #    )
    #    st.dataframe(pivot, use_container_width=True)

    st.info(
        "💡 **Lab 4:** Add governance charts here using the commented-out snippets above.\n\n"
        "Data is already available from `get_daily_flag_rates()` and `get_audit_log()` in `db.py`."
    )

    st.divider()
    st.caption(f"Corrections store: **{get_correction_count()}** labels accumulated "
               f"(retraining threshold: 50)")
