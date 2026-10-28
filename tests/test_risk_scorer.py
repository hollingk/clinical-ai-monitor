"""
tests/test_risk_scorer.py: Lab 1 specification
DataConnect Conference Workshop

These 8 tests describe the correct behavior of risk_scorer.py.
On the stub (before Lab 1), all tests will FAIL.
After you implement the four check_* functions and score_risk(), all tests should PASS.

Run with:
    pytest tests/test_risk_scorer.py -v
"""

import pytest
from risk_scorer import (
    score_risk,
    check_icd_validity,
    check_phi_leakage,
    check_confidence,
    check_high_stakes_category,
    FLAG_ICD_HALLUCINATION,
    FLAG_PHI_DETECTED,
    FLAG_LOW_CONFIDENCE,
    FLAG_HIGH_STAKES,
    CONFIDENCE_THRESHOLD,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _decision(
    code="I10",
    rationale="Routine hypertension management visit.",
    confidence=0.95,
    category="cardiovascular",
) -> dict:
    """Build a minimal mock decision dict."""
    return {
        "decision_id": "TEST-001",
        "suggested_code": code,
        "rationale": rationale,
        "confidence": confidence,
        "category": category,
        "note_excerpt": "Test note.",
        "code_description": "Test code description.",
        "timestamp": "2026-06-11 09:00:00",
    }


# ─── Test 1: Valid ICD code passes ────────────────────────────────────────────

def test_valid_code_passes():
    """A code that exists in the ICD-10 vocabulary should NOT trigger FLAG_ICD_HALLUCINATION."""
    flags = check_icd_validity("I10")
    assert FLAG_ICD_HALLUCINATION not in flags, (
        "I10 is a real ICD-10 code and should not be flagged as a hallucination."
    )


# ─── Test 2: Hallucinated code is flagged ─────────────────────────────────────

def test_invalid_code_flagged():
    """A made-up ICD code should trigger FLAG_ICD_HALLUCINATION."""
    flags = check_icd_validity("I16.9")  # I16.9 does not exist; I16.0 and I16.1 do
    assert FLAG_ICD_HALLUCINATION in flags, (
        "I16.9 is not a real ICD-10 code and should be flagged as a hallucination."
    )


# ─── Test 3: Clean rationale passes PHI check ─────────────────────────────────

def test_clean_rationale_no_phi():
    """A rationale with no patient identifiers should NOT trigger FLAG_PHI_DETECTED."""
    flags = check_phi_leakage(
        "Routine hypertension management. BP well-controlled on current regimen."
    )
    assert FLAG_PHI_DETECTED not in flags, (
        "A clean rationale should not be flagged for PHI."
    )


# ─── Test 4: Rationale with a name is flagged ─────────────────────────────────

def test_phi_name_detected():
    """A rationale containing a patient name should trigger FLAG_PHI_DETECTED."""
    flags = check_phi_leakage(
        "Patient Margaret O'Brien presents with classic UTI symptoms."
    )
    assert FLAG_PHI_DETECTED in flags, (
        "The rationale contains a patient name and should be flagged for PHI."
    )


# ─── Test 5: Rationale with MRN is flagged ────────────────────────────────────

def test_phi_mrn_detected():
    """A rationale containing an MRN pattern should trigger FLAG_PHI_DETECTED."""
    flags = check_phi_leakage(
        "For patient MRN #882741, UTI coding is appropriate."
    )
    assert FLAG_PHI_DETECTED in flags, (
        "The rationale contains an MRN and should be flagged for PHI."
    )


# ─── Test 6: Low confidence is flagged ───────────────────────────────────────

def test_low_confidence_flagged():
    """Confidence below CONFIDENCE_THRESHOLD should trigger FLAG_LOW_CONFIDENCE."""
    low_conf = CONFIDENCE_THRESHOLD - 0.05
    flags = check_confidence(low_conf)
    assert FLAG_LOW_CONFIDENCE in flags, (
        f"Confidence {low_conf:.2f} is below threshold {CONFIDENCE_THRESHOLD} "
        "and should be flagged."
    )


# ─── Test 7: High-stakes category is flagged ─────────────────────────────────

def test_high_stakes_category_flagged():
    """A mental_health or oncology category should trigger FLAG_HIGH_STAKES."""
    flags = check_high_stakes_category("mental_health")
    assert FLAG_HIGH_STAKES in flags, (
        "mental_health is a high-stakes category and should trigger FLAG_HIGH_STAKES."
    )


# ─── Test 8: Multiple flags escalate to RED ───────────────────────────────────

def test_phi_and_hallucination_returns_red():
    """
    A decision with both PHI in the rationale AND a hallucinated code
    should return risk level RED (not just YELLOW).
    """
    decision = _decision(
        code="I16.9",  # hallucinated
        rationale=(
            "Patient Robert Chen (SSN: 562-87-4391) presents with hypertensive crisis."
        ),
        confidence=0.82,
        category="cardiovascular",
    )
    result = score_risk(decision)
    assert result.level == "RED", (
        "A decision with PHI and an invalid ICD code should be rated RED."
    )
    assert FLAG_ICD_HALLUCINATION in result.flags
    assert FLAG_PHI_DETECTED in result.flags
