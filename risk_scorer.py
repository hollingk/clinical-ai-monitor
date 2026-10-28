"""
risk_scorer.py: Risk detection for clinical AI coding decisions
DataConnect Conference Workshop, Lab 1

Four detection functions each check one kind of risk, and score_risk()
combines their flags into a GREEN / YELLOW / RED level for the dashboard.

To run the tests:
    pytest tests/test_risk_scorer.py -v
"""

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
import re
import pandas as pd

# ─── Risk Categories ─────────────────────────────────────────────────────────

# Flag labels used throughout the dashboard
FLAG_ICD_HALLUCINATION = "INVALID_ICD_CODE"
FLAG_PHI_DETECTED = "PHI_IN_RATIONALE"
FLAG_LOW_CONFIDENCE = "LOW_CONFIDENCE"
FLAG_HIGH_STAKES = "MANDATORY_REVIEW_CATEGORY"

# Flags that make a decision RED on their own; any other flag is YELLOW-level
RED_FLAGS = {FLAG_ICD_HALLUCINATION, FLAG_PHI_DETECTED}

# ICD categories that require mandatory human review regardless of confidence
HIGH_STAKES_CATEGORIES = {"mental_health", "oncology"}

# Confidence threshold below which a decision is flagged as low-confidence
CONFIDENCE_THRESHOLD = 0.70

# Path to the offline ICD-10 vocabulary
VOCAB_PATH = Path(__file__).parent / "data" / "icd10_sample.csv"


# ─── Result Type ─────────────────────────────────────────────────────────────

@dataclass
class RiskResult:
    """
    The output of score_risk(). Drives the dashboard color coding and queue logic.

    level:      "GREEN" | "YELLOW" | "RED"
    flags:      list of FLAG_* strings that were triggered
    confidence: the confidence value used in scoring (pass-through from input)
    """
    level: str = "GREEN"
    flags: list = field(default_factory=list)
    confidence: float = 1.0


# ─── Helper: load ICD vocabulary ─────────────────────────────────────────────

@cache
def _load_vocab() -> frozenset:
    """Return the set of valid ICD-10 codes from the offline vocabulary CSV (read once)."""
    df = pd.read_csv(VOCAB_PATH)
    return frozenset(df["code"].str.strip())


# ─── Helper: PHI analyzer ────────────────────────────────────────────────────

# Only these Presidio entity types count as PHI. Presidio's defaults also
# include US_DRIVER_LICENSE, DATE_TIME and LOCATION, which misfire on ICD
# codes ("I10"), clinical words ("Seasonal") and abbreviations ("STEMI").
PHI_ENTITIES = ["PERSON", "PHONE_NUMBER", "US_SSN", "EMAIL_ADDRESS", "MRN", "DOB"]

# Presidio results scoring below this are ignored as noise
PHI_SCORE_THRESHOLD = 0.5

_ICD_CODE_RE = re.compile(r"[A-Z]\d{2}(\.\d+)?")

_phi_analyzer = None


def _get_phi_analyzer():
    """Build the Presidio analyzer once (it's slow) and reuse it."""
    global _phi_analyzer
    if _phi_analyzer is None:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
        _phi_analyzer = AnalyzerEngine()
        # Presidio has no built-in MRN recognizer, so add one
        _phi_analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="MRN",
            patterns=[Pattern("mrn", r"\bMRN\s*[#:]?\s*\d{5,10}\b", 0.85)],
        ))
        # DATE_TIME is too noisy ("Seasonal", "3 days"), so only flag dates
        # labeled as a date of birth, e.g. "DOB 1971-04-22", "D.O.B.: 4/22/1971"
        # or "date of birth (4-22-1971)"
        _phi_analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="DOB",
            patterns=[Pattern(
                "dob",
                r"(?i)\b(?:D\.?O\.?B\.?|date of birth|born(?: on)?)\s*[:#]?\s*\(?\s*"
                r"(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
                r"|[A-Z][a-z]+\.? \d{1,2},? \d{4})",
                0.85,
            )],
        ))
    return _phi_analyzer


# ─── Detection Functions ─────────────────────────────────────────────────────
# Each function returns a list of flag strings (empty list = nothing found).

def check_icd_validity(code: str) -> list[str]:
    """
    Return [FLAG_ICD_HALLUCINATION] if `code` is not found in the ICD-10
    vocabulary CSV, otherwise return [].
    """
    if code not in _load_vocab():
        return [FLAG_ICD_HALLUCINATION]
    return []


def check_phi_leakage(rationale: str) -> list[str]:
    """
    Return [FLAG_PHI_DETECTED] if `rationale` appears to contain Protected
    Health Information (PHI), otherwise return [].

    Uses Microsoft Presidio, via _get_phi_analyzer() above, which adds MRN
    and date-of-birth recognizers to Presidio's built-in ones.
    """
    results = _get_phi_analyzer().analyze(
        text=rationale,
        language="en",
        entities=PHI_ENTITIES,
        score_threshold=PHI_SCORE_THRESHOLD,
    )
    # spaCy sometimes tags ICD codes (e.g. "N39.0") as PERSON; ignore those
    results = [
        r for r in results
        if not _ICD_CODE_RE.fullmatch(rationale[r.start:r.end])
    ]
    if results:
        return [FLAG_PHI_DETECTED]
    return []


def check_confidence(confidence: float) -> list[str]:
    """
    Return [FLAG_LOW_CONFIDENCE] if `confidence` is below CONFIDENCE_THRESHOLD,
    otherwise return [].
    """
    if confidence < CONFIDENCE_THRESHOLD:
        return [FLAG_LOW_CONFIDENCE]
    return []


def check_high_stakes_category(category: str) -> list[str]:
    """
    Return [FLAG_HIGH_STAKES] if `category` is in HIGH_STAKES_CATEGORIES,
    otherwise return [].
    """
    if category in HIGH_STAKES_CATEGORIES:
        return [FLAG_HIGH_STAKES]
    return []


# ─── Main Scorer ─────────────────────────────────────────────────────────────

def score_risk(decision: dict) -> RiskResult:
    """
    Run all four checks against a single AI decision and return a RiskResult.

    Risk level logic:
      - Any RED-triggering flag (PHI or invalid ICD code) → level = "RED"
      - Any YELLOW-triggering flag (low confidence or high-stakes category) → level = "YELLOW"
      - No flags → level = "GREEN"

    RED flags take priority over YELLOW. A decision with both a low confidence
    score AND a hallucinated code is RED.

    Args:
        decision: a dict with keys:
            suggested_code (str)   the ICD-10 code the AI assigned
            rationale      (str)   the AI's explanation
            confidence     (float) AI's self-reported confidence (0.0–1.0)
            category       (str)   ICD category for high-stakes checking

    Returns:
        RiskResult with level, flags, and confidence set.
    """
    confidence = decision.get("confidence", 1.0)
    flags = (
        check_icd_validity(decision.get("suggested_code", ""))
        + check_phi_leakage(decision.get("rationale", ""))
        + check_confidence(confidence)
        + check_high_stakes_category(decision.get("category", ""))
    )

    if RED_FLAGS.intersection(flags):
        level = "RED"
    elif flags:
        level = "YELLOW"
    else:
        level = "GREEN"

    return RiskResult(level=level, flags=flags, confidence=confidence)
