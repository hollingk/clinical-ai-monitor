"""
risk_scorer.py — Risk detection for clinical AI coding decisions
DataConnect Conference Workshop — Lab 1

Your task in Lab 1 is to implement the four detection functions below,
then combine them in score_risk().

When you're done, run the test suite to verify:
    pytest tests/test_risk_scorer.py -v

All 8 tests should pass.
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

    Hint: call _load_vocab() to get the set of valid codes.
    """
    # TODO (Lab 1): Implement this function.
    # Load the vocabulary with _load_vocab(), then check whether `code` is in it.
    # If the code is NOT found, return [FLAG_ICD_HALLUCINATION].
    # If it IS found, return [].
    return []  # stub — remove when implemented


def check_phi_leakage(rationale: str) -> list[str]:
    """
    Return [FLAG_PHI_DETECTED] if `rationale` appears to contain Protected
    Health Information (PHI), otherwise return [].

    Use Microsoft Presidio for PHI detection, via the pre-built helper above
    (it builds the analyzer once and adds MRN and date-of-birth recognizers):

        results = _get_phi_analyzer().analyze(
            text=rationale,
            language="en",
            entities=PHI_ENTITIES,
            score_threshold=PHI_SCORE_THRESHOLD,
        )
        # results is a list of RecognizerResult objects — check if it's non-empty

    Watch out: spaCy sometimes tags ICD codes like "N39.0" as a PERSON. Drop
    any result whose matched text, rationale[r.start:r.end], is an ICD code
    (_ICD_CODE_RE.fullmatch(...) tells you).

    Return [FLAG_PHI_DETECTED] if anything is left, otherwise return [].
    """
    # TODO (Lab 1): Implement this function.
    return []  # stub — remove when implemented


def check_confidence(confidence: float) -> list[str]:
    """
    Return [FLAG_LOW_CONFIDENCE] if `confidence` is below CONFIDENCE_THRESHOLD,
    otherwise return [].
    """
    # TODO (Lab 1): Implement this function.
    return []  # stub — remove when implemented


def check_high_stakes_category(category: str) -> list[str]:
    """
    Return [FLAG_HIGH_STAKES] if `category` is in HIGH_STAKES_CATEGORIES,
    otherwise return [].
    """
    # TODO (Lab 1): Implement this function.
    return []  # stub — remove when implemented


# ─── Main Scorer ─────────────────────────────────────────────────────────────

def score_risk(decision: dict) -> RiskResult:
    """
    Run all four checks against a single AI decision and return a RiskResult.

    Risk level logic:
      - Any RED-triggering flag (PHI or invalid ICD code) → level = "RED"
      - Any YELLOW-triggering flag (low confidence or high-stakes category) → level = "YELLOW"
      - No flags → level = "GREEN"

    RED flags take priority over YELLOW. A decision with both a low confidence
    score AND a hallucinated code should be RED.

    Args:
        decision: a dict with keys:
            suggested_code (str)   — the ICD-10 code the AI assigned
            rationale      (str)   — the AI's explanation
            confidence     (float) — AI's self-reported confidence (0.0–1.0)
            category       (str)   — ICD category for high-stakes checking

    Returns:
        RiskResult with level, flags, and confidence set.
    """
    # TODO (Lab 1): Implement this function.
    # 1. Call each of the four check_* functions.
    # 2. Combine their flag lists.
    # 3. Determine the level:
    #      RED flags:    FLAG_ICD_HALLUCINATION, FLAG_PHI_DETECTED
    #      YELLOW flags: FLAG_LOW_CONFIDENCE, FLAG_HIGH_STAKES
    # 4. Return a RiskResult.

    # Stub: returns GREEN for everything until you implement the checks.
    return RiskResult(level="GREEN", flags=[], confidence=decision.get("confidence", 1.0))
