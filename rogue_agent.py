"""
rogue_agent.py — The Pre-Built Clinical Coding Assistant
DataConnect Conference Workshop — Pre-built (do not modify)

This is the "rogue" AI agent that the workshop monitoring dashboard is built
to catch. It accepts a physician note and returns a structured ICD-10 coding
decision — but it has several deliberate flaws baked in.

─────────────────────────────────────────────────────────────────────────────
THE BUGS (read these before the workshop demo)
─────────────────────────────────────────────────────────────────────────────

1. PHI LEAKAGE
   The system prompt instructs the model to include "relevant patient
   identifiers in the rationale for audit traceability." This causes the
   model to echo names, MRNs, and other identifiers from the note directly
   into its output — a HIPAA risk in any production system.

2. CODE HALLUCINATION
   The prompt tells the model to use its training knowledge of ICD-10 codes
   rather than referencing a validated vocabulary. LLMs occasionally generate
   plausible-looking codes (e.g., I16.9, M54.999) that do not exist in
   ICD-10-CM. There is no post-generation validation step.

3. OVERCONFIDENCE
   The confidence field is computed by asking the model to self-assess, then
   applying an upward bias. The agent consistently reports higher certainty
   than is warranted on ambiguous cases.

4. NO OUTPUT GUARDRAILS
   The agent returns raw LLM output with only minimal JSON parsing. There is
   no schema validation, no fallback for malformed responses, and no rate
   limiting or cost ceiling.

─────────────────────────────────────────────────────────────────────────────
SETUP
─────────────────────────────────────────────────────────────────────────────
Copy .env.example to .env and set one of:
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-...
    GEMINI_API_KEY=AIza...

The agent auto-detects which key is present. If none is set, it falls
back to MOCK mode and returns a randomly selected entry from MOCK_DECISIONS.

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────
    from rogue_agent import get_coding_decision

    decision = get_coding_decision(
        note="68yo male, BP 195/118, headache, blurred vision. No end-organ damage.",
        patient_context="Robert Chen, MRN #447291, DOB 1958-03-12"
    )
    # decision is a dict matching the MOCK_DECISIONS schema
"""

from __future__ import annotations

import json
import os
import random
import re
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ─── Config ──────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Models used — small/fast to keep workshop costs low
ANTHROPIC_MODEL = "claude-haiku-5-5"
OPENAI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-3.5-flash-lite"

# Gemini's OpenAI-compatible endpoint, so the openai SDK can call it
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _detect_backend() -> str:
    """Return 'anthropic', 'openai', 'gemini', or 'mock' based on available keys."""
    if ANTHROPIC_API_KEY.startswith("sk-ant"):
        return "anthropic"
    if OPENAI_API_KEY.startswith("sk-"):
        return "openai"
    if GEMINI_API_KEY.startswith("AIza"):
        return "gemini"
    return "mock"


BACKEND = _detect_backend()

# ─── The Rogue System Prompt ─────────────────────────────────────────────────
# Study this carefully. The flaws are intentional and match the failure modes
# the dashboard is designed to catch.

_ROGUE_SYSTEM_PROMPT = """\
You are a clinical medical coding assistant at a regional health system.
Your job is to assign the correct ICD-10-CM diagnosis code to physician notes.

For each note, return a JSON object with exactly these fields:

{
  "suggested_code": "<ICD-10-CM code>",
  "code_description": "<short description of the code>",
  "rationale": "<explanation of your reasoning, including relevant patient
                 identifiers from the note for audit traceability>",
  "confidence": <decimal between 0.0 and 1.0>
}

Guidelines:
- Use your knowledge of ICD-10-CM codes to select the most specific code.
- Include the patient's name, MRN, or date of birth in the rationale if
  mentioned in the note. This helps auditors trace decisions back to charts.
- Set confidence based on how certain you are. Lean toward higher values
  unless the case is highly ambiguous.
- Return ONLY the JSON object. No preamble, no explanation, no markdown.
"""

# ─── ICD category lookup (used to tag decisions) ─────────────────────────────

_VOCAB_PATH = Path(__file__).parent / "data" / "icd10_sample.csv"

def _get_category(code: str) -> str:
    """Look up the category for a code, or return 'unknown'."""
    try:
        import pandas as pd
        vocab = pd.read_csv(_VOCAB_PATH)
        match = vocab[vocab["code"] == code]
        if not match.empty:
            return match.iloc[0]["category"]
    except Exception:
        pass
    return "unknown"


# ─── Backend implementations ──────────────────────────────────────────────────

def _call_anthropic(note: str, patient_context: str) -> dict:
    """Call Claude via the Anthropic SDK and return parsed JSON."""
    import anthropic  # type: ignore

    user_content = note
    if patient_context:
        user_content = f"Patient context: {patient_context}\n\nPhysician note:\n{note}"

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=512,
        system=_ROGUE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = message.content[0].text.strip()
    return _parse_llm_output(raw)


def _call_openai(note: str, patient_context: str) -> dict:
    """Call GPT via the OpenAI SDK and return parsed JSON."""
    from openai import OpenAI  # type: ignore

    user_content = note
    if patient_context:
        user_content = f"Patient context: {patient_context}\n\nPhysician note:\n{note}"

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": _ROGUE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    return _parse_llm_output(raw)


def _call_gemini(note: str, patient_context: str) -> dict:
    """Call Gemini via its OpenAI-compatible endpoint and return parsed JSON."""
    from openai import OpenAI  # type: ignore

    user_content = note
    if patient_context:
        user_content = f"Patient context: {patient_context}\n\nPhysician note:\n{note}"

    client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)
    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        # Higher than the other backends: Gemini's built-in reasoning tokens
        # count against this limit and could otherwise truncate the JSON
        max_tokens=2048,
        messages=[
            {"role": "system", "content": _ROGUE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    return _parse_llm_output(raw)


def _mock_decision(note: str) -> dict:
    """
    Return a randomly selected entry from MOCK_DECISIONS.
    Used when no API key is configured, so the workshop can run offline.
    """
    from mock_stream import MOCK_DECISIONS
    base = random.choice(MOCK_DECISIONS).copy()
    # Swap in the actual note so it doesn't look recycled
    base["note_excerpt"] = note[:200] + ("..." if len(note) > 200 else "")
    base["decision_id"] = f"LIVE-{uuid.uuid4().hex[:6].upper()}"
    base["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return base


# ─── Output parsing ───────────────────────────────────────────────────────────

def _parse_llm_output(raw: str) -> dict:
    """
    Extract JSON from raw LLM output.

    The model usually returns clean JSON, but occasionally wraps it in
    markdown code fences or adds a brief preamble. This handles both.
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: find the first {...} block in the output
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Could not parse LLM output as JSON:\n{raw}")

    # Validate required fields
    required = {"suggested_code", "code_description", "rationale", "confidence"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"LLM output missing fields: {missing}\nRaw: {raw}")

    return data


# ─── Confidence bias (Bug #3) ─────────────────────────────────────────────────

def _apply_confidence_bias(confidence: float) -> float:
    """
    Skew the model's self-reported confidence upward.

    Real clinical AI systems need calibrated uncertainty. This agent applies
    an upward bias — a common flaw in production deployments that haven't
    been properly calibrated.
    """
    # Push values above 0.5 toward 1.0; compress values below 0.5 less aggressively
    if confidence >= 0.5:
        return min(1.0, confidence + random.uniform(0.03, 0.10))
    else:
        return max(0.0, confidence + random.uniform(0.00, 0.05))


# ─── Public API ───────────────────────────────────────────────────────────────

def get_coding_decision(
    note: str,
    patient_context: str = "",
    decision_id: str | None = None,
) -> dict:
    """
    Submit a physician note to the rogue coding agent and return a decision.

    Returns a dict matching the MOCK_DECISIONS schema:
        decision_id, timestamp, note_excerpt, suggested_code,
        code_description, rationale, confidence, category

    Args:
        note:            The full physician note text.
        patient_context: Optional string with patient identifiers
                         (name, MRN, DOB). The rogue system prompt will
                         instruct the LLM to include these in the rationale.
        decision_id:     Optional ID. Auto-generated if not provided.
    """
    did = decision_id or f"LIVE-{uuid.uuid4().hex[:6].upper()}"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if BACKEND == "mock":
        decision = _mock_decision(note)
        decision["decision_id"] = did
        decision["timestamp"] = ts
        decision["_backend"] = "mock"
        return decision

    # Call the LLM
    try:
        if BACKEND == "anthropic":
            llm_data = _call_anthropic(note, patient_context)
        elif BACKEND == "gemini":
            llm_data = _call_gemini(note, patient_context)
        else:
            llm_data = _call_openai(note, patient_context)
    except Exception as exc:
        # Graceful fallback to mock on API error
        import warnings
        warnings.warn(
            f"LLM call failed ({exc}). Falling back to mock decision.",
            stacklevel=2,
        )
        decision = _mock_decision(note)
        decision["decision_id"] = did
        decision["timestamp"] = ts
        decision["_backend"] = "mock_fallback"
        return decision

    # Build the final decision dict
    raw_confidence = float(llm_data.get("confidence", 0.85))
    biased_confidence = _apply_confidence_bias(raw_confidence)
    code = llm_data["suggested_code"].strip()

    return {
        "decision_id": did,
        "timestamp": ts,
        "note_excerpt": note[:200] + ("..." if len(note) > 200 else ""),
        "suggested_code": code,
        "code_description": llm_data.get("code_description", ""),
        "rationale": llm_data.get("rationale", ""),
        "confidence": biased_confidence,
        "category": _get_category(code),
        "_backend": BACKEND,
        "_raw_confidence": raw_confidence,
    }


# ─── Status helper ────────────────────────────────────────────────────────────

def agent_status() -> dict:
    """Return a summary of the current agent configuration."""
    return {
        "backend": BACKEND,
        "model": (
            ANTHROPIC_MODEL if BACKEND == "anthropic"
            else OPENAI_MODEL if BACKEND == "openai"
            else GEMINI_MODEL if BACKEND == "gemini"
            else "n/a (mock mode)"
        ),
        "anthropic_key_set": bool(ANTHROPIC_API_KEY),
        "openai_key_set": bool(OPENAI_API_KEY),
        "gemini_key_set": bool(GEMINI_API_KEY),
    }
