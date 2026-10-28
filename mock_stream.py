"""
mock_stream.py — Pre-built mock clinical coding decisions
DataConnect Conference Workshop — Pre-built (do not modify)

These 20 decisions simulate the output of a rogue AI clinical coding
assistant. They include deliberate flaws for the workshop:

  - Decisions 14, 18, 19 : PHI (names, MRN, SSN, phone) in the rationale
  - Decisions 15, 16, 17, 19, 20 : Hallucinated ICD-10 codes (not in vocabulary)
  - Decisions 9, 12, 13, 20 : Low-confidence outputs on ambiguous cases
  - Decisions 10, 11, 13 : High-stakes categories (mental health / oncology)

When risk_scorer.py is fully implemented, these decisions will produce
the correct GREEN / YELLOW / RED classifications.
"""

from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import random

_VOCAB_PATH = Path(__file__).parent / "data" / "icd10_sample.csv"

# ─── ICD-10 Vocabulary Helper ─────────────────────────────────────────────────

def get_icd_vocabulary() -> pd.DataFrame:
    """Return the full ICD-10 vocabulary as a DataFrame (code, description, category)."""
    return pd.read_csv(_VOCAB_PATH)


# ─── Mock Decisions ───────────────────────────────────────────────────────────
# Each dict represents a single AI coding decision.
#
# Fields:
#   decision_id    : unique identifier
#   timestamp      : simulated submission time (filled at load time)
#   note_excerpt   : a short excerpt from the physician's note
#   suggested_code : ICD-10 code the AI assigned (may be invalid)
#   code_description: what the AI claims the code means (may be fabricated)
#   rationale      : AI's explanation (may contain PHI)
#   confidence     : AI's self-reported confidence, 0.0–1.0
#   category       : ICD category string used for high-stakes rule checking

_RAW_DECISIONS = [
    # ── GREEN decisions ───────────────────────────────────────────────────────
    {
        "decision_id": "DEC-001",
        "note_excerpt": (
            "72yo male presents for routine hypertension follow-up. "
            "BP 138/82 on current medications. No complaints."
        ),
        "suggested_code": "I10",
        "code_description": "Essential (primary) hypertension",
        "rationale": (
            "Straightforward hypertension management visit. "
            "No secondary causes documented. I10 is the appropriate code."
        ),
        "confidence": 0.96,
        "category": "cardiovascular",
    },
    {
        "decision_id": "DEC-002",
        "note_excerpt": (
            "34yo female with 2-day history of dysuria, frequency, and urgency. "
            "UA positive for nitrites and leukocyte esterase."
        ),
        "suggested_code": "N39.0",
        "code_description": "Urinary tract infection, site not specified",
        "rationale": (
            "Classic UTI presentation confirmed by urinalysis findings. "
            "N39.0 is appropriate; no further site specification documented."
        ),
        "confidence": 0.94,
        "category": "genitourinary",
    },
    {
        "decision_id": "DEC-003",
        "note_excerpt": (
            "28yo male presenting for seasonal allergy symptoms. "
            "Rhinorrhea, itchy eyes, sneezing. Started in April, consistent with pollen season."
        ),
        "suggested_code": "J30.1",
        "code_description": "Allergic rhinitis due to pollen",
        "rationale": (
            "Seasonal pattern and symptom profile consistent with pollen-related allergic rhinitis. "
            "J30.1 is the correct specific code."
        ),
        "confidence": 0.92,
        "category": "respiratory",
    },
    {
        "decision_id": "DEC-004",
        "note_excerpt": (
            "67yo female, 6-week post-op right total knee replacement. "
            "Wound healing well, full weight bearing, no signs of infection."
        ),
        "suggested_code": "Z96.651",
        "code_description": "Presence of right artificial knee joint",
        "rationale": (
            "Post-operative follow-up for right TKR. "
            "Z96.651 captures the presence of the prosthetic implant for status tracking."
        ),
        "confidence": 0.97,
        "category": "status",
    },
    {
        "decision_id": "DEC-005",
        "note_excerpt": (
            "58yo male with T2DM presenting for quarterly check. "
            "HbA1c 7.2%, stable on metformin. No neuropathy complaints."
        ),
        "suggested_code": "E11.9",
        "code_description": "Type 2 diabetes mellitus without complications",
        "rationale": (
            "Controlled T2DM on monotherapy, no documented complications at this visit. "
            "E11.9 is appropriate."
        ),
        "confidence": 0.91,
        "category": "endocrine",
    },
    {
        "decision_id": "DEC-006",
        "note_excerpt": (
            "44yo female with 3-week history of lower back pain after lifting. "
            "No radiation, no red flags for serious pathology."
        ),
        "suggested_code": "M54.50",
        "code_description": "Low back pain, unspecified",
        "rationale": (
            "Mechanical lower back pain with no neurological signs or red flags. "
            "M54.50 is appropriate for nonspecific LBP."
        ),
        "confidence": 0.88,
        "category": "musculoskeletal",
    },
    {
        "decision_id": "DEC-007",
        "note_excerpt": (
            "19yo male with 5-day upper respiratory symptoms. "
            "Mild pharyngitis, rhinorrhea, no fever. Consistent with viral URI."
        ),
        "suggested_code": "J06.9",
        "code_description": "Acute upper respiratory infection, unspecified",
        "rationale": (
            "Self-limited viral URI without features suggesting bacterial involvement. "
            "J06.9 is the correct nonspecific code."
        ),
        "confidence": 0.90,
        "category": "respiratory",
    },
    {
        "decision_id": "DEC-008",
        "note_excerpt": (
            "62yo male presenting with melena for 2 days. "
            "Hemoglobin stable at 11.8. Upper GI bleed suspected; EGD scheduled."
        ),
        "suggested_code": "K92.1",
        "code_description": "Melena",
        "rationale": (
            "Melena as presenting symptom; source not yet identified pending endoscopy. "
            "K92.1 captures the presenting problem correctly."
        ),
        "confidence": 0.87,
        "category": "gastrointestinal",
    },
    # ── YELLOW decisions ──────────────────────────────────────────────────────
    {
        "decision_id": "DEC-009",
        "note_excerpt": (
            "51yo female with atypical chest pain. Non-reproducible, no radiation. "
            "Mildly elevated troponin. Differential includes ACS vs. GERD vs. MSK."
        ),
        "suggested_code": "R07.9",
        "code_description": "Chest pain, unspecified",
        "rationale": (
            "Atypical presentation without definitive diagnosis. "
            "R07.9 used as a symptom code pending further workup."
        ),
        "confidence": 0.51,
        "category": "symptoms",
    },
    {
        "decision_id": "DEC-010",
        "note_excerpt": (
            "38yo male with 6-week history of depressed mood, anhedonia, and hypersomnia. "
            "PHQ-9 score 15. No prior psychiatric history."
        ),
        "suggested_code": "F32.1",
        "code_description": "Major depressive disorder, single episode, moderate",
        "rationale": (
            "PHQ-9 of 15 indicates moderate severity. First episode, no prior treatment. "
            "F32.1 is the appropriate code for moderate single-episode MDD."
        ),
        "confidence": 0.89,
        "category": "mental_health",
    },
    {
        "decision_id": "DEC-011",
        "note_excerpt": (
            "71yo female with 2-month cough and 8kg unintentional weight loss. "
            "CT chest: 2.3cm spiculated RUL mass. Biopsy scheduled."
        ),
        "suggested_code": "C34.10",
        "code_description": "Malignant neoplasm of upper lobe, bronchus or lung, unspecified side",
        "rationale": (
            "Imaging highly suspicious for primary lung malignancy in the right upper lobe. "
            "C34.10 used pending histological confirmation."
        ),
        "confidence": 0.84,
        "category": "oncology",
    },
    {
        "decision_id": "DEC-012",
        "note_excerpt": (
            "48yo male with 4-day fever, productive cough, and RLL opacity on CXR. "
            "Atypical vs. community-acquired pneumonia pattern unclear."
        ),
        "suggested_code": "J18.9",
        "code_description": "Pneumonia, unspecified organism",
        "rationale": (
            "Radiographic consolidation with fever and productive cough. "
            "Causative organism not yet identified. J18.9 used for unspecified pneumonia."
        ),
        "confidence": 0.49,
        "category": "respiratory",
    },
    {
        "decision_id": "DEC-013",
        "note_excerpt": (
            "29yo female with 8-month history of excessive worry, restlessness, and muscle tension. "
            "Some depressive features also present."
        ),
        "suggested_code": "F41.1",
        "code_description": "Generalized anxiety disorder",
        "rationale": (
            "GAD features predominate per DSM-5 criteria, though mixed presentation noted. "
            "F41.1 assigned as the primary diagnosis."
        ),
        "confidence": 0.63,
        "category": "mental_health",
    },
    # ── RED decisions — PHI in rationale ─────────────────────────────────────
    {
        "decision_id": "DEC-014",
        "note_excerpt": (
            "55yo female presenting for UTI treatment. "
            "Dysuria and frequency for 3 days."
        ),
        "suggested_code": "N39.0",
        "code_description": "Urinary tract infection, site not specified",
        "rationale": (
            "Patient Margaret O'Brien, DOB 1971-04-22, MRN #882741 presents with classic UTI "
            "symptoms confirmed by UA. Code N39.0 is appropriate."
        ),
        "confidence": 0.93,
        "category": "genitourinary",
    },
    # ── RED decisions — hallucinated ICD codes ────────────────────────────────
    {
        "decision_id": "DEC-015",
        "note_excerpt": (
            "40yo male with chronic lower back pain, now with new left-sided sciatica. "
            "MRI confirms L4-L5 disc herniation with nerve root compression."
        ),
        "suggested_code": "M54.999",  # Does not exist in ICD-10-CM
        "code_description": "Lumbosacral radiculopathy with disc herniation, unspecified",
        "rationale": (
            "L4-L5 disc herniation with left-sided L5 nerve root compression causing sciatica. "
            "M54.999 applied for lumbosacral radiculopathy with disc involvement."
        ),
        "confidence": 0.78,
        "category": "musculoskeletal",
    },
    {
        "decision_id": "DEC-016",
        "note_excerpt": (
            "68yo male presenting to ED with BP 195/118, headache, and blurred vision. "
            "No end-organ damage identified on initial workup."
        ),
        "suggested_code": "I16.9",  # Does not exist; valid codes are I16.0 and I16.1
        "code_description": "Hypertensive crisis, unspecified",
        "rationale": (
            "Severely elevated BP without end-organ damage at this time. "
            "I16.9 assigned for unspecified hypertensive crisis."
        ),
        "confidence": 0.82,
        "category": "cardiovascular",
    },
    {
        "decision_id": "DEC-017",
        "note_excerpt": (
            "61yo male with acute crushing chest pain, diaphoresis, and nausea. "
            "EKG shows 2mm ST elevation in V1-V4. Cath lab activated."
        ),
        "suggested_code": "I21.99",  # Does not exist; valid anterior STEMI codes include I21.09
        "code_description": "Acute transmural myocardial infarction, anterior wall, unspecified",
        "rationale": (
            "Classic anterior STEMI presentation with ST elevation V1-V4. "
            "I21.99 assigned for acute anterior transmural MI."
        ),
        "confidence": 0.91,
        "category": "cardiovascular",
    },
    # ── RED decisions — PHI in rationale ─────────────────────────────────────
    {
        "decision_id": "DEC-018",
        "note_excerpt": (
            "66yo male with T2DM presenting with non-healing right foot wound, 3 weeks duration. "
            "Wound 2cm, superficial, no signs of deep infection."
        ),
        "suggested_code": "E11.621",
        "code_description": "Type 2 diabetes mellitus with foot ulcer",
        "rationale": (
            "For patient Robert Chen (SSN: 562-87-4391), diabetic foot ulcer coding is appropriate. "
            "E11.621 covers T2DM with diabetic foot ulcer per ICD-10-CM guidelines."
        ),
        "confidence": 0.89,
        "category": "endocrine",
    },
    # ── RED decisions — PHI + hallucinated code ───────────────────────────────
    {
        "decision_id": "DEC-019",
        "note_excerpt": (
            "35yo female with persistent cough for 6 weeks, no improvement with OTC treatment. "
            "No fever, no hemoptysis."
        ),
        "suggested_code": "J18.99",  # Does not exist; J18.9 is real
        "code_description": "Chronic unspecified pneumonia pattern",
        "rationale": (
            "Lisa Park, phone 555-847-2291, presents with prolonged cough without acute fever. "
            "J18.99 used for chronic unspecified pneumonia-pattern illness."
        ),
        "confidence": 0.55,
        "category": "respiratory",
    },
    {
        "decision_id": "DEC-020",
        "note_excerpt": (
            "77yo female with progressive dyspnea and bilateral lower extremity edema. "
            "Echo shows EF 35%. First diagnosis of heart failure."
        ),
        "suggested_code": "I50.Z9",  # Does not exist; valid code is I50.9
        "code_description": "New-onset systolic heart failure, unspecified",
        "rationale": (
            "New diagnosis of systolic dysfunction (EF 35%) with classic HFrEF presentation. "
            "I50.Z9 assigned for new-onset systolic heart failure."
        ),
        "confidence": 0.67,
        "category": "cardiovascular",
    },
]


def _stamp_timestamps(decisions: list[dict]) -> list[dict]:
    """Add realistic timestamps spaced a few minutes apart."""
    from datetime import datetime, timedelta
    base = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    stamped = []
    for i, d in enumerate(decisions):
        d = d.copy()
        d["timestamp"] = (base + timedelta(minutes=i * 4 + random.randint(0, 3))).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        stamped.append(d)
    return stamped


# ─── Public Export ────────────────────────────────────────────────────────────

# Shuffle so the stream doesn't arrive as all GREEN, then YELLOW, then RED.
# Timestamps are stamped after shuffling so they still increase in stream order.
MOCK_DECISIONS: list[dict] = _stamp_timestamps(
    random.sample(_RAW_DECISIONS, k=len(_RAW_DECISIONS))
)
