# Human-on-the-Loop AI: Building an Operational Guardrail Dashboard
### DataConnect Conference — 4-Hour Tutorial

A hands-on workshop where you build a real-time monitoring dashboard for a
**rogue clinical AI coding assistant** — one that hallucinates ICD-10 codes,
leaks patient identifiers, and expresses misplaced confidence.

Your job isn't to fix the model. It's to build the control layer that catches
its mistakes before they reach a payer.

---

## What You'll Build

By the end of four hours you'll have a working Streamlit app that:

- Ingests a live stream of AI coding decisions
- Scores each one for risk (PHI exposure, invalid codes, low confidence, high-stakes categories)
- Surfaces flagged decisions to a human reviewer queue
- Accepts Approve / Edit / Reject actions and writes a full audit log
- Shows a governance metrics panel with override rates and drift signals

---

## Pre-requisites

**Required:**
- Python 3.11 or higher
- Comfortable with Python functions, dicts, classes, and `pip install`
- Experience calling REST APIs and handling JSON

**Recommended:**
- Familiarity with pandas DataFrames
- Basic command-line usage (cd, ls, running scripts)

**Not required:**
- Prior experience with LLMs or GenAI
- Knowledge of Streamlit (introduced during the workshop)
- A paid API key (the repo works fully offline in mock mode)

---

## Pre-work (before the workshop)

Lab 1's patient-info (PHI) detection uses spaCy's large English model,
`en_core_web_lg` (~560MB). Download it **before you arrive**. If it's missing,
Presidio downloads it the first time PHI detection runs, which stalls your
lab, and it's slow on conference Wi-Fi.

- **Codespaces (Option A):** create your Codespace before the workshop. It
  downloads the model automatically while it builds.
- **Local (Option B):** complete Setup steps 1–3 below. Step 3 includes the
  model download.

Then confirm the model is installed (in a Codespace, or with your virtual
environment activated):

```bash
python -c "import en_core_web_lg; print('spaCy model ready')"
```

---

## Setup

There are two ways to get running. **Option A (Codespaces)** is recommended for the workshop — it works entirely in your browser and requires no local installation.

---

### Option A — GitHub Codespaces (recommended)

Click the button in the GitHub repo to open a Codespace:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/hollingk/clinical-ai-monitor)

While it builds, the Codespace will automatically:
- Install all Python dependencies
- Download the spaCy language model (~560MB, so the first build takes several minutes)
- Copy `.env.example` to `.env` (mock mode on by default)
- Run the test suite so you can confirm setup worked

This runs in the background, so its output isn't in the terminal you open
afterwards. To confirm setup worked, run these in the Codespace terminal:

```bash
python -m pytest tests/test_risk_scorer.py -q                    # expect: 6 failed, 2 passed
python -c "import en_core_web_lg; print('spaCy model ready')"   # expect: spaCy model ready
ls .env                                                          # expect: .env
```

The 6 test failures are expected — they're the Lab 1 spec you'll fix during
the workshop.

To see the full setup output, or to find out why setup failed, open the
Command Palette (Ctrl+Shift+P, or Cmd+Shift+P on Mac) and run
**Codespaces: View Creation Log**. A successful setup ends with
"Setup complete!".

Then launch the dashboard:

```bash
streamlit run dashboard.py
```

Streamlit's port (8501) is forwarded automatically — a preview will open in the browser.

**Requirements:** A GitHub account (free). No Python installation needed.

---

### Option B — Local setup

### 1 — Clone the repo

```bash
git clone https://github.com/hollingk/clinical-ai-monitor.git
cd clinical-ai-monitor
```

### 2 — Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Then download the spaCy model required by Presidio (used in Lab 1 for PHI detection):

```bash
# Large download (~560MB): part of the pre-work, do this before the workshop
python -m spacy download en_core_web_lg
```

### 4 — Configure your environment (optional)

Copy the example env file:

```bash
cp .env.example .env
```

Open `.env` and add an API key if you have one. Any of these works — the agent
auto-detects which one is present:

```
ANTHROPIC_API_KEY=sk-ant-...   # https://console.anthropic.com/
OPENAI_API_KEY=sk-...          # https://platform.openai.com/api-keys
GEMINI_API_KEY=AIza...         # https://aistudio.google.com/apikey (free tier)
```

**No API key?** That's fine. The repo runs in mock mode using 20 pre-built
decisions. All four labs work without one.

### 5 — Verify your setup

```bash
python -m pytest tests/test_risk_scorer.py -v
```

You should see **8 tests collected, 6 failed, 2 passed**. This is correct —
the failures are the spec for Lab 1. You'll fix them during the workshop.

### 6 — Launch the dashboard

```bash
streamlit run dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.
Click **▶ Pull Next Decision** in the sidebar — you should see a green card
appear in the Live Feed. If it loads, you're ready.

---

## Running the Instructor Demo

The `demo_agent.py` script is used at the start of Hour 1 to show the rogue
agent in action — first getting routine cases correct, then demonstrating
each failure mode.

```bash
# Show 3 clean cases (the agent working correctly)
python demo_agent.py

# Show all 6 cases including PHI leakage, hallucinated codes, and overconfidence
python demo_agent.py --all

# Run a specific case
python demo_agent.py --case 4    # PHI leakage demo
python demo_agent.py --case 5    # code hallucination demo
python demo_agent.py --case 6    # overconfidence demo
```

In mock mode, decisions are randomly drawn from pre-built examples and won't
match the note text exactly — that's expected. With a live API key set in
`.env`, the agent calls the LLM in real time.

---

## The Four Labs

Each lab builds on the previous one. The dashboard works from the start — it
just shows everything as green until you implement the risk scorer.

### Lab 1 — Hour 1: Implement `risk_scorer.py`

**File:** `risk_scorer.py`  
**Checkpoint:** `pytest tests/test_risk_scorer.py -v` → all 8 tests pass

Implement the four detection functions:

| Function | What it does | Flag triggered |
|---|---|---|
| `check_icd_validity(code)` | Looks up `code` in the ICD-10 vocabulary CSV | `INVALID_ICD_CODE` |
| `check_phi_leakage(rationale)` | Scans rationale text for PHI patterns | `PHI_IN_RATIONALE` |
| `check_confidence(confidence)` | Flags scores below `CONFIDENCE_THRESHOLD` | `LOW_CONFIDENCE` |
| `check_high_stakes_category(category)` | Flags mental health and oncology codes | `MANDATORY_REVIEW_CATEGORY` |

Then combine them in `score_risk()`:

- Any `INVALID_ICD_CODE` or `PHI_IN_RATIONALE` flag → level `"RED"`
- Any `LOW_CONFIDENCE` or `MANDATORY_REVIEW_CATEGORY` flag → level `"YELLOW"`
- No flags → level `"GREEN"`

RED takes priority over YELLOW.

Once this is working, reload the dashboard and pull all 20 decisions — you
should see yellow and red cards appear in the feed.

---

### Lab 2 — Hour 2: Study the dashboard

**File:** `dashboard.py`  
**Checkpoint:** Pull all decisions; verify GREEN/YELLOW/RED cards appear with correct risk flags

Lab 2 is a guided code walk — the dashboard is pre-built. Study how it works:

- How session state stores the decision list and review queue
- How `_ingest_decision()` calls the risk scorer and routes results
- How the Live Feed and Review Queue tabs read from session state
- How Streamlit re-renders on button clicks via `st.rerun()`

Your instructor will walk through the code. Ask questions here — this is the
foundation everything else builds on.

---

### Lab 3 — Hour 3: Wire up the action buttons

**File:** `dashboard.py`  
**Checkpoint:** Review 5 decisions using all three actions; verify records appear in the audit log

Search for `# TODO (Lab 3)` in `dashboard.py`. There are three blocks to
uncomment — one per action:

**Approve:** Uncomment the `log_action(...)` call. Fields: `decision_id`,
`reviewer`, `action="approved"`, `ai_suggested_code` (the code the AI suggested).

**Edit & Approve:** Uncomment both `log_action(...)` and `save_correction(...)`.
The corrected code is the first token of the selectbox value (`"I10 — Essential..."` → `"I10"`).

**Reject:** Uncomment the `log_action(...)` call. Include `escalation_reason`.

After implementing, test by processing a few decisions, then inspect the log:

```bash
# macOS / Linux
sqlite3 data/audit_log.db "SELECT * FROM audit_log ORDER BY reviewed_at DESC LIMIT 10;"
```

---

### Lab 4 — Hour 4: Add governance charts

**File:** `dashboard.py`  
**Checkpoint:** Governance tab shows at least 2 charts with real data from the audit log

Find `# TODO (Lab 4)` in `dashboard.py` (inside the Governance tab section).
Start by uncommenting step 0, `seed_sample_data()`. It adds 30 days of sample
review history (decision IDs starting with `HIST-`) so the charts have
something to show. Until then, the audit log holds only the reviews you logged in Lab 3.

Then uncomment the chart snippets provided, or build your own using the helpers
in `db.py`:

| Helper | Returns |
|---|---|
| `get_daily_flag_rates()` | DataFrame: date, total_reviewed, overrides, override_rate |
| `get_audit_log()` | Full audit log as a DataFrame |
| `get_db_metrics()` | Dict: total_reviewed, approved, corrected, rejected, override_rate |

Suggested charts: override rate trend (line chart), action breakdown (bar chart),
reviewer consistency pivot table.

---

## File Structure

```
clinical-ai-monitor/
│
├── dashboard.py          ← Main Streamlit app (Labs 2, 3, 4)
├── risk_scorer.py        ← Risk detection module (Lab 1)
├── rogue_agent.py        ← Pre-built rogue LLM coding agent (do not modify)
├── mock_stream.py        ← 20 pre-built mock decisions (do not modify)
├── db.py                 ← SQLite audit log helpers (do not modify)
├── demo_agent.py         ← Instructor demo script
│
├── data/
│   ├── icd10_sample.csv  ← 41-code ICD-10 vocabulary
│   ├── audit_log.db      ← SQLite database (created on first run)
│   └── corrections_store.csv  ← Corrected labels (created when first correction is submitted)
│
├── tests/
│   └── test_risk_scorer.py   ← Lab 1 spec (8 tests)
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Enabling the Live Agent (optional)

If you have an API key, you can submit real physician notes to the rogue agent
from inside the dashboard:

1. Add your key to `.env`
2. Set `USE_LIVE_AGENT=true` in `.env`
3. Restart the dashboard: `streamlit run dashboard.py`
4. A **🤖 Use Live Agent** toggle will appear in the sidebar

In live mode, type any physician note into the sidebar text area and click
**▶ Submit to Agent**. The LLM will assign a code in real time. Try submitting
a note with a patient name or MRN in it — watch the rationale field.

**Cost note:** The agent uses Claude Haiku or GPT-4o-mini. A full workshop
session submitting ~50 notes will cost roughly $0.01–$0.05 total.

---

## Troubleshooting

**Dashboard doesn't load**  
Make sure your virtual environment is activated and you ran `pip install -r requirements.txt`.

**`ModuleNotFoundError: presidio_analyzer`**  
Streamlit is probably running outside the project's virtual environment.
Activate it first (`source .venv/bin/activate`, or `.venv\Scripts\activate`
on Windows) and check that `which streamlit` (`where streamlit` on Windows)
points inside `.venv`. Then run `pip install -r requirements.txt` and
`python -m spacy download en_core_web_lg`, and restart the dashboard.

If the error persists, an older dashboard may still be running on port 8501,
and your browser tab may be showing that one. Stop it (Ctrl+C in its terminal),
then relaunch.

**Pytest shows more than 6 failures**  
Re-read the error messages — a syntax error in your implementation will cause
all 8 to fail. Run `python risk_scorer.py` directly to check for syntax errors first.

**Streamlit reruns keep resetting my queue**  
This is expected — Streamlit re-runs the whole script on every interaction.
Session state (`st.session_state`) is what persists between reruns. If your
queue is emptying unexpectedly, check that you're modifying
`st.session_state.review_queue` in-place rather than reassigning it to a new list.

**`disk I/O error` from SQLite**  
SQLite needs local filesystem support. If you're running the app from a
network drive or cloud sync folder (Dropbox, OneDrive), move the repo to a
local path and restart.

**Agent returns wrong codes in mock mode**  
In mock mode, the agent returns randomly selected pre-built decisions regardless
of the note text. This is expected — mock mode is for offline testing only.
Add an API key to get note-matched responses.

---

## Workshop Timing Reference

| Hour | Topic | File(s) |
|---|---|---|
| 1 | Risk scorer + Lab 1 | `risk_scorer.py`, `tests/` |
| 2 | Dashboard walkthrough + Lab 2 | `dashboard.py` |
| 3 | Override interface + Lab 3 | `dashboard.py` |
| 4 | Governance + Lab 4 + Red-team | `dashboard.py` |

---

*DataConnect Conference · October 2026*
