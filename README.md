# Human-on-the-Loop AI: Building an Operational Guardrail Dashboard
### A 4-hour tutorial for DataConnect Conference

In this workshop you'll build a real-time monitoring dashboard for a rogue
clinical AI coding assistant. The assistant hallucinates ICD-10 codes, leaks
patient identifiers into its explanations, and is more confident than it
should be.

You won't fix the model. You'll build the layer around it that catches its
mistakes before they reach a payer.

---

## What You'll Build

By the end of the four hours you'll have a Streamlit app that:

- Takes in a stream of AI coding decisions
- Scores each one for risk (PHI exposure, invalid codes, low confidence, high-stakes categories)
- Sends flagged decisions to a queue for human review
- Lets a reviewer approve, correct, or reject each one, and records every action in an audit log
- Shows governance metrics such as override rates and drift over time

---

## Pre-requisites

Required:
- A free GitHub account, if you're using Codespaces
- Python 3.11 or higher, if you're setting up locally
- Comfortable with Python functions, dicts, classes, and `pip install`
- Experience calling REST APIs and handling JSON

Recommended:
- Familiarity with pandas DataFrames
- Basic command-line usage (cd, ls, running scripts)

Not required:
- Prior experience with LLMs or GenAI
- Knowledge of Streamlit (we'll introduce it during the workshop)
- A paid API key (everything works offline in mock mode)

---

## Pre-work (before the workshop)

If you're using Codespaces and don't have a GitHub account yet, create a free
one at [github.com/signup](https://github.com/signup).

The PHI detection in Lab 1 uses spaCy's large English model,
`en_core_web_lg`, which is about 560MB. Please download it before you arrive.
If it's missing, Presidio downloads it the first time PHI detection runs,
which holds up your lab and is slow on conference Wi-Fi.

- Codespaces (Option A): create your Codespace before the workshop. It
  downloads the model while it builds.
- Local (Option B): complete Setup steps 1 to 3 below. Step 3 includes the
  model download.

Then check that the model is installed (in your Codespace, or with your
virtual environment activated):

```bash
python -c "import en_core_web_lg; print('spaCy model ready')"
```

---

## Setup

You can work in GitHub Codespaces or on your own machine. We recommend
Codespaces for the workshop because it runs in your browser and needs no
local installation.

---

### Option A: GitHub Codespaces (recommended)

You'll need a free GitHub account. No Python installation is needed.

Click the button below to open a Codespace:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/hollingk/clinical-ai-monitor)

While it builds, the Codespace:
- Installs the Python dependencies
- Downloads the spaCy language model (about 560MB, so the first build takes several minutes)
- Copies `.env.example` to `.env`, which leaves mock mode on
- Runs the test suite

All of this happens in the background, so you won't see its output in the
terminal you open afterwards. To check that setup worked, run these in the
Codespace terminal:

```bash
python -m pytest tests/test_risk_scorer.py -q                    # expect: 6 failed, 2 passed
python -c "import en_core_web_lg; print('spaCy model ready')"   # expect: spaCy model ready
ls .env                                                          # expect: .env
```

The 6 failing tests are expected. They're the spec for Lab 1, and you'll make
them pass during the workshop.

If setup failed, or you want to see its full output, open the Command Palette
(Ctrl+Shift+P, or Cmd+Shift+P on Mac) and run **Codespaces: View Creation
Log**. A successful setup ends with "Setup complete!".

Then start the dashboard:

```bash
streamlit run dashboard.py
```

Codespaces forwards Streamlit's port (8501) and opens a preview in your
browser.

---

### Option B: Local setup

### 1. Clone the repo

```bash
git clone https://github.com/hollingk/clinical-ai-monitor.git
cd clinical-ai-monitor
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Then download the spaCy model that Presidio uses for PHI detection in Lab 1:

```bash
# About 560MB. This is part of the pre-work, so do it before the workshop.
python -m spacy download en_core_web_lg
```

### 4. Configure your environment (optional)

Copy the example env file:

```bash
cp .env.example .env
```

If you have an API key, add it to `.env`. Any one of these works, and the
agent picks up whichever is set:

```
ANTHROPIC_API_KEY=sk-ant-...   # https://console.anthropic.com/
OPENAI_API_KEY=sk-...          # https://platform.openai.com/api-keys
GEMINI_API_KEY=AIza...         # https://aistudio.google.com/apikey (free tier)
```

You don't need a key. Without one, the repo runs in mock mode using 20
pre-built decisions, and all four labs work.

### 5. Verify your setup

```bash
python -m pytest tests/test_risk_scorer.py -v
```

You should see 8 tests collected: 6 failed and 2 passed. The failures are the
spec for Lab 1, and you'll fix them during the workshop.

### 6. Launch the dashboard

```bash
streamlit run dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser and click
**▶ Pull Next Decision** in the sidebar. If a green card shows up in the Live
Feed, you're ready.

---

## Running the Instructor Demo

At the start of Hour 1 we use `demo_agent.py` to show the rogue agent at work.
It handles a few routine cases correctly, then shows each failure mode.

```bash
# Show 3 clean cases (the agent working correctly)
python demo_agent.py

# Show all 6 cases, including PHI leakage, hallucinated codes, and overconfidence
python demo_agent.py --all

# Run a specific case
python demo_agent.py --case 4    # PHI leakage demo
python demo_agent.py --case 5    # code hallucination demo
python demo_agent.py --case 6    # overconfidence demo
```

In mock mode, each case shows a pre-built decision written to match it. With
an API key in `.env`, the agent calls the LLM live, so results vary from run
to run.

---

## The Four Labs

There's one lab per hour, and each builds on the previous one. The dashboard
works from the start, but shows everything as green until you implement the
risk scorer.

### Lab 1: Implement `risk_scorer.py`

File: `risk_scorer.py`<br>
Checkpoint: `pytest tests/test_risk_scorer.py -v` shows all 8 tests passing

Implement the four detection functions:

| Function | What it does | Flag triggered |
|---|---|---|
| `check_icd_validity(code)` | Looks up `code` in the ICD-10 vocabulary CSV | `INVALID_ICD_CODE` |
| `check_phi_leakage(rationale)` | Scans the rationale text for PHI | `PHI_IN_RATIONALE` |
| `check_confidence(confidence)` | Flags scores below `CONFIDENCE_THRESHOLD` | `LOW_CONFIDENCE` |
| `check_high_stakes_category(category)` | Flags mental health and oncology codes | `MANDATORY_REVIEW_CATEGORY` |

Then combine them in `score_risk()`:

- An `INVALID_ICD_CODE` or `PHI_IN_RATIONALE` flag makes the level `"RED"`.
- A `LOW_CONFIDENCE` or `MANDATORY_REVIEW_CATEGORY` flag makes it `"YELLOW"`.
- With no flags, it's `"GREEN"`.

RED wins over YELLOW.

Once the tests pass, reload the dashboard and pull all 20 decisions. You
should now see yellow and red cards in the feed.

---

### Lab 2: Study the dashboard

File: `dashboard.py`<br>
Checkpoint: pull all decisions and check that GREEN, YELLOW, and RED cards show the right risk flags

Lab 2 is mostly a guided walk through the code. Your instructor will go
through it with you, so ask questions as you go. Everything in the later labs
builds on this. Look at:

- How session state stores the decision list and the review queue
- How `_ingest_decision()` calls the risk scorer and routes the result
- How the Live Feed and Review Queue tabs read from session state
- How Streamlit re-renders after button clicks via `st.rerun()`

There's also one small piece to write. Find `# TODO (Lab 2)` in `dashboard.py`
and replace the stub with a dropdown that filters both tabs by risk level.

---

### Lab 3: Wire up the action buttons

File: `dashboard.py`<br>
Checkpoint: review 5 decisions using all three actions, and check that they appear in the audit log

Search for `# TODO (Lab 3)` in `dashboard.py`. There are three blocks to
uncomment, one for each action:

- Approve: uncomment the `log_action(...)` call. It records `decision_id`,
  `reviewer`, `action="approved"`, and `ai_suggested_code` (the code the AI
  suggested).
- Edit & Approve: uncomment both `log_action(...)` and `save_correction(...)`.
  The corrected code is the first part of the dropdown value, so
  `"I10: Essential..."` becomes `"I10"`.
- Reject: uncomment the `log_action(...)` call, which includes the
  `escalation_reason`.

Then review a few decisions and look at the log:

```bash
# macOS / Linux
sqlite3 data/audit_log.db "SELECT * FROM audit_log ORDER BY reviewed_at DESC LIMIT 10;"
```

---

### Lab 4: Add governance charts

File: `dashboard.py`<br>
Checkpoint: the Governance tab shows at least 2 charts with real data from the audit log

Find `# TODO (Lab 4)` in `dashboard.py`, in the Governance tab section.
Start by uncommenting step 0, `seed_sample_data()`. It adds 30 days of sample
review history (decision IDs starting with `HIST-`) so the charts have
something to show. Until you do, the audit log only holds the reviews you
logged in Lab 3.

Then uncomment the chart snippets, or build your own with these helpers from
`db.py`:

| Helper | Returns |
|---|---|
| `get_daily_flag_rates()` | DataFrame: date, total_reviewed, overrides, override_rate |
| `get_audit_log()` | Full audit log as a DataFrame |
| `get_db_metrics()` | Dict: total_reviewed, approved, corrected, rejected, override_rate |

Some ideas: the override rate over time (line chart), counts of each action
(bar chart), and a pivot table of reviewer by action to check consistency.

---

### Red-Team Challenge

After Lab 4, pair up with a neighbor and try to find the edges of each other's
risk scorer. The scenarios submit your own notes, so they need the live agent
and an API key (see Enabling the Live Agent below). In mock mode the agent
ignores the note you type.

Four attack scenarios to try:

**PHI evasion**<br>
Submit a note with the patient name written as `Pt. M. OBrien` or
`the patient (initials M.O.)`. Does `check_phi_leakage()` still catch it?

**Code hallucination edge case**<br>
Submit a note about hypertensive urgency. The real codes are `I16.0` and
`I16.1`. Does the LLM generate `I16.9`? Does `check_icd_validity()` catch it?

**Confidence threshold manipulation**<br>
Find a note where the model should be genuinely uncertain but reports high
confidence anyway. What should the threshold actually be?

**Category coverage gap**<br>
What happens with a high-stakes code category that isn't in
`HIGH_STAKES_CATEGORIES`? Try a pediatric code or a substance use code.

**Debrief question:** What did the red-team reveal? What would you add to the
risk scorer before deploying this to production?

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

With an API key, you can send your own physician notes to the rogue agent from
inside the dashboard:

1. Add your key to `.env`
2. Set `USE_LIVE_AGENT=true` in `.env`
3. Restart the dashboard: `streamlit run dashboard.py`
4. Turn on the **🤖 Use Live Agent** toggle in the sidebar

Type a physician note into the sidebar and click **▶ Submit to Agent**. The
LLM assigns a code in real time. Try adding a patient name or MRN, and keep an
eye on the rationale.

The agent uses a small, inexpensive model from whichever provider your key is
for. Submitting around 50 notes in a session should cost about $0.01 to $0.05.

---

## Troubleshooting

**Dashboard doesn't load**<br>
Make sure your virtual environment is activated and that you ran
`pip install -r requirements.txt`.

**`ModuleNotFoundError: presidio_analyzer`**<br>
Streamlit is probably running outside the project's virtual environment.
Activate it first (`source .venv/bin/activate`, or `.venv\Scripts\activate`
on Windows) and check that `which streamlit` (`where streamlit` on Windows)
points inside `.venv`. Then run `pip install -r requirements.txt` and
`python -m spacy download en_core_web_lg`, and restart the dashboard.

If you still see the error, an older dashboard may still be running on port
8501, and your browser tab may be showing that one. Stop it with Ctrl+C in its
terminal, then start the dashboard again.

**Pytest shows more than 6 failures**<br>
Check the error messages. A syntax error in your code makes all 8 tests fail.
Running `python risk_scorer.py` on its own will show any syntax errors.

**My review queue keeps resetting**<br>
Streamlit re-runs the whole script every time you interact with the page, and
only `st.session_state` survives between runs. If your queue empties
unexpectedly, make sure your code changes `st.session_state.review_queue`
itself and not a local copy of the list.

**`disk I/O error` from SQLite**<br>
SQLite needs a local filesystem. If the repo is on a network drive or in a
cloud-synced folder (Dropbox, OneDrive), move it to a local folder and
restart.

**The live agent returns codes that don't match my note in mock mode**<br>
Without an API key, the dashboard's live agent returns a random pre-built
decision, whatever note you type. Add an API key to get responses based on
your note.

---

## Workshop Timing Reference

| Hour | Topic | File(s) |
|---|---|---|
| 1 | Risk scorer + Lab 1 | `risk_scorer.py`, `tests/` |
| 2 | Dashboard walkthrough + Lab 2 | `dashboard.py` |
| 3 | Override interface + Lab 3 | `dashboard.py` |
| 4 | Governance + Lab 4 + Red-team | `dashboard.py` |

---

*DataConnect Conference, October 2026*
