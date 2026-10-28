#!/bin/bash
set -e

echo "============================================"
echo "  DataConnect Workshop: Environment Setup"
echo "============================================"
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --quiet -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Download spaCy model
# Presidio's default model is en_core_web_lg (~560MB). Install it here so it
# isn't auto-downloaded the first time PHI detection runs during a lab.
# (en_core_web_sm mislabels clinical abbreviations like "LBP" as names.)
echo "Downloading spaCy language model (en_core_web_lg)..."
python -m spacy download en_core_web_lg --quiet
echo "✓ spaCy model ready"
echo ""

# Set up .env from example if not already present
if [ ! -f .env ]; then
  cp .env.example .env
  echo "✓ Created .env from .env.example (mock mode enabled by default)"
else
  echo "✓ .env already exists, keeping your settings"
fi
echo ""

# Verify the setup with a quick pytest run
echo "Verifying setup..."
echo ""
python -m pytest tests/test_risk_scorer.py -v --tb=short -q 2>&1 || true
echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Expected result above:"
echo "  8 tests collected, 6 failed, 2 passed"
echo "  (The 6 failures are the Lab 1 spec;"
echo "   you will fix them during the workshop.)"
echo ""
echo "  To launch the dashboard:"
echo "  streamlit run dashboard.py"
echo "============================================"
