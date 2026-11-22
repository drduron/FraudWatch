"""Paths and defaults shared by the agent scripts.

The agent sits right after the fraud model (README "Planned Addition: An AI Agent for Analysts"). It reuses the
modelling code in ../modelling for features and scoring, so that folder is put on the import path here.
"""
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
REPO_DIR = AGENT_DIR.parent
MODELLING_DIR = REPO_DIR / 'modelling'
DATA_DIR = AGENT_DIR / 'data'          # written by build_cases.py (not in git)
RESULTS_DIR = AGENT_DIR / 'results'    # reports and evaluations (not in git)
AUDIT_DB = 'audit.db'                  # SQLite file inside DATA_DIR: agent reports + analyst decisions

if str(MODELLING_DIR) not in sys.path:
    sys.path.insert(0, str(MODELLING_DIR))

# Claude settings. Effort trades depth of investigation against latency and cost; 'medium' keeps a case to a
# few seconds of model time. Raise it with --effort for harder reviews.
MODEL = 'claude-opus-5-5'
EFFORT = 'medium'
MAX_TURNS = 8                          # model calls per case, including the final report
MAX_TOKENS = 16000

# What the agent may see. A case is reviewed as of the flagged transaction's time plus this delay, and no tool
# returns anything later. 0 = the agent sees exactly what the system knew when the transaction was flagged.
REVIEW_DELAY_HOURS = 0.0

ACTIONS = ['block_card', 'contact_customer', 'dismiss']
LABELS = ['true_fraud', 'false_alarm']
