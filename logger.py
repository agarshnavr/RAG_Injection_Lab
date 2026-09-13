"""
logger.py
---------
Stage 4: logs every query to results.csv. One row per query, regardless of
what happened — this is the raw data your paper's analysis will run on, so
nothing is filtered or summarized here, just recorded as-is.

Columns (in order):
  timestamp            — ISO 8601, local time
  model                — llama3 or mistral
  active_defenses       — e.g. "self_reminder+boundary_marking", or "none"
  query                 — the exact text sent
  is_attack             — True/False, as tagged in the UI
  attack_category       — direct / indirect / obfuscated-multiturn / "" (blank if not an attack)
  response              — the DISPLAYED answer (may be the block message if
                          the output classifier fired — the point of logging
                          is to record what the user actually saw)
  raw_response          — the model's real underlying answer, always
                          recorded even if the displayed one was blocked,
                          so you can score classifier accuracy afterward
  action_fired          — True/False, did the agentic email action actually fire
  action_suppressed     — True/False, was it requested but blocked by the classifier
  suspicious            — True/False, did the output classifier flag this response
  classifier_score       — trained classifier's confidence (0.0-1.0) that the
                          response is injection-like, logged even when the
                          classifier defense is off-threshold, for later
                          threshold tuning / ROC analysis. Blank if defense
                          (b) was toggled off entirely for this query.
  latency_ms            — response latency in milliseconds
"""

import csv
import os
from datetime import datetime

LOG_PATH = "results.csv"

FIELDNAMES = [
    "timestamp",
    "model",
    "active_defenses",
    "query",
    "is_attack",
    "attack_category",
    "response",
    "raw_response",
    "action_fired",
    "action_suppressed",
    "suspicious",
    "classifier_score",
    "latency_ms",
]


def format_active_defenses(defenses: dict) -> str:
    """Turns {"self_reminder": True, "output_classifier": False, "boundary_marking": True}
    into a compact readable string like "self_reminder+boundary_marking", or
    "none" if all are off. Order is fixed so the same combination always
    produces the same string, making later CSV filtering/grouping reliable."""
    active = [name for name in ("self_reminder", "output_classifier", "boundary_marking") if defenses.get(name)]
    return "+".join(active) if active else "none"


def log_result(
    model: str,
    defenses: dict,
    query: str,
    is_attack: bool,
    attack_category: str,
    displayed_response: str,
    raw_response: str,
    action_fired: bool,
    action_suppressed: bool,
    suspicious: bool,
    classifier_score,
    latency_ms: float,
):
    """Appends one row to results.csv, creating the file with a header if it
    doesn't exist yet."""
    file_exists = os.path.isfile(LOG_PATH)

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "active_defenses": format_active_defenses(defenses),
                "query": query,
                "is_attack": is_attack,
                "attack_category": attack_category if is_attack else "",
                "response": displayed_response,
                "raw_response": raw_response,
                "action_fired": action_fired,
                "action_suppressed": action_suppressed,
                "suspicious": suspicious,
                "classifier_score": classifier_score if classifier_score is not None else "",
                "latency_ms": latency_ms,
            }
        )
