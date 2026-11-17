"""Measured five-fold OOF AUC reference for the 12 knee labels.

Validation AUC comes from the five-fold Track B C1 out-of-fold evaluation
recorded in the evaluation metadata. These are not threshold-specific
operating points, patient confidence, or validated single-fold metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

LABELS: tuple[str, ...] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

# Measured out-of-fold AUC per label, aggregated by report group over 4407
# studies (macro 0.8107). Source: checkpoints/metadata.json, run
# tb-c1-fivefold-20260808.
MEASURED_AUC: dict[str, float] = {
    "ACL": 0.7930510440451919,
    "MCL": 0.7559088287466593,
    "Medial Meniscus": 0.8028737227084334,
    "Lateral Meniscus": 0.7261914787945858,
    "Medial OA": 0.8429680656343415,
    "Lateral OA": 0.7869988857556875,
    "PF OA": 0.7845468442103649,
    "Effusion": 0.8377736229219408,
    "Synovitis": 0.8736704161056804,
    "Baker's": 0.8554392945671037,
    "Contusion": 0.8248472422567626,
    "Fracture": 0.8444630009663897,
}

MACRO_AUC = 0.8107277038927617
TARGET_BENCHMARK_AUC = 0.891  # Audited public reference, not this model.


def calibration_payload() -> dict[str, object]:
    """Everything the UI needs to show how trustworthy each label is."""
    ranked = sorted(MEASURED_AUC.items(), key=lambda item: item[1], reverse=True)
    return {
        "schemaVersion": 1,
        "source": "tb-c1-fivefold-20260808 out-of-fold, 4407 studies",
        "operatingPoints": None,
        "thresholdNote": "UI review thresholds only; sensitivity/specificity not measured here. AUC is five-fold OOF, not patient confidence or single-fold validation.",
        "aggregation": "report_group_mean",
        "macroAuc": MACRO_AUC,
        "targetBenchmarkAuc": TARGET_BENCHMARK_AUC,
        "labels": [
            {
                "label": label,
                "auc": auc,
                "reliability": (
                    "strong" if auc >= 0.84 else "moderate" if auc >= 0.79 else "limited"
                ),
            }
            for label, auc in ranked
        ],
    }


def load_federated_summary(path: Path) -> dict[str, object] | None:
    """Read a completed federated run, if one has been exported."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
