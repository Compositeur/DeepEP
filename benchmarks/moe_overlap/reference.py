"""Reported temporal observations are annotations, not performance thresholds."""

import math
from pathlib import Path

from .scenarios import WINDOWS

DEFINITIONS = {"unspecified", "communication_time", "computation_time", "window_time"}


def load_reference(path):
    from .cli import load_json

    data = load_json(Path(path))
    if not {"ratio_definition", "cases"} <= set(data) or set(data) - {
        "ratio_definition",
        "cases",
        "context",
        "aggregation",
    }:
        raise ValueError("Unknown or missing reference fields")
    if data["ratio_definition"] not in DEFINITIONS:
        raise ValueError("Reference requires ratio_definition and cases")
    if not isinstance(data["cases"], dict) or set(data["cases"]) != set(WINDOWS):
        raise ValueError("Reference must contain exactly A, B, R, C and D")
    for case, value in data["cases"].items():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError("Invalid reported overlap percentage: " + case)
    if not isinstance(data.get("context", {}), dict):
        raise ValueError("Reference context must be an object")
    return dict(
        ratio_definition=data["ratio_definition"],
        cases=dict(data["cases"]),
        context=data.get("context", {}),
        aggregation=data.get("aggregation", "unspecified"),
        role="reported_temporal_observation_not_speedup_or_acceptance_threshold",
    )


def comparison(reference, case, measurement=None, configuration=None):
    if reference is None:
        return None
    result = dict(
        reported_overlap_percent=reference["cases"][case],
        ratio_definition=reference["ratio_definition"],
        status="requires_device_trace_and_matching_workload",
    )
    definition = reference["ratio_definition"]
    if definition == "unspecified":
        result["status"] = "definition_unconfirmed"
    elif measurement is not None:
        field = {
            "communication_time": "communication_coverage_percent",
            "computation_time": "computation_coverage_percent",
            "window_time": "window_overlap_percent",
        }[definition]
        measured = measurement[field]
        result.update(
            status="numerical_difference_only_workload_alignment_not_proven",
            measured_overlap_percent=measured,
            difference_percentage_points=measured - reference["cases"][case],
        )
    if configuration is not None:
        differences = {
            key: dict(reported=value, benchmark=configuration[key])
            for key, value in reference.get("context", {}).items()
            if key in configuration and configuration[key] != value
        }
        if differences:
            result.update(status="workload_mismatch", configuration_differences=differences)
    return result
