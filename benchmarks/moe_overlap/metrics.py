"""Independent repeat means, slowest rank per sample, explicit uncertainty."""

import math
import statistics as st

from .scenarios import WINDOWS, CONTROLS, modes_for_suite, layer_metrics, layer_scope
from .reference import comparison


def stats(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Timing samples must be finite and positive")
    return dict(
        mean=st.mean(values), median=st.median(values), stdev=st.stdev(values) if len(values) > 1 else 0, n=len(values)
    )


def critical(df):
    # Conservative two-sided 95% t values, using the smaller available df.
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        15: 2.132,
        20: 2.086,
        30: 2.042,
        60: 2.0,
    }
    if df < 1:
        raise ValueError("Need independent repeats")
    return next((table[k] for k in reversed(sorted(table)) if k <= df), 12.706)


def verdict(mean, half, margin):
    if mean - half > margin:
        return "speedup"
    if mean + half < -margin:
        return "regression"
    if margin > 0 and abs(mean) + half <= margin:
        return "equivalent"
    if mean == half == 0:
        return "unchanged"
    return "inconclusive"


def paired(baseline, candidate, margin=0):
    if len(baseline) != len(candidate) or len(baseline) < 2:
        raise ValueError("Need paired independent repeats")
    stats(baseline)
    stats(candidate)
    gains = [100 * (a - b) / a for a, b in zip(baseline, candidate)]
    mean = st.mean(gains)
    half = critical(len(gains) - 1) * st.stdev(gains) / math.sqrt(len(gains))
    return dict(
        benefit_percent=mean,
        ci95=[mean - half, mean + half],
        independent_repeats=len(gains),
        margin_percent=margin,
        conclusion=verdict(mean, half, margin),
    )


def window_summary(reports, case, cfg, margin, reference):
    controls = {}
    for control in CONTROLS:
        repetitions = [r["windows"][case][control] for r in reports]
        if any(len(reps) != cfg["repeats"] for reps in repetitions):
            raise ValueError("Wrong window repetition count")
        means, values = [], []
        for rep in range(cfg["repeats"]):
            arrays = [reps[rep] for reps in repetitions]
            if any(len(a) != cfg["iterations"] for a in arrays):
                raise ValueError("Wrong window sample count")
            samples = [max(a[i] for a in arrays) for i in range(cfg["iterations"])]
            stats(samples)
            values.extend(samples)
            means.append(st.mean(samples))
        controls[control] = dict(**stats(values), repeat_means=means)
    return dict(
        controls=controls,
        overlap_benefit=paired(controls["serial"]["repeat_means"], controls["overlap"]["repeat_means"], margin),
        benefit_kind="measured_window_time_reduction_not_temporal_overlap",
        temporal_overlap=dict(status="requires_device_trace", derived_from_timings=False),
        scenario=WINDOWS[case],
        reported_reference=comparison(reference, case, configuration=cfg),
    )


def summarize(reports, margin=0):
    if not reports:
        raise ValueError("Rank reports are required")
    first = reports[0]
    cfg = first["config"]
    ep = cfg["ep"]
    if ep not in (4, 16) or len(reports) != ep or {r["rank"] for r in reports} != set(range(ep)):
        raise ValueError("Exactly the configured EP group's rank reports are required")
    reference = first.get("reported_overlap_reference")
    for r in reports:
        if r["config"] != cfg or not r["precision_passed"] or not r["destroyed"]:
            raise ValueError("Configuration/precision/lifecycle mismatch")
        for key in ("torch", "torch_npu", "benchmark_sha256", "deepep_artifact_sha256"):
            if r["environment"].get(key) != first["environment"].get(key):
                raise ValueError("Rank implementation mismatch: " + key)
        if r.get("case", "all") != first.get("case", "all"):
            raise ValueError("Rank case mismatch")
        if r.get("reported_overlap_reference") != reference:
            raise ValueError("Rank reference mismatch")
    common = dict(
        config=cfg,
        precision_passed=True,
        environment=first["environment"],
        rank_environments=[r["environment"] for r in reports],
        units="us",
        method="per-sample slowest rank; CI uses independent repeat means",
        reported_overlap_reference=reference,
    )
    if first.get("case") in WINDOWS:
        case = first["case"]
        return dict(
            scope=f"real_NPU_EP{ep}_MoE_window_" + case,
            **common,
            window_summary={case: window_summary(reports, case, cfg, margin, reference)},
        )
    suite = cfg.get("layer_suite", "AD")
    modes = modes_for_suite(suite)
    expected = {(mode, rep) for mode in modes for rep in range(cfg["repeats"])}
    maps = []
    for r in reports:
        indexed = {(t["mode"], t["repeat"]): t for t in r["trials"]}
        if set(indexed) != expected or len(indexed) != len(r["trials"]):
            raise ValueError("Missing/duplicate mode/repeat")
        if any(len(t["samples"]) != cfg["iterations"] for t in indexed.values()):
            raise ValueError("Wrong sample count")
        maps.append(indexed)
    rows = {}
    for mode in modes:
        rows[mode] = {}
        for metric in layer_metrics(suite):
            means, values = [], []
            for rep in range(cfg["repeats"]):
                samples = [max(ix[(mode, rep)]["samples"][i][metric] for ix in maps) for i in range(cfg["iterations"])]
                stats(samples)
                means.append(st.mean(samples))
                values.extend(samples)
            rows[mode][metric] = dict(**stats(values), repeat_means=means)
    benefits = {
        mode: {
            metric: paired(rows["serial"][metric]["repeat_means"], rows[mode][metric]["repeat_means"], margin)
            for metric in layer_metrics(suite)
        }
        for mode in modes
        if mode != "serial"
    }
    cases = set(first["windows"])
    if not cases <= set(WINDOWS):
        raise ValueError("Unknown window report")
    for r in reports:
        if set(r["windows"]) != cases or set(r.get("window_cases", cases)) != cases:
            raise ValueError("Incomplete window reports")
    return dict(
        scope=layer_scope(suite, ep),
        **common,
        modes=rows,
        overlap_benefit=benefits,
        benefit_kind="measured_layer_or_window_time_reduction_not_temporal_overlap",
        window_summary={
            case: window_summary(reports, case, cfg, margin, reference) for case in WINDOWS if case in cases
        },
    )


def regression(baseline, candidate, margin=0):
    if margin < 0 or not math.isfinite(margin):
        raise ValueError("Invalid regression margin")
    required_scope = layer_scope(
        baseline.get("config", {}).get("layer_suite", "AD"), baseline.get("config", {}).get("ep", 4)
    )
    if baseline.get("scope") != required_scope or candidate.get("scope") != required_scope:
        return dict(status="invalid", reason="Only real NPU layer reports may establish performance baselines")
    # Sample counts may grow; workload, API/backend, hardware and framework must match.
    ignored = {"iterations", "repeats", "warmup", "profile_iterations"}

    def signature(report):
        return {key: value for key, value in report["config"].items() if key not in ignored}

    if signature(baseline) != signature(candidate):
        return dict(status="invalid", reason="Workload/API/backend changed")
    for key in (
        "torch",
        "torch_npu",
        "device_names",
        "driver",
        "cann",
        "task_queue",
        "cpu_affinity",
        "benchmark_sha256",
    ):
        if baseline["environment"].get(key) != candidate["environment"].get(key):
            return dict(status="invalid", reason="Environment changed: " + key)
    if not baseline.get("precision_passed") or not candidate.get("precision_passed"):
        return dict(status="invalid", reason="Correctness did not pass")
    result = {}
    for mode in modes_for_suite(baseline["config"].get("layer_suite", "AD")):
        result[mode] = {}
        for metric in layer_metrics(baseline["config"].get("layer_suite", "AD")):
            a = baseline["modes"][mode][metric]["repeat_means"]
            b = candidate["modes"][mode][metric]["repeat_means"]
            stats(a)
            stats(b)
            if min(len(a), len(b)) < 2:
                raise ValueError("Need independent repeat samples")
            scale = st.mean(a)
            mean = 100 * (scale - st.mean(b)) / scale
            half = (
                critical(min(len(a), len(b)) - 1)
                * math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
                * 100
                / scale
            )
            result[mode][metric] = dict(
                benefit_percent=mean, ci95=[mean - half, mean + half], conclusion=verdict(mean, half, margin)
            )
    return dict(
        status="compared",
        margin_percent=margin,
        results=result,
        scope="same-mode code regression, not operator replacement or overlap benefit",
    )
