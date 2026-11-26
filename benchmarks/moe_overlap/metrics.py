"""Independent repeat means, slowest rank per sample, explicit uncertainty."""

import math
import statistics as st


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


def summarize(reports, margin=0):
    if len(reports) != 4 or {r["rank"] for r in reports} != {0, 1, 2, 3}:
        raise ValueError("Exactly four rank reports required")
    first = reports[0]
    cfg = first["config"]
    for r in reports:
        if r["config"] != cfg or not r["precision_passed"] or not r["destroyed"]:
            raise ValueError("Configuration/precision/lifecycle mismatch")
        for key in ("torch", "torch_npu", "benchmark_sha256", "deepep_artifact_sha256"):
            if r["environment"].get(key) != first["environment"].get(key):
                raise ValueError("Rank implementation mismatch: " + key)
        if r.get("case", "all") != first.get("case", "all"):
            raise ValueError("Rank case mismatch")
    if first.get("case") in ("A", "D"):
        case = first["case"]
        controls = {}
        for control in ("comm_only", "compute_only", "serial", "overlap"):
            reps = []
            values = []
            for rep in range(cfg["repeats"]):
                arrays = [r["windows"][case][control][rep] for r in reports]
                if any(len(a) != cfg["iterations"] for a in arrays):
                    raise ValueError("Wrong window sample count")
                v = [max(a[i] for a in arrays) for i in range(cfg["iterations"])]
                stats(v)
                values.extend(v)
                reps.append(st.mean(v))
            controls[control] = dict(**stats(values), repeat_means=reps)
        return dict(
            scope="real_NPU_EP4_MoE_window_" + case,
            config=cfg,
            precision_passed=True,
            environment=first["environment"],
            rank_environments=[r["environment"] for r in reports],
            window_summary={
                case: dict(
                    controls=controls,
                    overlap_benefit=paired(
                        controls["serial"]["repeat_means"], controls["overlap"]["repeat_means"], margin
                    ),
                )
            },
            units="us",
            method="per-sample slowest rank; independent repeat means",
        )
    expected = {(m, rep) for m in ("serial", "d_only", "a_only", "ad") for rep in range(cfg["repeats"])}
    maps = []
    for r in reports:
        ix = {(t["mode"], t["repeat"]): t for t in r["trials"]}
        if set(ix) != expected or len(ix) != len(r["trials"]):
            raise ValueError("Missing/duplicate mode/repeat")
        if any(len(t["samples"]) != cfg["iterations"] for t in ix.values()):
            raise ValueError("Wrong sample count")
        maps.append(ix)
    rows = {}
    means = {}
    for mode in ("serial", "d_only", "a_only", "ad"):
        rows[mode] = {}
        means[mode] = {}
        for metric in ("a_window_us", "d_window_us", "total_us"):
            reps = []
            all_samples = []
            for rep in range(cfg["repeats"]):
                values = [max(ix[(mode, rep)]["samples"][i][metric] for ix in maps) for i in range(cfg["iterations"])]
                stats(values)
                reps.append(st.mean(values))
                all_samples.extend(values)
            rows[mode][metric] = dict(**stats(all_samples), repeat_means=reps)
            means[mode][metric] = reps
    comparisons = {}
    for mode in ("d_only", "a_only", "ad"):
        comparisons[mode] = {
            metric: paired(means["serial"][metric], means[mode][metric], margin) for metric in means[mode]
        }
    windows = {}
    present = [bool(r["windows"]) for r in reports]
    if any(present) and not all(present):
        raise ValueError("Incomplete window reports")
    if all(present):
        for case in ("A", "D"):
            rows_w = {}
            for control in ("comm_only", "compute_only", "serial", "overlap"):
                reps = []
                values = []
                for rep in range(cfg["repeats"]):
                    arrays = [r["windows"][case][control][rep] for r in reports]
                    if any(len(a) != cfg["iterations"] for a in arrays):
                        raise ValueError("Wrong window sample count")
                    v = [max(a[i] for a in arrays) for i in range(cfg["iterations"])]
                    stats(v)
                    values.extend(v)
                    reps.append(st.mean(v))
                rows_w[control] = dict(**stats(values), repeat_means=reps)
            windows[case] = dict(
                controls=rows_w,
                overlap_benefit=paired(rows_w["serial"]["repeat_means"], rows_w["overlap"]["repeat_means"], margin),
            )
    return dict(
        scope="real_NPU_EP4_single_MoE_layer_A_D",
        config=cfg,
        method="max over ranks for each sample; CI uses independent repeat means",
        units="us",
        modes=rows,
        overlap_benefit=comparisons,
        precision_passed=True,
        environment=first["environment"],
        rank_environments=[r["environment"] for r in reports],
        window_summary=windows,
        windows={str(r["rank"]): r["windows"] for r in reports},
    )


def regression(baseline, candidate, margin=0):
    if margin < 0 or not math.isfinite(margin):
        raise ValueError("Invalid regression margin")
    required_scope = "real_NPU_EP4_single_MoE_layer_A_D"
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
    for mode in ("serial", "d_only", "a_only", "ad"):
        result[mode] = {}
        for metric in ("total_us", "a_window_us", "d_window_us"):
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
