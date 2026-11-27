"""Measure temporal overlap in an explicitly selected single-rank device window."""

import argparse
import json
import math
from pathlib import Path
import re

from .cli import DEFAULT_WORKSPACE, output_path, load_json
from .reference import load_reference, comparison
from .scenarios import WINDOWS
from .trace_io import iter_trace


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def intersection(left, right):
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze(events, *, category, pid, start, end, comm, compute):
    if not all(math.isfinite(x) for x in (start, end)) or end <= start:
        raise ValueError("Require finite start < end in trace microseconds")
    if category in {"cpu_op", "enqueue", "dequeue", "python_function", "user_annotation"}:
        raise ValueError("Host scopes cannot establish device overlap")
    patterns = [re.compile(comm), re.compile(compute)]
    intervals = [[], []]
    counts = [0, 0]
    process_name = None
    for event in events:
        if event.get("ph") == "M" and event.get("name") == "process_name" and str(event.get("pid")) == str(pid):
            process_name = event.get("args", {}).get("name")
        if event.get("ph") != "X" or str(event.get("pid")) != str(pid) or (event.get("cat") or "") != category:
            continue
        timestamp, duration = float(event["ts"]), float(event["dur"])
        if not math.isfinite(timestamp) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid device interval")
        low, high = max(start, timestamp), min(end, timestamp + duration)
        if high <= low:
            continue
        matches = [bool(pattern.search(event.get("name", ""))) for pattern in patterns]
        if all(matches):
            raise ValueError("Ambiguous communication/compute selectors: " + event["name"])
        for index, matched in enumerate(matches):
            if matched:
                intervals[index].append((low, high))
                counts[index] += 1
    if not all(counts):
        raise ValueError("Both communication and compute must match device events")
    if process_name in {"Python", "CANN", "Communication", "Overlap Analysis", "AI Core Freq"}:
        raise ValueError("Selected process is not a device kernel timeline")
    comm_intervals, compute_intervals = map(union, intervals)
    communication_us = sum(b - a for a, b in comm_intervals)
    computation_us = sum(b - a for a, b in compute_intervals)
    overlap_us = intersection(comm_intervals, compute_intervals)
    return dict(
        scope="one_rank_device_temporal_overlap_not_performance_speedup",
        device_pid=str(pid),
        process_name=process_name,
        device_process_verification=(
            "profiler_metadata"
            if process_name and ("Ascend" in process_name or "GPU" in process_name)
            else "requires_reviewed_manual_selection"
        ),
        category=category,
        window_us=[start, end],
        communication_pattern=comm,
        computation_pattern=compute,
        communication_event_count=counts[0],
        computation_event_count=counts[1],
        communication_us=communication_us,
        computation_us=computation_us,
        overlap_us=overlap_us,
        exposed_communication_us=communication_us - overlap_us,
        communication_coverage_percent=100 * overlap_us / communication_us,
        computation_coverage_percent=100 * overlap_us / computation_us,
        window_overlap_percent=100 * overlap_us / (end - start),
        performance_benefit=None,
        limitation="Temporal intersection only; no critical-path or end-to-end speedup inference",
    )


def aggregate(reports):
    """Sum per-window durations; never intersect clocks from different ranks."""
    if not reports or len({row.get("case") for row in reports}) != 1:
        raise ValueError("Aggregate reports must have one common case")
    case = reports[0].get("case")
    if case not in WINDOWS:
        raise ValueError("Unknown aggregate case")
    identities = set()
    communication = computation = covered = window = 0.0
    selectors = None
    for row in reports:
        if row.get("scope") != "one_rank_device_temporal_overlap_not_performance_speedup":
            raise ValueError("Only separate device-window reports may be aggregated")
        if type(row.get("rank")) is not int or row["rank"] < 0:
            raise ValueError("Every window report needs an explicit nonnegative rank")
        identity = (row["rank"], row["device_pid"], tuple(row["window_us"]))
        if identity in identities:
            raise ValueError("Duplicate rank/window report")
        identities.add(identity)
        current = (row["communication_pattern"], row["computation_pattern"])
        if selectors is not None and current != selectors:
            raise ValueError("Window selectors must match for aggregation")
        selectors = current
        busy, calc, overlap = (row[key] for key in ("communication_us", "computation_us", "overlap_us"))
        duration = row["window_us"][1] - row["window_us"][0]
        if not all(type(x) in (int, float) and math.isfinite(x) for x in (busy, calc, overlap, duration)):
            raise ValueError("Invalid aggregate durations")
        if min(busy, calc, duration) <= 0 or not 0 <= overlap <= min(busy, calc) or max(busy, calc) > duration:
            raise ValueError("Impossible temporal overlap durations")
        communication += busy
        computation += calc
        covered += overlap
        window += duration
    return dict(
        scope="duration_weighted_temporal_overlap_not_speedup",
        case=case,
        samples=len(reports),
        ranks=sorted({r["rank"] for r in reports}),
        communication_us=communication,
        computation_us=computation,
        overlap_us=covered,
        exposed_communication_us=communication - covered,
        communication_coverage_percent=100 * covered / communication,
        computation_coverage_percent=100 * covered / computation,
        window_overlap_percent=100 * covered / window,
        performance_benefit=None,
        aggregation="sum of overlap durations / sum of communication durations; not mean of percentages",
        limitation="Summed rank/window accounting is not elapsed step time; workload alignment still required",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--trace", type=Path)
    mode.add_argument(
        "--aggregate", nargs="+", type=Path, help="Weight separate call/rank reports by communication time"
    )
    parser.add_argument("--case", choices=WINDOWS)
    parser.add_argument("--rank", type=int, help="Global rank label, required for later aggregation")
    parser.add_argument("--device-pid", help="Reviewed device process for ONE rank, not a host process")
    parser.add_argument("--category", help="Exact device category; empty string is valid")
    parser.add_argument("--start-us", type=float)
    parser.add_argument("--end-us", type=float)
    parser.add_argument("--comm", help="Reviewed communication kernel regex")
    parser.add_argument("--compute", help="Reviewed compute kernel regex")
    parser.add_argument("--reference-overlap", type=Path)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        out = output_path(args.workspace.resolve(), args.out)
        reference = load_reference(args.reference_overlap) if args.reference_overlap else None
        if args.aggregate:
            result = aggregate([load_json(path) for path in args.aggregate])
        else:
            if any(
                value is None
                for value in (
                    args.case,
                    args.device_pid,
                    args.category,
                    args.start_us,
                    args.end_us,
                    args.comm,
                    args.compute,
                )
            ):
                raise ValueError("Trace analysis requires case, device-pid, category, start/end and selectors")
            if args.rank is not None and args.rank < 0:
                raise ValueError("rank must be nonnegative")
            result = analyze(
                iter_trace(args.trace),
                category=args.category,
                pid=args.device_pid,
                start=args.start_us,
                end=args.end_us,
                comm=args.comm,
                compute=args.compute,
            )
            result.update(case=args.case, rank=args.rank)
        case = result["case"]
        result.update(scenario=WINDOWS[case], reported_reference=comparison(reference, case, result))
    except (OSError, ValueError, TypeError, re.error) as exc:
        parser.error(str(exc))
    out.mkdir(parents=True, exist_ok=False)
    (out / "temporal_overlap.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
