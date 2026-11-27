"""Overlap is device interval intersection, never a latency-derived speedup."""

import json

import pytest

from benchmarks.moe_overlap.reference import load_reference, comparison
from benchmarks.moe_overlap.trace import analyze
from benchmarks.moe_overlap.trace_io import iter_trace


def event(name, start, duration, **kwargs):
    return dict(name=name, ts=start, dur=duration, pid=10, tid=1, ph="X", cat="", **kwargs)


def measured(events):
    return analyze(events, category="", pid=10, start=0, end=100, comm="comm", compute="compute")


def test_intersection_uses_unions_clipping_and_one_rank_only():
    events = [
        event("comm", -10, 50),
        event("comm", 20, 40),
        event("comm", 80, 30),
        event("compute", 30, 60),
        event("compute", 40, 10),
    ]
    other = event("compute", 0, 100)
    other["pid"] = 11
    events.append(other)
    host = event("compute", 0, 100)
    host["cat"] = "cpu_op"
    events.append(host)
    out = measured(events)
    assert out["communication_us"] == 80
    assert out["computation_us"] == 60
    assert out["overlap_us"] == 40
    assert out["communication_coverage_percent"] == 50
    assert out["exposed_communication_us"] == 40
    assert out["performance_benefit"] is None


def test_dependency_control_can_have_zero_overlap():
    out = measured([event("comm", 0, 20), event("compute", 20, 80)])
    assert out["communication_coverage_percent"] == 0
    assert out["overlap_us"] == 0


@pytest.mark.parametrize(
    "events", [[event("comm_compute", 0, 1)], [event("comm", 0, 10)], [event("comm", 0, float("nan"))]]
)
def test_fail_closed_for_ambiguous_missing_or_invalid_intervals(events):
    with pytest.raises(ValueError):
        measured(events)


def test_percentages_are_reference_annotations_not_speedup(tmp_path):
    path = tmp_path / "reference.json"
    data = dict(ratio_definition="unspecified", cases=dict(A=18.35, B=0, R=80, C=99.69, D=82.19))
    path.write_text(json.dumps(data))
    ref = load_reference(path)
    out = measured([event("comm", 0, 100), event("compute", 0, 50)])
    assert comparison(ref, "A", out)["status"] == "definition_unconfirmed"
    data["ratio_definition"] = "communication_time"
    path.write_text(json.dumps(data))
    ref = load_reference(path)
    compared = comparison(ref, "A", out)
    assert compared["difference_percentage_points"] == pytest.approx(31.65)
    assert "benefit_percent" not in compared
    data["cases"]["C"] = float("nan")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_reference(path)


@pytest.mark.parametrize("wrapped", [True, False])
def test_incremental_trace_crosses_buffers_and_ignores_header_string(tmp_path, wrapped):
    events = [event("comm", i, 1) for i in range(3000)]
    document = {"description": "traceEvents [ inside a string", "traceEvents": events} if wrapped else events
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(document))
    assert list(iter_trace(path)) == events


@pytest.mark.parametrize("text", ["[{},]", '[{"ph":"X"}', '{"traceEvents":[5]}'])
def test_truncated_or_invalid_trace_is_rejected(tmp_path, text):
    path = tmp_path / "trace.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        list(iter_trace(path))


def test_weighted_rank_aggregation_does_not_average_percentages():
    from benchmarks.moe_overlap.trace import aggregate

    first = measured([event("comm", 0, 80), event("compute", 40, 40)])
    second = measured([event("comm", 0, 10), event("compute", 0, 10)])
    first.update(case="A", rank=0)
    second.update(case="A", rank=8)
    result = aggregate([first, second])
    assert result["communication_coverage_percent"] == pytest.approx(100 * 50 / 90)
    assert result["communication_coverage_percent"] != 75
    assert result["performance_benefit"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([first, first])
    second["rank"] = None
    with pytest.raises(ValueError, match="rank"):
        aggregate([first, second])


def test_trace_cli_writes_temporal_report_with_rank(tmp_path):
    from benchmarks.moe_overlap.trace import main

    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps([event("comm", 0, 100), event("compute", 0, 20)]))
    main(
        [
            "--trace",
            str(trace),
            "--case",
            "A",
            "--rank",
            "0",
            "--device-pid",
            "10",
            "--category",
            "",
            "--start-us",
            "0",
            "--end-us",
            "100",
            "--comm",
            "comm",
            "--compute",
            "compute",
            "--workspace",
            str(tmp_path),
            "--out",
            "analysis",
        ]
    )
    result = json.loads((tmp_path / "analysis/temporal_overlap.json").read_text())
    assert result["communication_coverage_percent"] == 20
    assert result["rank"] == 0
    assert result["performance_benefit"] is None


def test_ascend_missing_category_and_string_timestamp():
    communication = event("comm", 0, 100)
    communication.pop("cat")
    communication["ts"] = "0.0"
    computation = event("compute", 0, 20)
    computation.pop("cat")
    assert measured([communication, computation])["communication_coverage_percent"] == 20


def test_host_timeline_cannot_be_reported_as_device_overlap():
    events = [event("comm", 0, 100), event("compute", 0, 20)]
    with pytest.raises(ValueError, match="Host"):
        analyze(events, category="cpu_op", pid=10, start=0, end=100, comm="comm", compute="compute")
    events.append(dict(ph="M", name="process_name", pid=10, args=dict(name="Python")))
    with pytest.raises(ValueError, match="not a device"):
        measured(events)
    events[-1]["args"]["name"] = "Ascend Hardware"
    assert measured(events)["device_process_verification"] == "profiler_metadata"
