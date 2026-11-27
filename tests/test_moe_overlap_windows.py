"""Scheduling, recomputation and workload-sized rank reduction checks."""

import copy
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks.moe_overlap.config import Config
from benchmarks.moe_overlap.metrics import summarize, regression
from benchmarks.moe_overlap.reference import load_reference
from benchmarks.moe_overlap.scenarios import WINDOWS, CONTROLS, modes_for_suite, layer_metrics, layer_scope


@pytest.fixture
def layer():
    pytest.importorskip("torch")
    from benchmarks.moe_overlap.adapters import LocalReference
    from benchmarks.moe_overlap.layer import Layer

    c = Config(
        tokens=3, hidden=4, experts=8, topk=2, width=3, shared_width=3, layer_suite="ACD", recompute_experts=True
    )
    return Layer(c, LocalReference(c.experts))


def test_acd_all_eight_modes_match_autograd(layer):
    import torch
    from benchmarks.moe_overlap.layer import autograd_reference

    oracle = autograd_reference(layer.data)
    assert len(modes_for_suite("ACD")) == 8
    for a, cached, d in modes_for_suite("ACD").values():
        layer.reset()
        with torch.no_grad():
            out, _ = layer.step(a, d, overlap_c=cached)
        torch.testing.assert_close(out, oracle, rtol=1e-10, atol=1e-12)


def test_cached_dispatch_wait_moves_across_only_independent_work(layer):
    import torch
    import benchmarks.moe_overlap.layer as module
    from benchmarks.moe_overlap.adapters import LocalReference

    events = []

    class Logging(LocalReference):
        def cached(self, *args):
            value, _ = super().cached(*args)
            events.append("cached")
            return value, "cached_event"

        def wait(self, event):
            if event == "cached_event":
                events.append("wait")

    layer.adapter = Logging(layer.c.experts)
    shared, expert = module.shared_backward, module.expert_forward

    def shared_call(*args):
        events.append("shared_backward")
        return shared(*args)

    def expert_call(*args):
        events.append("expert_forward")
        return expert(*args)

    for cached in (False, True):
        events.clear()
        layer.reset()
        with (
            torch.no_grad(),
            patch.object(module, "shared_backward", shared_call),
            patch.object(module, "expert_forward", expert_call),
        ):
            layer.step(overlap_c=cached)
        start = events.index("cached")
        assert events[start : start + 4] == (
            ["cached", "shared_backward", "expert_forward", "wait"]
            if cached
            else ["cached", "wait", "shared_backward", "expert_forward"]
        )


@pytest.mark.parametrize("case", WINDOWS)
def test_five_windows_serial_parallel_and_isolated_outputs(layer, case):
    import torch

    with torch.no_grad():
        layer.step()
        layer.prepare_windows([case])
        results = {}
        for control in CONTROLS:
            layer.reset()
            result = layer.window(case, control, timed=False)
            results[control] = copy.deepcopy(result)
        torch.testing.assert_close(results["serial"], results["overlap"], rtol=1e-10, atol=1e-12)
        assert results["comm_only"]["compute_output"] is None
        assert results["compute_only"]["communication_output"] is None
        torch.testing.assert_close(results["serial"]["compute_output"], results["compute_only"]["compute_output"])
        if case in ("A", "R"):
            torch.testing.assert_close(results["serial"]["communication_output"], layer.data["x"])


def test_forward_combine_cannot_read_result_before_wait(layer):
    import torch
    from benchmarks.moe_overlap.adapters import LocalReference

    with torch.no_grad():
        expected, _ = layer.step()
    state = {"ready": False}

    class Guard:
        def __init__(self, value):
            self.value = value

        def __add__(self, other):
            assert state["ready"], "Read an unfinished combine result"
            return self.value + other

    class Checked(LocalReference):
        def combine(self, *args):
            out, _ = super().combine(*args)
            return Guard(out), object()

        def wait(self, event):
            state["ready"] = True

    layer.adapter = Checked(layer.c.experts)
    for control in ("serial", "overlap"):
        state["ready"] = False
        out = layer.window("B", control, timed=False)
        torch.testing.assert_close(out["compute_output"], expected["Y"])


def test_attention_proxy_matches_independent_sdpa(layer):
    import torch

    layer.prepare_windows(["R"])
    x, wqkv, wo, heads = layer.attention_fixture
    q, k, v = (x @ wqkv).chunk(3, -1)
    q, k, v = [part.reshape(x.shape[0], heads, -1).transpose(0, 1) for part in (q, k, v)]
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    expected = expected.transpose(0, 1).reshape(x.shape[0], -1) @ wo
    out = layer.window("R", "compute_only", timed=False)["compute_output"]
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-9)


def reports(ep=4, suite="ACD"):
    c = Config(
        tokens=2,
        hidden=4,
        experts=32,
        topk=2,
        width=3,
        shared_width=3,
        ep=ep,
        layer_suite=suite,
        iterations=2,
        repeats=3,
    ).dictionary()
    result = []
    for rank in range(ep):
        trials = [
            dict(
                mode=mode,
                repeat=rep,
                samples=[
                    {metric: (100 + rank + rep) * (1 if mode == "serial" else 0.9) for metric in layer_metrics(suite)}
                    for _ in range(2)
                ],
            )
            for mode in modes_for_suite(suite)
            for rep in range(3)
        ]
        windows = {
            case: {control: [[float(100 + rank)] * 2 for _ in range(3)] for control in CONTROLS} for case in WINDOWS
        }
        result.append(
            dict(
                rank=rank,
                world_size=ep,
                config=c,
                precision_passed=True,
                destroyed=True,
                trials=trials,
                windows=windows,
                window_cases=list(WINDOWS),
                case="all",
                environment=dict(torch="test", torch_npu="test", benchmark_sha256="test"),
            )
        )
    return result


@pytest.mark.parametrize("ep", [4, 16])
def test_acd_rank_aggregation_and_regression(ep):
    data = reports(ep)
    out = summarize(data)
    assert out["scope"] == layer_scope("ACD", ep)
    assert len(out["modes"]) == 8
    assert out["modes"]["serial"]["c_window_us"]["repeat_means"] == [99 + ep, 100 + ep, 101 + ep]
    assert set(out["window_summary"]) == set(WINDOWS)
    assert out["window_summary"]["C"]["temporal_overlap"] == dict(
        status="requires_device_trace", derived_from_timings=False
    )
    assert regression(out, out)["status"] == "compared"
    with pytest.raises(ValueError):
        summarize(data[:-1])


def test_ep4_is_not_calibrated_to_ep16_reported_overlap():
    data = reports()
    reference = load_reference(Path("benchmarks/moe_overlap/configs/reported_overlap.json"))
    for report in data:
        report["reported_overlap_reference"] = reference
    out = summarize(data)
    assert out["window_summary"]["C"]["reported_reference"]["status"] == "workload_mismatch"
    assert out["window_summary"]["C"]["overlap_benefit"]["benefit_percent"] == 0
    assert out["window_summary"]["C"]["reported_reference"]["reported_overlap_percent"] == 99.69
