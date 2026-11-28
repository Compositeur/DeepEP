import copy
import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

from benchmarks.moe_overlap.config import Config, MODES, plan
from benchmarks.moe_overlap.metrics import paired, regression, summarize


class ConfigurationTests(unittest.TestCase):
    def test_default_is_gmm_and_separate_api_contracts(self):
        self.assertEqual(plan(Config())["configuration"]["wgrad_backend"], "gmm_add")
        self.assertEqual(len(plan(Config())["modes"]), 4)
        with self.assertRaises(ValueError):
            Config(api="v2_elastic").validate()
        Config(api="v2_elastic", num_sms=0).validate()
        with self.assertRaises(ValueError):
            Config(ep=8).validate()


class PerformanceTests(unittest.TestCase):
    def reports(self):
        cfg = Config(
            tokens=2, hidden=4, experts=8, topk=2, width=3, shared_width=3, iterations=2, repeats=3
        ).dictionary()
        env = dict(
            torch="test",
            torch_npu="test",
            device_names=["test"] * 4,
            driver="test",
            cann="test",
            task_queue=None,
            cpu_affinity=None,
            benchmark_sha256="test",
        )
        result = []
        for rank in range(4):
            trials = []
            for mode in MODES:
                for rep in range(3):
                    scale = 1 if mode == "serial" else 0.8
                    samples = [
                        {m: (100 + rank + rep) * scale for m in ["total_us", "a_window_us", "d_window_us"]}
                        for _ in range(2)
                    ]
                    trials.append(dict(mode=mode, repeat=rep, samples=samples))
            result.append(
                dict(
                    rank=rank,
                    config=cfg,
                    precision_passed=True,
                    destroyed=True,
                    trials=trials,
                    windows={},
                    environment=env,
                )
            )
        return result

    def test_slowest_rank_and_same_backend_overlap(self):
        s = summarize(self.reports())
        self.assertEqual(s["modes"]["serial"]["total_us"]["repeat_means"], [103, 104, 105])
        self.assertAlmostEqual(s["overlap_benefit"]["ad"]["total_us"]["benefit_percent"], 20)
        self.assertEqual(s["overlap_benefit"]["ad"]["total_us"]["conclusion"], "speedup")
        r = self.reports()
        r[3]["trials"].pop()
        with self.assertRaises(ValueError):
            summarize(r)

    def test_regression_rejects_estimates_api_and_environment_changes(self):
        a = summarize(self.reports())
        b = copy.deepcopy(a)
        self.assertEqual(regression({"scope": "CPU"}, b)["status"], "invalid")
        b["config"]["api"] = "v2_elastic"
        self.assertEqual(regression(a, b)["status"], "invalid")
        b = copy.deepcopy(a)
        b["environment"]["torch"] = "other"
        self.assertEqual(regression(a, b)["status"], "invalid")
        b = copy.deepcopy(a)
        for row in b["modes"].values():
            for item in row.values():
                item["repeat_means"] = [v * 1.5 for v in item["repeat_means"]]
        self.assertEqual(regression(a, b)["results"]["ad"]["total_us"]["conclusion"], "regression")

    def test_noise_not_forced_to_positive_gain(self):
        self.assertEqual(paired([100] * 3, [100, 101, 99])["conclusion"], "inconclusive")
        self.assertEqual(paired([100] * 3, [100] * 3)["conclusion"], "unchanged")
        with self.assertRaises(ValueError):
            paired([1], [1])

    def test_window_rank_aggregation_and_precision_failure(self):
        reports = self.reports()
        for r in reports:
            r["windows"] = {
                case: {
                    control: [[100 + r["rank"] - (10 if control == "overlap" else 0)] * 2 for _ in range(3)]
                    for control in ["comm_only", "compute_only", "serial", "overlap"]
                }
                for case in ["A", "D"]
            }
        s = summarize(reports)
        self.assertEqual(s["window_summary"]["D"]["controls"]["serial"]["mean"], 103)
        self.assertEqual(s["window_summary"]["D"]["overlap_benefit"]["conclusion"], "speedup")
        for r in reports:
            r["case"] = "D"
        partial = summarize(reports)
        self.assertEqual(partial["scope"], "real_NPU_EP4_MoE_window_D")
        self.assertEqual(regression(partial, partial)["status"], "invalid")
        reports[2]["precision_passed"] = False
        with self.assertRaises(ValueError):
            summarize(reports)

    def test_mixed_rank_code_is_rejected(self):
        reports = self.reports()
        reports[2]["environment"] = dict(reports[2]["environment"], benchmark_sha256="different")
        with self.assertRaisesRegex(ValueError, "Rank implementation mismatch"):
            summarize(reports)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires Torch for CPU math checks")
class RealDataflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from benchmarks.moe_overlap.adapters import LocalReference
        from benchmarks.moe_overlap.layer import Layer, autograd_reference

        cls.torch = torch
        cls.LocalReference = LocalReference
        cls.Layer = Layer
        cls.oracle = staticmethod(autograd_reference)

    def layer(self, clamp=0):
        c = Config(tokens=5, hidden=4, experts=4, topk=2, width=3, shared_width=3, clamp=clamp)
        return self.Layer(c, self.LocalReference(c.experts))

    def assert_reference(self, layer):
        reference = self.oracle(layer.data, layer.c.clamp)
        with self.torch.no_grad():
            for mode, (a, d) in MODES.items():
                layer.reset()
                out, _ = layer.step(a, d)
                for key in reference:
                    self.torch.testing.assert_close(
                        out[key], reference[key], rtol=1e-10, atol=1e-12, msg=lambda msg: f"{mode}/{key}: " + msg
                    )

    def test_full_output_input_and_all_weight_gradients(self):
        self.assert_reference(self.layer())

    def test_clamped_activation_backward(self):
        self.assert_reference(self.layer(0.001))

    def test_empty_experts_duplicate_routes_and_gradient_accumulation(self):
        layer = self.layer()
        layer.data["routes"].zero_()
        self.assert_reference(layer)
        layer.reset()
        with self.torch.no_grad():
            out, _ = layer.step(True, True)
            g1 = out["expert_dW1"].clone()
            g2 = out["expert_dW2"].clone()
            out, _ = layer.step(True, True)
        self.torch.testing.assert_close(out["expert_dW1"], 2 * g1)
        self.torch.testing.assert_close(out["expert_dW2"], 2 * g2)
        self.assertEqual(self.torch.count_nonzero(g1[1:]).item(), 0)

    def test_expert_output_really_affects_combine_and_input_gradient(self):
        layer = self.layer()
        with self.torch.no_grad():
            layer.reset()
            a, _ = layer.step()
            y = a["Y"].clone()
            dx = a["dX"].clone()
            layer.data["w2"].zero_()
            layer.data["w2t"].zero_()
            layer.reset()
            b, _ = layer.step()
        self.assertFalse(self.torch.allclose(y, b["Y"]))
        self.assertFalse(self.torch.allclose(dx, b["dX"]))

    def test_individual_switches_only_move_waits(self):
        import benchmarks.moe_overlap.layer as mod

        log = []

        class Logging(self.LocalReference):
            def dispatch(self, *args):
                log.append("dispatch")
                rx, ctx, _ = super().dispatch(*args)
                return rx, ctx, "event"

            def combine(self, *args):
                log.append("combine")
                y, _ = super().combine(*args)
                return y, "event"

            def wait(self, event):
                log.append("wait")

        layer = self.layer()
        layer.adapter = Logging(layer.c.experts)
        original_s, original_w = mod.shared_forward, mod.expert_wgrad

        def shared(*args):
            log.append("shared")
            return original_s(*args)

        def wgrad(*args):
            log.append("wgrad")
            return original_w(*args)

        for mode, (a, d) in MODES.items():
            log.clear()
            layer.reset()
            with (
                self.torch.no_grad(),
                patch.object(mod, "shared_forward", shared),
                patch.object(mod, "expert_wgrad", wgrad),
            ):
                layer.step(a, d)
            self.assertEqual(log[:3], ["dispatch", "shared", "wait"] if a else ["dispatch", "wait", "shared"])
            last = len(log) - 1 - log[::-1].index("combine")
            self.assertEqual(log[last : last + 3], ["combine", "wgrad", "wait"] if d else ["combine", "wait", "wgrad"])

    def test_v1_factory_and_layout_do_not_require_pointer(self):
        from benchmarks.moe_overlap.adapters import DeepEP

        torch = self.torch
        calls = []

        class Event:
            def current_stream_wait(self):
                calls.append("wait")

        class Sizing:
            def get_nvl_buffer_size_hint(self, *a):
                return 1024

            def get_rdma_buffer_size_hint(self, *a):
                return 0

        class Buffer:
            @staticmethod
            def get_dispatch_config(ep):
                return Sizing()

            get_combine_config = get_dispatch_config

            @staticmethod
            def set_num_sms(n):
                calls.append(("sms", n))

            def __init__(self, *a, **kw):
                calls.append(kw)

            def capture(self):
                return Event()

            def get_dispatch_layout(self, *a, **kw):
                assert kw["allocate_on_comm_stream"] is True
                return None, None, None, None, Event()

            def dispatch(self, x, **kw):
                assert kw["allocate_on_comm_stream"] is True
                calls.append(kw)
                if "handle" in kw:
                    return x, None, None, None, None, Event()
                return x, kw["topk_idx"], kw["topk_weights"], [2, 2], object(), Event()

            def combine(self, x, handle, **kw):
                assert kw["allocate_on_comm_stream"] is True
                calls.append(kw)
                return x, None, Event()

            def destroy(self):
                calls.append("destroy")

        module = types.SimpleNamespace(Buffer=Buffer, ElasticBuffer=object)
        with patch.dict(sys.modules, {"deep_ep": module}):
            adapter = DeepEP(Config(tokens=2, hidden=4, experts=8, topk=2), object())
            x = torch.arange(8, dtype=torch.float64).reshape(2, 4)
            ids = torch.tensor([[0, 1], [1, 0]])
            gates = torch.full((2, 2), 0.5)
            rx, ctx, event = adapter.dispatch(x, ids, gates)
            adapter.wait(event)
            packed, g, c = adapter.unpack(rx, ctx)
            self.assertEqual(c.tolist(), [2, 2])
            y, event = adapter.combine(adapter.prepare_combine(packed * g[:, None], ctx), ctx)
            torch.testing.assert_close(y, x)
            adapter.destroy()
        self.assertEqual(calls[0]["machine_type"], "global_clos")
        self.assertNotIn("transport", calls[0])

    def test_v2_expanded_layout_cached_dispatch_and_ht_capacity(self):
        from benchmarks.moe_overlap.adapters import DeepEP

        torch = self.torch
        calls = []

        class Event:
            def current_stream_wait(self):
                pass

        class Elastic:
            def __init__(self, **kw):
                calls.append(kw)

            def capture(self):
                return Event()

            def dispatch(self, x, handle=None, **kw):
                assert kw["allocate_on_comm_stream"] is True
                assert kw["do_expand"] is True
                if handle is not None:
                    self.test_cached = kw
                    return x[handle.rows], None, None, handle, Event()
                routes = kw["topk_idx"]
                rows, slots = torch.nonzero(routes >= 0, as_tuple=True)
                ids = routes[rows, slots]
                ix = torch.argsort(ids, stable=True)
                rows, slots = rows[ix], slots[ix]
                h = types.SimpleNamespace(
                    rows=rows, num_recv_tokens_per_expert_list=[2, 2], num_sms=kw["num_sms"] or 32
                )
                return x[rows], None, kw["topk_weights"][rows, slots], h, Event()

            def combine(self, x, handle, **kw):
                assert kw["allocate_on_comm_stream"] is True
                y = torch.zeros((2, 4), dtype=x.dtype)
                y.index_add_(0, handle.rows, x)
                return y, None, Event()

            def destroy(self):
                pass

        with patch.dict(sys.modules, {"deep_ep": types.SimpleNamespace(Buffer=object, ElasticBuffer=Elastic)}):
            a = DeepEP(Config(tokens=2, hidden=4, experts=8, topk=2, api="v2_elastic", num_sms=0), object())
            x = torch.arange(8, dtype=torch.float64).reshape(2, 4)
            rx, ctx, event = a.dispatch(x, torch.tensor([[0, 1], [1, 0]]), torch.full((2, 2), 0.5))
            packed, g, counts = a.unpack(rx, ctx)
            y, _ = a.combine(a.prepare_combine(packed * g[:, None], ctx), ctx)
            torch.testing.assert_close(y, x)
            grad, _ = a.cached(x, ctx)
            torch.testing.assert_close(a.pack_gradient(grad, ctx), x[ctx.handle.rows])
            self.assertFalse(a.buffer.test_cached["do_cpu_sync"])
        self.assertGreater(calls[0]["num_max_tokens_per_rank"], 256)
