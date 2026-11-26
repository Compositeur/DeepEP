"""Image preflight must diagnose software without invoking a device API."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.moe_overlap.config import Config
from benchmarks.moe_overlap import environment
from benchmarks.moe_overlap.environment import probe


class SoftwareProbeTests(unittest.TestCase):
    def modules(self):
        class Buffer:
            def __init__(self):
                pass

            def capture(self):
                pass

            def dispatch(self, previous_event=None, async_finish=False):
                pass

            def combine(self, previous_event=None, async_finish=False):
                pass

            def destroy(self):
                pass

        schema = "group_list split_item group_type group_list_type"
        torch = SimpleNamespace(
            __file__="/nonexistent/torch.py",
            ops=SimpleNamespace(
                npu=SimpleNamespace(npu_grouped_matmul=SimpleNamespace(default=SimpleNamespace(_schema=schema)))
            ),
        )
        return {
            "torch": torch,
            "torch_npu": SimpleNamespace(__file__="/nonexistent/npu.py"),
            "deep_ep": SimpleNamespace(__file__="/nonexistent/deep_ep/__init__.py"),
            "deep_ep.backends.direct.buffer": SimpleNamespace(Buffer=Buffer),
        }

    def run_probe(self, c, fail=None):
        modules = self.modules()

        def load(name):
            if name == fail:
                raise ImportError("missing libhccl.so")
            return modules[name]

        with (
            patch.object(environment.importlib.util, "find_spec", return_value=None),
            patch.object(environment.importlib, "import_module", side_effect=load),
        ):
            return probe(c)

    def test_gmm_does_not_require_pointer_or_device_api(self):
        r = self.run_probe(Config())
        self.assertTrue(r["software_checks_passed"])
        self.assertFalse(r["pointer_gmm"]["required"])
        self.assertFalse(r["NPU_operations_executed"])
        self.assertIn("NOT_PROVEN", r["revision_verification"])

    def test_pointer_is_required_only_when_selected(self):
        r = self.run_probe(Config(wgrad_backend="pointer"))
        self.assertFalse(r["software_checks_passed"])
        self.assertFalse(r["checks"]["pointer_available"])

    def test_missing_cann_library_is_reported_as_failure(self):
        r = self.run_probe(Config(), fail="torch_npu")
        self.assertFalse(r["software_checks_passed"])
        self.assertIn("libhccl.so", r["imports"]["torch_npu"]["error"])
