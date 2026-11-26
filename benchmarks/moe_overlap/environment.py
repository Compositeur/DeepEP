"""Software import and API checks. Does not select a device or allocate NPU tensors."""

import importlib
import importlib.metadata as md
import importlib.util
import inspect
import json
from pathlib import Path

from .config import REFERENCE_DEEPEP_REVISION


def probe(c):
    result = dict(
        scope="software_import_and_API_only_not_NPU_validation",
        packages={},
        imports={},
        device_nodes=[str(p) for p in Path("/dev").glob("davinci[0-9]*")],
        NPU_operations_executed=False,
        api=c.api,
        reference_deepep_revision=REFERENCE_DEEPEP_REVISION,
        revision_verification="NOT_PROVEN_BY_IMPORT; retain wheel/source build evidence",
    )
    for name in ("torch", "torch-npu", "ascend-deepep"):
        try:
            result["packages"][name] = md.version(name)
        except md.PackageNotFoundError:
            result["packages"][name] = None
    modules = {}
    for name in ("torch", "torch_npu", "deep_ep"):
        try:
            module = importlib.import_module(name)
            modules[name] = module
            result["imports"][name] = dict(ok=True, path=module.__file__)
        except Exception as exc:
            result["imports"][name] = dict(ok=False, error_type=type(exc).__name__, error=str(exc))
    checks = {}
    if "torch_npu" in modules and "torch" in modules:
        try:
            schema = str(modules["torch"].ops.npu.npu_grouped_matmul.default._schema)
            result["gmm_schema"] = schema
            checks["gmm_keywords"] = all(
                k in schema for k in ("group_list", "split_item", "group_type", "group_list_type")
            )
        except Exception as exc:
            checks["gmm_keywords"] = False
            result["gmm_error"] = str(exc)
    if "deep_ep" in modules:
        root = Path(modules["deep_ep"].__file__).parent
        build_info = root / "_build_info.json"
        if build_info.is_file():
            try:
                result["deep_ep_build_info"] = json.loads(build_info.read_text())
            except (OSError, ValueError) as exc:
                result["build_info_error"] = str(exc)
        try:
            direct_url = md.distribution("ascend-deepep").read_text("direct_url.json")
            if direct_url:
                result["deep_ep_install_origin"] = json.loads(direct_url)
        except (md.PackageNotFoundError, OSError, ValueError):
            pass
        try:
            module = importlib.import_module(
                "deep_ep.backends.direct.buffer" if c.api == "v1_global_clos" else "deep_ep.backends.direct.elastic"
            )
            cls = getattr(module, "Buffer" if c.api == "v1_global_clos" else "ElasticBuffer")
            result["signatures"] = {
                key: str(inspect.signature(getattr(cls, key)))
                for key in ("__init__", "capture", "dispatch", "combine", "destroy")
            }
            dispatch = inspect.signature(cls.dispatch).parameters
            combine = inspect.signature(cls.combine).parameters
            needed = (
                {"previous_event", "async_finish"}
                if c.api == "v1_global_clos"
                else {"previous_event", "async_with_compute_stream", "do_expand", "do_cpu_sync"}
            )
            checks["dispatch_contract"] = needed <= set(dispatch)
            checks["combine_contract"] = {
                "previous_event",
                "async_finish" if c.api == "v1_global_clos" else "async_with_compute_stream",
            } <= set(combine)
        except Exception as exc:
            checks["api_contract"] = False
            result["api_error"] = str(exc)
    spec = importlib.util.find_spec("pointer_gmm")
    result["pointer_gmm"] = dict(
        required=c.wgrad_backend == "pointer", available=spec is not None, path=spec.origin if spec else None
    )
    if c.wgrad_backend == "pointer":
        checks["pointer_available"] = spec is not None
    result["checks"] = checks
    result["software_checks_passed"] = (
        all(r["ok"] for r in result["imports"].values()) and bool(checks) and all(checks.values())
    )
    result["NPU_validation"] = "NOT_RUN; visible device nodes alone would not prove usability"
    return result
