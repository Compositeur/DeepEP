"""Hardware execution. No remote access, installation, or device selection automation."""

import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from .scenarios import WINDOWS, CONTROLS, modes_for_suite, layer_scope
from .oracle import sampled_reference, reference_errors


def fingerprint():
    import torch
    import torch_npu
    import deep_ep

    root = Path(deep_ep.__file__).resolve().parent
    files = {root / "__init__.py", root / "buffer.py", *root.rglob("*.so")}
    distribution_version = None
    # Native extensions may live in sibling ascend_deepep, not deep_ep.
    try:
        distribution = md.distribution("ascend-deepep")
        distribution_version = distribution.version
        for item in distribution.files or ():
            if str(item).endswith(".so"):
                files.add(Path(distribution.locate_file(item)).resolve())
    except md.PackageNotFoundError:
        pass
    hashes = {}
    for p in sorted(files):
        if not p.is_file():
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        key = (
            str(p.relative_to(root))
            if p.is_relative_to(root)
            else "distribution/" + str(p.relative_to(root.parent))
            if p.is_relative_to(root.parent)
            else str(p)
        )
        hashes[key] = h.hexdigest()
    revision = None
    for p in root.parents:
        if (p / ".git").exists():
            result = subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"], capture_output=True, text=True)
            revision = result.stdout.strip() if result.returncode == 0 else None
            break

    def read(path):
        try:
            return Path(path).read_text()
        except OSError:
            return None

    cann = read("/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascend_toolkit_install.info")
    code = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        code.update(p.name.encode() + p.read_bytes())
    return dict(
        torch=torch.__version__,
        torch_npu=getattr(torch_npu, "__version__", "unknown"),
        python=sys.version.split()[0],
        platform=platform.platform(),
        device_names=[torch.npu.get_device_name(i) for i in range(torch.npu.device_count())],
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        driver=read("/usr/local/Ascend/driver/version.info"),
        cann=cann,
        task_queue=os.environ.get("TASK_QUEUE_ENABLE"),
        cpu_affinity=os.environ.get("CPU_AFFINITY_CONF"),
        deepep_path=str(root),
        deepep_git_revision=revision,
        deepep_artifact_sha256=hashes,
        deepep_distribution_version=distribution_version,
        benchmark_sha256=code.hexdigest(),
    )


def run(c, out, margin=0, windows=True, baseline=None, case="all", reference=None):
    import datetime
    import torch
    import torch_npu
    import torch.distributed as dist
    from .adapters import DeepEP
    from .layer import Layer, cpu_check
    from .metrics import summarize, regression

    c.validate()
    mode_spec = modes_for_suite(c.layer_suite)
    window_cases = list(WINDOWS) if case == "all" else [case] if case in WINDOWS else []
    if not windows or case == "layer":
        window_cases = []
    rank = int(os.environ.get("RANK", "-1"))
    local = int(os.environ.get("LOCAL_RANK", "-1"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", c.ep))
    if int(os.environ.get("WORLD_SIZE", 0)) != c.ep or not 0 <= local < local_world or not 0 <= rank < c.ep:
        raise ValueError("Use torchrun with WORLD_SIZE equal to configured EP and valid local rank")
    torch.set_num_threads(1)
    torch.npu.set_device(local)
    if torch.npu.device_count() != local_world or not torch.npu.is_available():
        raise RuntimeError("Visible NPUs must match LOCAL_WORLD_SIZE and be usable")
    dest = out / f"rank{rank}"
    dest.mkdir(parents=True, exist_ok=False)
    adapter = None
    initialized = False
    report = dict(
        rank=rank,
        world_size=c.ep,
        config=c.dictionary(),
        precision_passed=False,
        destroyed=False,
        trials=[],
        windows={},
        case=case,
        scope=f"real_NPU_EP{c.ep}_MoE_window_" + case if case in WINDOWS else layer_scope(c.layer_suite, c.ep),
        window_cases=window_cases,
        reported_overlap_reference=reference,
        scenarios={key: WINDOWS[key] for key in window_cases},
    )

    def save():
        (dest / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    try:
        report["environment"] = fingerprint()
        save()
        if c.wgrad_backend == "pointer":
            import importlib

            try:
                module = importlib.import_module("pointer_gmm")
            except ImportError as exc:
                raise RuntimeError("pointer_gmm explicitly requested but unavailable") from exc
            if not callable(getattr(module, "npu_grouped_matmul_add_ptr_", None)):
                raise RuntimeError("pointer_gmm interface is incompatible")
        dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=180))
        initialized = True
        adapter = DeepEP(c, dist.group.WORLD)
        report["resolved_buffer_class"] = type(adapter.buffer).__module__ + "." + type(adapter.buffer).__name__
        layer = Layer(c, adapter, rank, npu=True)
        report["cpu_oracle"] = cpu_check(c)
        source = sampled_reference(layer, rank)
        fp32_source = source if c.source_reference_dtype == "fp32" else sampled_reference(layer, rank, "fp32")

        def prepare():
            layer.reset()
            torch.npu.synchronize()
            dist.barrier()

        def prepare_case(window_case):
            # Refresh cached handles after other cases performed fresh dispatch.
            # Fixture creation is excluded from every measured/profiler window.
            prepare()
            layer.step()
            layer.prepare_windows([window_case])
            torch.npu.synchronize()
            dist.barrier()

        precision_reference = None
        with torch.no_grad():
            for mode, (a, cached, d) in mode_spec.items():
                prepare()
                actual, _ = layer.step(a, d, overlap_c=cached)
                if precision_reference is None:
                    precision_reference = {k: v.clone() for k, v in actual.items()}
                    report["communication_core_counts"] = adapter.core_counts
                    report["source_reference"] = dict(dtype=c.source_reference_dtype, rtol=0.06, atol=0.002)
                    report["source_reference_diagnostic"] = reference_errors(actual, source)
                    report["fp32_reference_diagnostic"] = reference_errors(actual, fp32_source)
                    report["fp32_reference_diagnostic"]["gating"] = c.source_reference_dtype == "fp32"
                    save()
                    for token, y, dx in source:
                        torch.testing.assert_close(actual["Y"][token].cpu().float(), y, rtol=0.06, atol=0.002)
                        torch.testing.assert_close(actual["dX"][token].cpu().float(), dx, rtol=0.06, atol=0.002)
                    payload, ctx, wctx, counts = layer.last_d
                    x, dpre, act, dz = wctx
                    offset = 0
                    for e, n in enumerate(counts.cpu().tolist()):
                        if e in {0, len(counts) // 2, len(counts) - 1}:
                            for lhs, rhs, target in [(x, dpre, layer.g1), (act, dz, layer.g2)]:
                                gold = (
                                    lhs[offset : offset + n, :4].cpu().float().t()
                                    @ rhs[offset : offset + n, :4].cpu().float()
                                )
                                if c.wgrad_backend == "gmm_add":
                                    gold = gold.bfloat16().float()
                                torch.testing.assert_close(target[e, :4, :4].cpu(), gold, rtol=0.04, atol=0.002)
                        offset += n
                for key in precision_reference:
                    torch.testing.assert_close(
                        actual[key], precision_reference[key], rtol=c.precision_rtol, atol=c.precision_atol
                    )
                    if not bool(torch.isfinite(actual[key]).all().item()):
                        raise RuntimeError("Nonfinite " + key)
            del precision_reference
            layer.prepare_windows(window_cases)
            if "R" in window_cases:
                report["attention_proxy"] = dict(
                    tokens=min(c.tokens, c.attention_tokens),
                    heads=c.attention_heads,
                    head_dim=c.attention_head_dim,
                    implementation="eager_causal_QKV_softmax_output_projection",
                    matches_profile_MLA=False,
                )
            expected_y = actual["Y"].clone()

            def snapshot(value):
                if isinstance(value, torch.Tensor):
                    return value.clone()
                if isinstance(value, tuple):
                    return tuple(snapshot(item) for item in value)
                if isinstance(value, dict):
                    return {key: snapshot(item) for key, item in value.items()}
                return value

            for window_case in window_cases:
                prepare_case(window_case)
                expected = None
                for control in ("serial", "overlap"):
                    prepare()
                    checked = layer.window(window_case, control, timed=False)
                    torch.npu.synchronize()
                    if window_case in ("A", "R"):
                        torch.testing.assert_close(
                            checked["communication_output"],
                            layer.data["x"],
                            rtol=c.precision_rtol,
                            atol=c.precision_atol,
                        )
                    if window_case == "B":
                        torch.testing.assert_close(
                            checked["compute_output"], expected_y, rtol=c.precision_rtol, atol=c.precision_atol
                        )
                    if expected is None:
                        expected = snapshot(checked)
                    torch.testing.assert_close(checked, expected, rtol=c.precision_rtol, atol=c.precision_atol)
            report["window_precision_checked"] = window_cases
            report["precision_passed"] = True
            report["precision_scope"] = (
                "independent tiny CPU autograd; sampled source Y/dX using "
                + c.source_reference_dtype
                + " operator arithmetic; sampled local dW; all six outputs cross-mode; frozen gates"
            )
            save()
            for rep in range(c.repeats if case in ("all", "layer") else 0):
                modes = list(mode_spec)
                modes = modes[rep % len(modes) :] + modes[: rep % len(modes)]
                if rep % 2:
                    modes.reverse()
                for mode in modes:
                    a, cached, d = mode_spec[mode]
                    for _ in range(c.warmup):
                        prepare()
                        layer.step(a, d, True, overlap_c=cached)
                    samples = []
                    for _ in range(c.iterations):
                        prepare()
                        _, timing = layer.step(a, d, True, overlap_c=cached)
                        samples.append(timing)
                    report["trials"].append(dict(mode=mode, repeat=rep, samples=samples))
                    save()
            for window_case in window_cases:
                prepare_case(window_case)
                report["windows"][window_case] = {control: [] for control in CONTROLS}
                for rep in range(c.repeats):
                    controls = list(CONTROLS)
                    controls = controls[rep % 4 :] + controls[: rep % 4]
                    if rep % 2:
                        controls.reverse()
                    for control in controls:
                        for _ in range(c.warmup):
                            prepare()
                            layer.window(window_case, control)
                        values = []
                        for _ in range(c.iterations):
                            prepare()
                            with torch.profiler.record_function(f"moe/window/{window_case}/{control}"):
                                values.append(layer.window(window_case, control))
                        report["windows"][window_case][control].append(values)
                    save()
            if c.profile_iterations:
                profile_cases = ["layer", *window_cases] if case == "all" else [case]
                for profile_case in profile_cases:
                    if profile_case in WINDOWS:
                        prepare_case(profile_case)
                    modes = (
                        ("serial", "overlap")
                        if profile_case in WINDOWS
                        else ("serial", "acd" if c.layer_suite == "ACD" else "ad")
                    )
                    for mode in modes:
                        with torch_npu.profiler.profile(
                            activities=[
                                torch_npu.profiler.ProfilerActivity.CPU,
                                torch_npu.profiler.ProfilerActivity.NPU,
                            ],
                            record_shapes=True,
                            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                                str(dest / f"profile_{profile_case}_{mode}")
                            ),
                            experimental_config=torch_npu.profiler._ExperimentalConfig(
                                profiler_level=torch_npu.profiler.ProfilerLevel.Level1
                            ),
                        ) as prof:
                            for _ in range(c.profile_iterations):
                                prepare()
                                if profile_case in WINDOWS:
                                    with torch.profiler.record_function(f"moe/window/{profile_case}/{mode}"):
                                        layer.window(profile_case, mode)
                                else:
                                    a, cached, d = mode_spec[mode]
                                    layer.step(a, d, True, overlap_c=cached)
                                prof.step()
        torch.npu.synchronize()
        dist.barrier()
        adapter.destroy()
        report["destroyed"] = True
        save()
        gathered = [None] * c.ep
        dist.all_gather_object(gathered, report)
        if rank == 0:
            result = summarize(gathered, margin)
            (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            if baseline:
                comparison = regression(json.loads(baseline.read_text()), result, margin)
                (out / "regression.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2))
            print(
                json.dumps(result.get("overlap_benefit", result.get("window_summary")), ensure_ascii=False, indent=2),
                flush=True,
            )
        dist.barrier()
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        save()
        raise
    finally:
        if adapter is not None and not report["destroyed"]:
            # Device failure may require torchrun to terminate peers; retain first error.
            try:
                adapter.destroy()
            except Exception:
                pass
        if initialized:
            dist.destroy_process_group()
