"""Hardware execution. No remote access, installation, or device selection automation."""

import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from .config import MODES


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


def sampled_reference(layer, rank):
    """Source Y/dX oracle does not use transport handles or received row mappings."""
    import torch

    c = layer.c
    p = layer.data

    def weights(e):
        a = torch.Generator().manual_seed(c.seed + 10000 + e)
        b = torch.Generator().manual_seed(c.seed + 20000 + e)
        return (
            (torch.randn((c.hidden, 2 * c.width), generator=a) * 0.1).bfloat16().float(),
            (torch.randn((c.width, c.hidden), generator=b) * 0.1).bfloat16().float(),
        )

    source = []
    for token in sorted({0, c.tokens // 2, c.tokens - 1}):
        x = p["x"][token].cpu().float()
        dy = p["dy"][token].cpu().float()
        y = torch.zeros(c.hidden)
        dx = torch.zeros(c.hidden)
        for e, gate in zip(p["routes"][token].cpu().tolist(), p["gates"][token].cpu().tolist()):
            w1, w2 = weights(e)
            # Independent autograd graph, FP32 accumulation with BF16 inputs/weights.
            xx = x.clone().requires_grad_()
            pre = xx @ w1
            left, right = pre.chunk(2)
            if c.clamp:
                left, right = left.clamp(-c.clamp, c.clamp), right.clamp(-c.clamp, c.clamp)
            yy = (torch.nn.functional.silu(left) * right) @ w2 * gate
            dd = torch.autograd.grad((yy * dy).sum(), xx)[0]
            y += yy.detach()
            dx += dd.detach()
        sw1, sw2 = p["sw1"].cpu().float(), p["sw2"].cpu().float()
        xx = x.clone().requires_grad_()
        left, right = (xx @ sw1).chunk(2)
        if c.clamp:
            left, right = left.clamp(-c.clamp, c.clamp), right.clamp(-c.clamp, c.clamp)
        yy = (torch.nn.functional.silu(left) * right) @ sw2
        dx += torch.autograd.grad((yy * dy).sum(), xx)[0]
        y += yy.detach()
        source.append((token, y, dx))
    return source


def run(c, out, margin=0, windows=True, baseline=None, case="all"):
    import datetime
    import torch
    import torch_npu
    import torch.distributed as dist
    from .adapters import DeepEP
    from .layer import Layer, cpu_check
    from .metrics import summarize, regression

    c.validate()
    rank = int(os.environ.get("RANK", "-1"))
    local = int(os.environ.get("LOCAL_RANK", "-1"))
    if int(os.environ.get("WORLD_SIZE", 0)) != 4 or min(rank, local) < 0:
        raise ValueError("Use torchrun --nproc-per-node=4 with four visible devices")
    torch.set_num_threads(1)
    torch.npu.set_device(local)
    if torch.npu.device_count() != 4 or not torch.npu.is_available():
        raise RuntimeError("Exactly four visible, usable NPUs required")
    dest = out / f"rank{rank}"
    dest.mkdir(parents=True, exist_ok=False)
    adapter = None
    initialized = False
    report = dict(
        rank=rank,
        world_size=4,
        config=c.dictionary(),
        precision_passed=False,
        destroyed=False,
        trials=[],
        windows={},
        case=case,
        scope="real_NPU_EP4_MoE_window_" + case if case in ("A", "D") else "real_NPU_EP4_single_MoE_layer_A_D",
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

        def prepare():
            layer.reset()
            torch.npu.synchronize()
            dist.barrier()

        reference = None
        with torch.no_grad():
            for mode, (a, d) in MODES.items():
                prepare()
                actual, _ = layer.step(a, d)
                if reference is None:
                    reference = {k: v.clone() for k, v in actual.items()}
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
                for key in reference:
                    torch.testing.assert_close(
                        actual[key], reference[key], rtol=c.precision_rtol, atol=c.precision_atol
                    )
                    if not bool(torch.isfinite(actual[key]).all().item()):
                        raise RuntimeError("Nonfinite " + key)
            del reference
            report["precision_passed"] = True
            report["precision_scope"] = (
                "independent tiny CPU autograd; sampled source Y/dX and local dW; all six outputs cross-mode; frozen gates"
            )
            save()
            for rep in range(c.repeats if case in ("all", "layer") else 0):
                modes = list(MODES)
                modes = modes[rep % 4 :] + modes[: rep % 4]
                if rep % 2:
                    modes.reverse()
                for mode in modes:
                    a, d = MODES[mode]
                    for _ in range(c.warmup):
                        prepare()
                        layer.step(a, d, True)
                    samples = []
                    for _ in range(c.iterations):
                        prepare()
                        _, timing = layer.step(a, d, True)
                        samples.append(timing)
                    report["trials"].append(dict(mode=mode, repeat=rep, samples=samples))
                    save()
            if windows and case != "layer":
                # last_d comes from real forward/cached-dispatch/dgrad, not random matrices.
                for window_case in ("D", "A") if case == "all" else (case,):
                    report["windows"][window_case] = {}
                    for control in ("comm_only", "compute_only", "serial", "overlap"):
                        reps = []
                        for rep in range(c.repeats):
                            for _ in range(c.warmup):
                                prepare()
                                layer.window(window_case, control)
                            values = []
                            for _ in range(c.iterations):
                                prepare()
                                values.append(layer.window(window_case, control))
                            reps.append(values)
                        report["windows"][window_case][control] = reps
                    save()
            if c.profile_iterations:
                # Trace both serial and A+D after all performance samples, never mix samples.
                for mode in ("serial", "overlap") if case in ("A", "D") else ("serial", "ad"):
                    with torch_npu.profiler.profile(
                        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                        record_shapes=True,
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(dest / f"profile_{mode}")),
                        experimental_config=torch_npu.profiler._ExperimentalConfig(
                            profiler_level=torch_npu.profiler.ProfilerLevel.Level1
                        ),
                    ) as prof:
                        for _ in range(c.profile_iterations):
                            prepare()
                            if case in ("A", "D"):
                                with torch.profiler.record_function(f"moe/window/{case}/{mode}"):
                                    layer.window(case, mode)
                            else:
                                layer.step(*MODES[mode], True)
                            prof.step()
        torch.npu.synchronize()
        dist.barrier()
        adapter.destroy()
        report["destroyed"] = True
        save()
        gathered = [None] * 4
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
