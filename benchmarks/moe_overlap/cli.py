"""Plan, validate, execute and compare an EP4/EP16 MoE overlap benchmark."""

import argparse
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import sys

from .config import Config, plan
from .scenarios import WINDOWS

CONFIG_DIR = Path(__file__).parent / "configs"
DEFAULT_WORKSPACE = Path(__file__).resolve().parents[2] / "benchmark-runs" / "moe-overlap"


def load_json(path):
    """Reject ambiguous or misspelled configuration before loading Torch."""

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate configuration field: {key}")
            result[key] = value
        return result

    if path.stat().st_size > 65536:
        raise ValueError("Configuration exceeds 64 KiB")
    raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a JSON object")
    return raw


def load_config(path):
    raw = load_json(path)
    unknown = set(raw) - {field.name for field in fields(Config)}
    if unknown:
        raise ValueError("Unknown configuration fields: " + ", ".join(sorted(unknown)))
    return Config(**raw).validate()


def output_path(workspace, requested, execute=False):
    """Allow each rank its own new directory under an explicitly chosen workspace."""
    out = (workspace / requested).resolve()
    if not out.is_relative_to(workspace):
        raise ValueError("Output must stay inside --workspace")
    target = out / f"rank{os.environ.get('RANK', '-1')}" if execute else out
    if target.exists() or (out / "summary.json").exists():
        raise ValueError("Output already exists; choose a new --out directory")
    return out


def runtime_environment(workspace):
    sys.dont_write_bytecode = True
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["TORCH_EXTENSIONS_DIR"] = str(workspace / "cache" / "extensions")
    os.environ["XDG_CACHE_HOME"] = str(workspace / "cache")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")


def write_result(out, name, result):
    if out is not None:
        out.mkdir(parents=True, exist_ok=False)
        (out / name).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "smoke_v1.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Execute on an explicitly allocated EP4/EP16 NPU group")
    mode.add_argument("--cpu-check", action="store_true", help="Check small float64 math against autograd")
    mode.add_argument("--check-env", action="store_true", help="Check software imports and signatures only")
    mode.add_argument("--compare", nargs=2, type=Path, metavar=("BASELINE", "CANDIDATE"))
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--out", type=Path, help="New result directory, relative to --workspace or absolute within it")
    parser.add_argument("--margin-percent", type=float, default=0)
    parser.add_argument("--skip-windows", action="store_true")
    parser.add_argument("--case", choices=[*WINDOWS, "layer", "all"], default="all")
    parser.add_argument("--layer-suite", choices=["AD", "ACD"], help="Four legacy modes or eight A/C/D combinations")
    parser.add_argument(
        "--reference-overlap", type=Path, help="Reported observations; never used to tune work or timings"
    )
    args = parser.parse_args(argv)
    if not 0 <= args.margin_percent < 100:
        parser.error("margin must be in [0, 100)")
    if args.skip_windows and args.case in WINDOWS:
        parser.error("Selected window cannot be skipped")
    try:
        config = None if args.compare else load_config(args.config)
        if config is not None and args.layer_suite:
            config = replace(config, layer_suite=args.layer_suite).validate()
        from .reference import load_reference

        reference = load_reference(args.reference_overlap) if args.reference_overlap else None
        workspace = args.workspace.resolve()
        active = args.execute or args.cpu_check or args.compare or args.check_env
        if not active:
            result = plan(config)
            result["selected_case"] = args.case
            result["isolated_windows_enabled"] = not args.skip_windows and args.case != "layer"
            result["reported_overlap_reference"] = reference
            print(json.dumps(result, indent=2))
            return
        if args.out is None and not args.check_env:
            parser.error("--out is required")
        out = output_path(workspace, args.out, args.execute) if args.out is not None else None
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    runtime_environment(workspace)
    if args.check_env:
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        from .environment import probe

        result = probe(config)
        write_result(out, "environment.json", result)
        if not result["software_checks_passed"]:
            raise SystemExit(2)
    elif args.cpu_check:
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        from .layer import cpu_check

        write_result(out, "cpu_check.json", cpu_check(config))
    elif args.execute:
        from .runner import run

        run(config, out, args.margin_percent, not args.skip_windows, case=args.case, reference=reference)
    else:
        from .metrics import regression

        result = regression(*(json.loads(path.read_text()) for path in args.compare), args.margin_percent)
        write_result(out, "regression.json", result)
        if result["status"] == "invalid":
            raise SystemExit(2)
        if any(item["conclusion"] == "regression" for row in result["results"].values() for item in row.values()):
            raise SystemExit(3)
