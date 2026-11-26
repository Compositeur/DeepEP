"""Offline entry points and output isolation for the MoE overlap benchmark."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.moe_overlap.cli import load_config, output_path

ROOT = Path(__file__).resolve().parents[1]


def test_plan_without_site_packages():
    result = subprocess.run(
        [sys.executable, "-S", "-m", "benchmarks.moe_overlap", "--case", "D"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["selected_case"] == "D"
    assert set(plan["modes"]) == {"serial", "d_only", "a_only", "ad"}
    assert "not full training" in plan["scope"]


@pytest.mark.parametrize("text", ['{"tokens": 4, "tokens": 8}', '{"toknes": 4}', "[]"])
def test_reject_ambiguous_configs(tmp_path, text):
    config = tmp_path / "config.json"
    config.write_text(text)
    with pytest.raises(ValueError):
        load_config(config)


def test_output_escape_and_overwrite_are_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="inside"):
        output_path(tmp_path, Path("../escape"))
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        output_path(tmp_path, existing)
    (existing / "rank0").mkdir()
    monkeypatch.setenv("RANK", "1")
    assert output_path(tmp_path, existing, execute=True) == existing
    monkeypatch.setenv("RANK", "0")
    with pytest.raises(ValueError, match="already exists"):
        output_path(tmp_path, existing, execute=True)


def test_compare_rejects_cpu_as_performance_baseline(tmp_path):
    report = tmp_path / "cpu.json"
    report.write_text('{"scope": "CPU"}')
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "benchmarks.moe_overlap",
            "--compare",
            str(report),
            str(report),
            "--workspace",
            str(tmp_path),
            "--out",
            "comparison",
        ],
        cwd=ROOT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert json.loads((tmp_path / "comparison/regression.json").read_text())["status"] == "invalid"
