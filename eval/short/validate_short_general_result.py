#!/usr/bin/env python3
"""Validate that an lm-eval phase produced the full intended benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_NON_MMLU = {
    "arc_challenge": 1172,
    "hellaswag": 10042,
    "winogrande": 1267,
    "minerva_math": 546,
    "ifeval": 541,
}

REQUIRED_NON_MMLU_METRICS = {
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "winogrande": "acc,none",
    "minerva_math": "math_verify,none",
    "ifeval": "prompt_level_strict_acc,none",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("mmlu", "non_mmlu"), required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    return parser.parse_args()


def latest_result(root: Path) -> Path:
    matches = sorted(root.rglob("results_*.json"))
    if not matches:
        raise ValueError(f"no results JSON below {root}")
    return matches[-1]


def task_block(payload: dict[str, Any], task: str) -> dict[str, Any]:
    groups = payload.get("groups") or {}
    results = payload.get("results") or {}
    block = groups.get(task) or results.get(task) or {}
    return block if isinstance(block, dict) else {}


def require_numeric_metric(payload: dict[str, Any], task: str, metric: str) -> None:
    value = task_block(payload, task).get(metric)
    if not isinstance(value, (int, float)):
        raise ValueError(f"missing numeric metric {task}.{metric}")


def validate_mmlu(payload: dict[str, Any]) -> None:
    samples = payload.get("n-samples") or {}
    if len(samples) != 57:
        raise ValueError(f"MMLU expected 57 subjects, found {len(samples)}")

    original = sum(int(item["original"]) for item in samples.values())
    effective = sum(int(item["effective"]) for item in samples.values())
    if original != 14042 or effective != 14042:
        raise ValueError(
            f"MMLU expected 14042/14042 samples, found {original}/{effective}"
        )

    configs = payload.get("configs") or {}
    shots = {configs.get(task, {}).get("num_fewshot") for task in samples}
    if shots != {5}:
        raise ValueError(f"MMLU expected 5-shot for every subject, found {shots}")
    require_numeric_metric(payload, "mmlu", "acc,none")


def validate_non_mmlu(payload: dict[str, Any]) -> None:
    samples = payload.get("n-samples") or {}
    for task, expected in EXPECTED_NON_MMLU.items():
        actual = samples.get(task)
        if not isinstance(actual, dict):
            raise ValueError(f"missing sample count for {task}")
        original = int(actual.get("original", -1))
        effective = int(actual.get("effective", -1))
        if (original, effective) != (expected, expected):
            raise ValueError(
                f"{task} expected {expected}/{expected} samples, "
                f"found {original}/{effective}"
            )

    math_config = payload.get("configs", {}).get("minerva_math", {})
    if math_config.get("dataset_name") != "precalculus":
        raise ValueError("Expected the historical Minerva MATH precalculus slice")
    for task, metric in REQUIRED_NON_MMLU_METRICS.items():
        require_numeric_metric(payload, task, metric)
    require_numeric_metric(payload, "ifeval", "inst_level_strict_acc,none")


def main() -> None:
    args = parse_args()
    result = latest_result(args.result_root)
    payload = json.loads(result.read_text(encoding="utf-8"))
    if args.phase == "mmlu":
        validate_mmlu(payload)
    else:
        validate_non_mmlu(payload)
    print(f"VALID phase={args.phase} result={result}")


if __name__ == "__main__":
    main()
