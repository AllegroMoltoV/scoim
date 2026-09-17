from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from llm_musical_composer.run_state import sha256_file

VALID_STATUSES = frozenset({"hard_gate", "selection", "diagnostic_only", "rejected"})
PURPOSE_STATUS = {
    "acceptance": "hard_gate",
    "ranking": "selection",
    "diagnostic": "diagnostic_only",
}


class EvaluatorRegistryError(ValueError):
    """Raised when an evaluator registry violates its policy contract."""


def _as_object(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluatorRegistryError(f"{location} must be a JSON object")
    return value


def _as_list(value: object, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise EvaluatorRegistryError(f"{location} must be a JSON array")
    return value


def _nonempty_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluatorRegistryError(f"{location} must be a non-empty string")
    return value


def _project_path(project_root: Path, raw_path: object, location: str) -> Path:
    relative = Path(_nonempty_string(raw_path, location))
    if relative.is_absolute() or ".." in relative.parts:
        raise EvaluatorRegistryError(f"{location} must be a project-relative path")
    root = project_root.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise EvaluatorRegistryError(f"{location} escapes project_root")
    return resolved


def _validate_evaluator(
    raw: object,
    *,
    index: int,
    project_root: Path,
) -> dict[str, Any]:
    evaluator = dict(_as_object(raw, f"evaluators[{index}]"))
    evaluator_id = _nonempty_string(evaluator.get("id"), f"evaluators[{index}].id")
    status = evaluator.get("status")
    if status not in VALID_STATUSES:
        raise EvaluatorRegistryError(
            f"evaluators[{index}].status must be one of {sorted(VALID_STATUSES)}"
        )
    implementation = _project_path(
        project_root,
        evaluator.get("implementation_path"),
        f"evaluators[{index}].implementation_path",
    )
    if not implementation.is_file():
        raise EvaluatorRegistryError(
            f"evaluators[{index}].implementation_path does not exist: {implementation}"
        )
    _nonempty_string(evaluator.get("entrypoint"), f"evaluators[{index}].entrypoint")
    _nonempty_string(evaluator.get("scope"), f"evaluators[{index}].scope")

    reports = _as_list(evaluator.get("evidence_reports"), f"evaluators[{index}].evidence_reports")
    if not reports:
        raise EvaluatorRegistryError(f"evaluators[{index}].evidence_reports must not be empty")
    for report_index, report in enumerate(reports):
        report_path = _project_path(
            project_root,
            report,
            f"evaluators[{index}].evidence_reports[{report_index}]",
        )
        if not report_path.is_file():
            raise EvaluatorRegistryError(
                f"evaluators[{index}].evidence_reports[{report_index}] does not exist"
            )

    for field in (
        "units",
        "invariances",
        "known_false_positives",
        "known_false_negatives",
    ):
        _as_list(evaluator.get(field), f"evaluators[{index}].{field}")
    if status == "selection" and not _as_list(
        evaluator.get("validated_controls"), f"evaluators[{index}].validated_controls"
    ):
        raise EvaluatorRegistryError(
            f"selection evaluator {evaluator_id!r} requires validated_controls"
        )
    if status == "rejected":
        _nonempty_string(evaluator.get("rejection_reason"), f"evaluators[{index}].rejection_reason")
    return evaluator


def _validate_baseline(
    raw: object,
    *,
    index: int,
    project_root: Path,
) -> dict[str, Any]:
    baseline = dict(_as_object(raw, f"baseline_artifacts[{index}]"))
    _nonempty_string(baseline.get("id"), f"baseline_artifacts[{index}].id")
    role = baseline.get("role")
    if role not in {"accepted_baseline", "negative_control", "diagnostic_control"}:
        raise EvaluatorRegistryError(f"baseline_artifacts[{index}].role is invalid")
    findings = _as_list(
        baseline.get("known_findings"), f"baseline_artifacts[{index}].known_findings"
    )
    if not findings:
        raise EvaluatorRegistryError(
            f"baseline_artifacts[{index}].known_findings must not be empty"
        )
    files = _as_list(baseline.get("files"), f"baseline_artifacts[{index}].files")
    if not files:
        raise EvaluatorRegistryError(f"baseline_artifacts[{index}].files must not be empty")
    for file_index, raw_file in enumerate(files):
        file_record = _as_object(raw_file, f"baseline_artifacts[{index}].files[{file_index}]")
        artifact_path = _project_path(
            project_root,
            file_record.get("path"),
            f"baseline_artifacts[{index}].files[{file_index}].path",
        )
        if not artifact_path.is_file():
            raise EvaluatorRegistryError(
                f"baseline_artifacts[{index}].files[{file_index}] does not exist"
            )
        expected = _nonempty_string(
            file_record.get("sha256"),
            f"baseline_artifacts[{index}].files[{file_index}].sha256",
        ).upper()
        invalid_character = any(character not in "0123456789ABCDEF" for character in expected)
        if len(expected) != 64 or invalid_character:
            raise EvaluatorRegistryError(
                f"baseline_artifacts[{index}].files[{file_index}].sha256 is invalid"
            )
        actual = sha256_file(artifact_path).upper()
        if actual != expected:
            raise EvaluatorRegistryError(
                f"baseline artifact SHA-256 mismatch for {artifact_path}: "
                f"expected {expected}, got {actual}"
            )
    return baseline


def validate_evaluator_registry(
    raw_registry: object,
    project_root: Path,
) -> dict[str, Any]:
    registry = dict(_as_object(raw_registry, "registry"))
    if registry.get("schema_version") != 1:
        raise EvaluatorRegistryError("schema_version must be 1")

    evaluators = [
        _validate_evaluator(item, index=index, project_root=project_root)
        for index, item in enumerate(_as_list(registry.get("evaluators"), "evaluators"))
    ]
    evaluator_ids = [item["id"] for item in evaluators]
    if len(evaluator_ids) != len(set(evaluator_ids)):
        raise EvaluatorRegistryError("duplicate evaluator id")
    by_id = {item["id"]: item for item in evaluators}

    baselines = [
        _validate_baseline(item, index=index, project_root=project_root)
        for index, item in enumerate(
            _as_list(registry.get("baseline_artifacts"), "baseline_artifacts")
        )
    ]
    baseline_ids = [item["id"] for item in baselines]
    if len(baseline_ids) != len(set(baseline_ids)):
        raise EvaluatorRegistryError("duplicate baseline artifact id")

    policies = _as_object(registry.get("policies"), "policies")
    expected_policy_status = {
        "acceptance": "hard_gate",
        "ranking": "selection",
        "diagnostic": "diagnostic_only",
        "forbidden": "rejected",
    }
    assigned: list[str] = []
    for policy, expected_status in expected_policy_status.items():
        ids = _as_list(policies.get(policy), f"policies.{policy}")
        for evaluator_id in ids:
            evaluator_id = _nonempty_string(evaluator_id, f"policies.{policy}[]")
            if evaluator_id not in by_id:
                raise EvaluatorRegistryError(
                    f"policies.{policy} references unknown evaluator {evaluator_id!r}"
                )
            if by_id[evaluator_id]["status"] != expected_status:
                raise EvaluatorRegistryError(
                    f"policies.{policy} cannot contain {by_id[evaluator_id]['status']} evaluator "
                    f"{evaluator_id!r}"
                )
            assigned.append(evaluator_id)
    if sorted(assigned) != sorted(evaluator_ids):
        raise EvaluatorRegistryError("each evaluator must appear in exactly one matching policy")

    registry["evaluators"] = evaluators
    registry["baseline_artifacts"] = baselines
    registry["policies"] = dict(policies)
    return registry


def load_evaluator_registry(path: Path, *, project_root: Path) -> dict[str, Any]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluatorRegistryError(f"unable to load evaluator registry: {error}") from error
    return validate_evaluator_registry(raw, Path(project_root))


def assert_evaluator_use(
    registry: Mapping[str, Any],
    evaluator_ids: Iterable[str],
    *,
    purpose: str,
) -> None:
    if purpose not in PURPOSE_STATUS:
        raise EvaluatorRegistryError(f"unknown evaluator use purpose: {purpose}")
    expected = PURPOSE_STATUS[purpose]
    by_id = {item["id"]: item for item in registry["evaluators"]}
    for evaluator_id in evaluator_ids:
        if evaluator_id not in by_id:
            raise EvaluatorRegistryError(f"unknown evaluator id: {evaluator_id}")
        actual = by_id[evaluator_id]["status"]
        if actual != expected:
            raise EvaluatorRegistryError(
                f"{actual} evaluator {evaluator_id!r} cannot be used for {purpose}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the evaluator registry and baselines")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    registry = load_evaluator_registry(args.registry, project_root=args.project_root)
    status_counts = {
        status: sum(item["status"] == status for item in registry["evaluators"])
        for status in sorted(VALID_STATUSES)
    }
    print(
        json.dumps(
            {
                "status": "pass",
                "schema_version": registry["schema_version"],
                "baseline_artifact_count": len(registry["baseline_artifacts"]),
                "evaluator_status_counts": status_counts,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
