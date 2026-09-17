import json
from pathlib import Path

import pytest

from llm_musical_composer.evaluator_registry import (
    EvaluatorRegistryError,
    assert_evaluator_use,
    load_evaluator_registry,
    main,
    validate_evaluator_registry,
)

PROJECT_ROOT = Path(__file__).parents[1]


def _write_registry_fixture(tmp_path: Path) -> dict[str, object]:
    (tmp_path / "impl.py").write_text("def evaluate():\n    return True\n", encoding="utf-8")
    (tmp_path / "evidence.md").write_text("# evidence\n", encoding="utf-8")
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"fixed artifact")
    return {
        "schema_version": 1,
        "baseline_artifacts": [
            {
                "id": "accepted",
                "role": "accepted_baseline",
                "files": [
                    {
                        "path": "artifact.bin",
                        "sha256": (
                            "D66E4EBC023CBE8D82F97067D74D7633377270ACF9564A3A7938A8C77ECE1D84"
                        ),
                    }
                ],
                "known_findings": ["fixed"],
            }
        ],
        "evaluators": [
            {
                "id": "hard",
                "status": "hard_gate",
                "implementation_path": "impl.py",
                "entrypoint": "evaluate",
                "evidence_reports": ["evidence.md"],
                "scope": "fixture",
                "units": ["boolean"],
                "invariances": [],
                "known_false_positives": [],
                "known_false_negatives": [],
            },
            {
                "id": "selection",
                "status": "selection",
                "implementation_path": "impl.py",
                "entrypoint": "evaluate",
                "evidence_reports": ["evidence.md"],
                "validated_controls": ["fixture-ordering"],
                "scope": "fixture",
                "units": ["rank"],
                "invariances": [],
                "known_false_positives": [],
                "known_false_negatives": [],
            },
            {
                "id": "diagnostic",
                "status": "diagnostic_only",
                "implementation_path": "impl.py",
                "entrypoint": "evaluate",
                "evidence_reports": ["evidence.md"],
                "scope": "fixture",
                "units": ["value"],
                "invariances": [],
                "known_false_positives": [],
                "known_false_negatives": [],
            },
            {
                "id": "rejected",
                "status": "rejected",
                "implementation_path": "impl.py",
                "entrypoint": "evaluate",
                "evidence_reports": ["evidence.md"],
                "rejection_reason": "failed control",
                "scope": "fixture",
                "units": ["value"],
                "invariances": [],
                "known_false_positives": [],
                "known_false_negatives": [],
            },
        ],
        "policies": {
            "acceptance": ["hard"],
            "ranking": ["selection"],
            "diagnostic": ["diagnostic"],
            "forbidden": ["rejected"],
        },
    }


def test_project_registry_and_baseline_hashes_are_valid() -> None:
    registry = load_evaluator_registry(
        PROJECT_ROOT / "configs" / "evaluator-registry-v1.json",
        project_root=PROJECT_ROOT,
    )

    assert registry["schema_version"] == 1
    assert {item["status"] for item in registry["evaluators"]} == {
        "hard_gate",
        "selection",
        "diagnostic_only",
        "rejected",
    }


@pytest.mark.parametrize("status", ["unknown", "pass", "hard"])
def test_unknown_status_is_rejected(tmp_path: Path, status: str) -> None:
    registry = _write_registry_fixture(tmp_path)
    registry["evaluators"][0]["status"] = status

    with pytest.raises(EvaluatorRegistryError, match="status"):
        validate_evaluator_registry(registry, project_root=tmp_path)


def test_duplicate_evaluator_id_is_rejected(tmp_path: Path) -> None:
    registry = _write_registry_fixture(tmp_path)
    registry["evaluators"][1]["id"] = "hard"

    with pytest.raises(EvaluatorRegistryError, match="duplicate"):
        validate_evaluator_registry(registry, project_root=tmp_path)


def test_selection_without_control_evidence_is_rejected(tmp_path: Path) -> None:
    registry = _write_registry_fixture(tmp_path)
    del registry["evaluators"][1]["validated_controls"]

    with pytest.raises(EvaluatorRegistryError, match="validated_controls"):
        validate_evaluator_registry(registry, project_root=tmp_path)


def test_missing_implementation_path_is_rejected(tmp_path: Path) -> None:
    registry = _write_registry_fixture(tmp_path)
    registry["evaluators"][0]["implementation_path"] = "missing.py"

    with pytest.raises(EvaluatorRegistryError, match="implementation_path"):
        validate_evaluator_registry(registry, project_root=tmp_path)


@pytest.mark.parametrize("field", ["entrypoint", "scope"])
def test_required_evaluator_text_is_rejected_when_blank(tmp_path: Path, field: str) -> None:
    registry = _write_registry_fixture(tmp_path)
    registry["evaluators"][0][field] = ""

    with pytest.raises(EvaluatorRegistryError, match=field):
        validate_evaluator_registry(registry, project_root=tmp_path)


def test_empty_or_missing_evidence_is_rejected(tmp_path: Path) -> None:
    registry = _write_registry_fixture(tmp_path)
    registry["evaluators"][0]["evidence_reports"] = []

    with pytest.raises(EvaluatorRegistryError, match="evidence_reports"):
        validate_evaluator_registry(registry, project_root=tmp_path)

    registry = _write_registry_fixture(tmp_path)
    registry["evaluators"][0]["evidence_reports"] = ["missing.md"]
    with pytest.raises(EvaluatorRegistryError, match="does not exist"):
        validate_evaluator_registry(registry, project_root=tmp_path)


def test_evaluator_list_metadata_is_required(tmp_path: Path) -> None:
    registry = _write_registry_fixture(tmp_path)
    registry["evaluators"][0]["units"] = "boolean"

    with pytest.raises(EvaluatorRegistryError, match="units"):
        validate_evaluator_registry(registry, project_root=tmp_path)


def test_rejected_evaluator_requires_reason(tmp_path: Path) -> None:
    registry = _write_registry_fixture(tmp_path)
    registry["evaluators"][3]["rejection_reason"] = ""

    with pytest.raises(EvaluatorRegistryError, match="rejection_reason"):
        validate_evaluator_registry(registry, project_root=tmp_path)


def test_baseline_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    registry = _write_registry_fixture(tmp_path)
    registry["baseline_artifacts"][0]["files"][0]["sha256"] = "0" * 64

    with pytest.raises(EvaluatorRegistryError, match="SHA-256"):
        validate_evaluator_registry(registry, project_root=tmp_path)


@pytest.mark.parametrize("role", ["accepted", "positive", None])
def test_unknown_baseline_role_is_rejected(tmp_path: Path, role: object) -> None:
    registry = _write_registry_fixture(tmp_path)
    registry["baseline_artifacts"][0]["role"] = role

    with pytest.raises(EvaluatorRegistryError, match="role"):
        validate_evaluator_registry(registry, project_root=tmp_path)


@pytest.mark.parametrize("field", ["known_findings", "files"])
def test_baseline_requires_findings_and_files(tmp_path: Path, field: str) -> None:
    registry = _write_registry_fixture(tmp_path)
    registry["baseline_artifacts"][0][field] = []

    with pytest.raises(EvaluatorRegistryError, match=field):
        validate_evaluator_registry(registry, project_root=tmp_path)


def test_missing_baseline_file_is_rejected(tmp_path: Path) -> None:
    registry = _write_registry_fixture(tmp_path)
    registry["baseline_artifacts"][0]["files"][0]["path"] = "missing.bin"

    with pytest.raises(EvaluatorRegistryError, match="does not exist"):
        validate_evaluator_registry(registry, project_root=tmp_path)


@pytest.mark.parametrize("digest", ["x" * 64, "0" * 63, ""])
def test_invalid_baseline_digest_is_rejected(tmp_path: Path, digest: str) -> None:
    registry = _write_registry_fixture(tmp_path)
    registry["baseline_artifacts"][0]["files"][0]["sha256"] = digest

    with pytest.raises(EvaluatorRegistryError, match="sha256"):
        validate_evaluator_registry(registry, project_root=tmp_path)


def test_duplicate_baseline_id_is_rejected(tmp_path: Path) -> None:
    registry = _write_registry_fixture(tmp_path)
    registry["baseline_artifacts"].append(dict(registry["baseline_artifacts"][0]))

    with pytest.raises(EvaluatorRegistryError, match="duplicate baseline"):
        validate_evaluator_registry(registry, project_root=tmp_path)


def test_wrong_schema_version_is_rejected(tmp_path: Path) -> None:
    registry = _write_registry_fixture(tmp_path)
    registry["schema_version"] = 2

    with pytest.raises(EvaluatorRegistryError, match="schema_version"):
        validate_evaluator_registry(registry, project_root=tmp_path)


def test_unknown_or_misclassified_policy_reference_is_rejected(tmp_path: Path) -> None:
    registry = _write_registry_fixture(tmp_path)
    registry["policies"]["acceptance"] = ["missing"]
    with pytest.raises(EvaluatorRegistryError, match="unknown evaluator"):
        validate_evaluator_registry(registry, project_root=tmp_path)

    registry = _write_registry_fixture(tmp_path)
    registry["policies"]["acceptance"] = ["diagnostic"]
    with pytest.raises(EvaluatorRegistryError, match="cannot contain"):
        validate_evaluator_registry(registry, project_root=tmp_path)


def test_evaluator_must_appear_in_exactly_one_policy(tmp_path: Path) -> None:
    registry = _write_registry_fixture(tmp_path)
    registry["policies"]["diagnostic"] = []

    with pytest.raises(EvaluatorRegistryError, match="exactly one"):
        validate_evaluator_registry(registry, project_root=tmp_path)


@pytest.mark.parametrize("evaluator_id", ["diagnostic", "rejected"])
@pytest.mark.parametrize("purpose", ["acceptance", "ranking"])
def test_diagnostic_and_rejected_evaluators_cannot_drive_decisions(
    tmp_path: Path, evaluator_id: str, purpose: str
) -> None:
    registry = validate_evaluator_registry(_write_registry_fixture(tmp_path), tmp_path)

    with pytest.raises(EvaluatorRegistryError, match="cannot be used"):
        assert_evaluator_use(registry, [evaluator_id], purpose=purpose)


def test_allowed_evaluator_use_and_invalid_use_requests(tmp_path: Path) -> None:
    registry = validate_evaluator_registry(_write_registry_fixture(tmp_path), tmp_path)

    assert_evaluator_use(registry, ["hard"], purpose="acceptance")
    assert_evaluator_use(registry, ["selection"], purpose="ranking")
    assert_evaluator_use(registry, ["diagnostic"], purpose="diagnostic")
    with pytest.raises(EvaluatorRegistryError, match="unknown evaluator use purpose"):
        assert_evaluator_use(registry, ["hard"], purpose="repair")
    with pytest.raises(EvaluatorRegistryError, match="unknown evaluator id"):
        assert_evaluator_use(registry, ["missing"], purpose="acceptance")


def test_registry_loader_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(EvaluatorRegistryError, match="object"):
        load_evaluator_registry(path, project_root=tmp_path)


@pytest.mark.parametrize("content", ["{", "not json"])
def test_registry_loader_wraps_json_errors(tmp_path: Path, content: str) -> None:
    path = tmp_path / "registry.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(EvaluatorRegistryError, match="unable to load"):
        load_evaluator_registry(path, project_root=tmp_path)


def test_registry_loader_wraps_missing_file(tmp_path: Path) -> None:
    with pytest.raises(EvaluatorRegistryError, match="unable to load"):
        load_evaluator_registry(tmp_path / "missing.json", project_root=tmp_path)


def test_registry_cli_prints_machine_readable_summary(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "--registry",
            str(PROJECT_ROOT / "configs" / "evaluator-registry-v1.json"),
            "--project-root",
            str(PROJECT_ROOT),
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "pass"
    assert summary["baseline_artifact_count"] == 4
