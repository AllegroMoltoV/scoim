"""Command-line interface for SCoIM music-script operations."""

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

from .codex_proposal import CodexStructuredRunner
from .document_operations import ApplyResult, ApproveResult
from .flow_operations import apply_flow_patch, approve_flow
from .flow_proposal import FlowProposalRequest, propose_flow
from .flow_query import FlowShowResult, show_flow
from .flow_validation import check_flow
from .public_realization import realize
from .storage import save_flow_document
from .validation import CheckResult, IssueCode, ValidationIssue


def _read_document(path: Path) -> tuple[dict[str, object] | None, ValidationIssue | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, ValidationIssue(IssueCode.STORAGE_ERROR, str(error), "")
    if not isinstance(value, dict):
        return None, ValidationIssue(
            IssueCode.SCHEMA_INVALID,
            "The document must be a JSON object",
            "",
        )
    return cast(dict[str, object], value), None


def _write_result(result: object) -> None:
    value = result if isinstance(result, Mapping) else asdict(result)
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _save_transformed_document(
    *,
    input_path: Path,
    output_path: Path,
    source: dict[str, object],
    transformed: dict[str, object],
) -> tuple[ValidationIssue, ...]:
    same_path = input_path.resolve() == output_path.resolve()
    expected_revision = cast(int, source["revision"]) if same_path else None
    return save_flow_document(
        output_path,
        transformed,
        expected_revision=expected_revision,
    ).issues


def _read_patch(path: Path) -> tuple[list[Mapping[str, object]] | None, ValidationIssue | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, ValidationIssue(IssueCode.PATCH_INVALID, str(error), "")
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        return None, ValidationIssue(
            IssueCode.PATCH_INVALID,
            "The patch must be a JSON array of objects",
            "",
        )
    return cast(list[Mapping[str, object]], value), None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scoim")
    subparsers = parser.add_subparsers(dest="command", required=True)
    propose_parser = subparsers.add_parser("propose", help="自然言語から楽曲台本を提案する")
    propose_parser.add_argument("instruction")
    propose_parser.add_argument("--document-id", required=True)
    propose_parser.add_argument("--instrumentation", required=True)
    propose_parser.add_argument("--duration", type=float, required=True)
    propose_parser.add_argument("--model", required=True)
    propose_parser.add_argument("--output", type=Path, required=True)
    check_parser = subparsers.add_parser("check", help="楽曲台本を検証する")
    check_parser.add_argument("flow", type=Path)
    show_parser = subparsers.add_parser("show", help="楽曲台本の指定場面と前後を表示する")
    show_parser.add_argument("flow", type=Path)
    show_parser.add_argument("scene_id")
    apply_parser = subparsers.add_parser("apply", help="楽曲台本へJSON Patchを適用する")
    apply_parser.add_argument("flow", type=Path)
    apply_parser.add_argument("patch", type=Path)
    apply_parser.add_argument("--output", type=Path, required=True)
    apply_parser.add_argument("--new-draft", action="store_true")
    approve_parser = subparsers.add_parser("approve", help="楽曲台本を承認する")
    approve_parser.add_argument("flow", type=Path)
    approve_parser.add_argument("--output", type=Path, required=True)
    realize_parser = subparsers.add_parser("realize", help="承認済み台本を演奏へ実現する")
    realize_parser.add_argument("input", type=Path)
    realize_parser.add_argument("--profile")
    realize_parser.add_argument("--trial-id")
    realize_parser.add_argument("--composition-id")
    realize_parser.add_argument("--model")
    realize_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one SCoIM command and return its process exit code."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "propose":
        proposal_result = propose_flow(
            FlowProposalRequest(
                instruction=arguments.instruction,
                document_id=arguments.document_id,
                instrumentation=arguments.instrumentation,
                target_duration=arguments.duration,
            ),
            CodexStructuredRunner(
                model=arguments.model,
                working_directory=Path.cwd(),
            ),
            arguments.output,
        )
        _write_result(proposal_result)
        return 0 if proposal_result.proposed else 1
    if arguments.command == "realize":
        runner = (
            CodexStructuredRunner(
                model=arguments.model,
                working_directory=Path.cwd(),
            )
            if arguments.model is not None
            else None
        )
        result = realize(
            arguments.input,
            arguments.output,
            runner=runner,
            model=arguments.model,
            trial_id=arguments.trial_id,
            composition_id=arguments.composition_id,
            profile=arguments.profile,
        )
        _write_result(result)
        return 0 if result.succeeded else 1
    document, issue = _read_document(arguments.flow)
    if arguments.command == "check":
        if issue is not None:
            result = CheckResult(valid=False, issues=(issue,))
        else:
            assert document is not None
            result = check_flow(document)
        _write_result(result)
        return 0 if result.valid else 1
    if arguments.command == "apply":
        if issue is not None:
            apply_result = ApplyResult(applied=False, document=None, issues=(issue,))
        else:
            assert document is not None
            operations, patch_issue = _read_patch(arguments.patch)
            if patch_issue is not None:
                apply_result = ApplyResult(
                    applied=False,
                    document=None,
                    issues=(patch_issue,),
                )
            else:
                assert operations is not None
                apply_result = apply_flow_patch(
                    document,
                    operations,
                    new_draft=arguments.new_draft,
                )
                if apply_result.document is not None:
                    save_issues = _save_transformed_document(
                        input_path=arguments.flow,
                        output_path=arguments.output,
                        source=document,
                        transformed=apply_result.document,
                    )
                    if save_issues:
                        apply_result = ApplyResult(
                            applied=False,
                            document=None,
                            issues=save_issues,
                        )
        _write_result(apply_result)
        return 0 if apply_result.applied else 1
    if arguments.command == "approve":
        if issue is not None:
            approve_result = ApproveResult(approved=False, document=None, issues=(issue,))
        else:
            assert document is not None
            approve_result = approve_flow(document)
            if approve_result.document is not None:
                save_issues = _save_transformed_document(
                    input_path=arguments.flow,
                    output_path=arguments.output,
                    source=document,
                    transformed=approve_result.document,
                )
                if save_issues:
                    approve_result = ApproveResult(
                        approved=False,
                        document=None,
                        issues=save_issues,
                    )
        _write_result(approve_result)
        return 0 if approve_result.approved else 1
    if issue is not None:
        show_result = FlowShowResult(False, None, None, None, None, (issue,))
    else:
        assert document is not None
        show_result = show_flow(document, arguments.scene_id)
    _write_result(show_result)
    return 0 if show_result.shown else 1
