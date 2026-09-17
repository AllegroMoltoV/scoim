"""Validation for human-editable SCoIM flow documents."""

import hashlib
import json
from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from typing import cast

import rfc8785
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .validation import CheckResult, IssueCode, ValidationIssue

FLOW_DOCUMENT_TYPE = "flow"
FLOW_SCHEMA_VERSION = "0.1.0"
_CONTENT_FIELDS = (
    "document_type",
    "schema_version",
    "document_id",
    "revision",
    "title",
    "overall_flow",
    "instrumentation",
    "target_duration",
    "scenes",
)


def check_flow(document: Mapping[str, object]) -> CheckResult:
    """Check one decoded flow document without raising for content errors."""
    schema_version = document.get("schema_version")
    if isinstance(schema_version, str) and schema_version != FLOW_SCHEMA_VERSION:
        return CheckResult(
            False,
            (
                ValidationIssue(
                    IssueCode.UNSUPPORTED_SCHEMA_VERSION,
                    f"Unsupported flow schema version: {schema_version!r}",
                    "/schema_version",
                ),
            ),
        )
    schema_issues = tuple(
        ValidationIssue(IssueCode.SCHEMA_INVALID, error.message, _error_pointer(error))
        for error in sorted(
            _validator().iter_errors(document),
            key=lambda item: (tuple(str(part) for part in item.absolute_path), item.message),
        )
    )
    if schema_issues:
        return CheckResult(False, schema_issues)
    issues: list[ValidationIssue] = []
    scenes = cast(list[object], document["scenes"])
    seen: set[str] = set()
    for index, raw_scene in enumerate(scenes):
        scene = cast(dict[str, object], raw_scene)
        scene_id = cast(str, scene["scene_id"])
        if scene_id in seen:
            issues.append(
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    f"A scene ID must be unique: {scene_id}",
                    f"/scenes/{index}/scene_id",
                )
            )
        seen.add(scene_id)
    issues.extend(_check_approval(document))
    return CheckResult(not issues, tuple(issues))


def flow_content_sha256(document: Mapping[str, object]) -> str:
    """Hash only the immutable musical content of a flow document."""
    content = {field: document[field] for field in _CONTENT_FIELDS}
    return hashlib.sha256(rfc8785.dumps(content)).hexdigest()


@cache
def _validator() -> Draft202012Validator:
    schema_text = files("scoim").joinpath("schemas", "flow-0.1.0.schema.json").read_text("utf-8")
    return Draft202012Validator(json.loads(schema_text))


def _error_pointer(error: ValidationError) -> str:
    return "".join(f"/{_escape_pointer_token(part)}" for part in error.absolute_path)


def _escape_pointer_token(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _check_approval(document: Mapping[str, object]) -> list[ValidationIssue]:
    status = document["status"]
    approval = document["approval"]
    if status == "draft" and approval is not None:
        return [
            ValidationIssue(
                IssueCode.SEMANTIC_INVALID,
                "A draft flow must not contain approval metadata",
                "/approval",
            )
        ]
    if status == "approved" and approval is None:
        return [
            ValidationIssue(
                IssueCode.SEMANTIC_INVALID,
                "An approved flow must contain approval metadata",
                "/approval",
            )
        ]
    if status == "approved":
        approval_object = cast(dict[str, object], approval)
        if approval_object["content_sha256"] != flow_content_sha256(document):
            return [
                ValidationIssue(
                    IssueCode.SEMANTIC_INVALID,
                    "The flow approval hash does not match its content",
                    "/approval/content_sha256",
                )
            ]
    return []
