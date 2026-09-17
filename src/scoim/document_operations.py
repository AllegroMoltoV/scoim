"""Shared immutable operations for versioned SCoIM documents."""

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import jsonpatch
import jsonpointer

from .validation import CheckResult, IssueCode, ValidationIssue

DocumentChecker = Callable[[Mapping[str, object]], CheckResult]
ContentHasher = Callable[[Mapping[str, object]], str]


@dataclass(frozen=True, slots=True)
class ApplyResult:
    """Result of applying an RFC 6902 patch to a document."""

    applied: bool
    document: dict[str, object] | None
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True, slots=True)
class ApproveResult:
    """Result of approving a valid draft document."""

    approved: bool
    document: dict[str, object] | None
    issues: tuple[ValidationIssue, ...]


def apply_document_patch(
    document: Mapping[str, object],
    operations: Sequence[Mapping[str, object]],
    *,
    checker: DocumentChecker,
    new_draft: bool = False,
) -> ApplyResult:
    """Apply a patch to a copy and return only a valid updated document."""
    working_document = document
    if document["status"] == "draft" and new_draft:
        return _apply_failure(
            IssueCode.SEMANTIC_INVALID,
            "A new draft can only be created from an approved document",
            "/status",
        )
    if document["status"] == "approved" and not new_draft:
        return _apply_failure(
            IssueCode.IMMUTABLE_APPROVED,
            "An approved document cannot be overwritten",
            "/status",
        )
    if document["status"] == "approved":
        working_document = cast(dict[str, object], copy.deepcopy(document))
        working_document["status"] = "draft"
        working_document["approval"] = None
    if (
        not operations
        or operations[0].get("op") != "test"
        or operations[0].get("path") != "/revision"
        or operations[0].get("value") != document["revision"]
    ):
        return _apply_failure(
            IssueCode.STALE_REVISION,
            "The patch does not test the current revision first",
            "/revision",
        )
    updated = cast(dict[str, object], copy.deepcopy(working_document))
    for operation in operations:
        try:
            jsonpatch.apply_patch(updated, [dict(operation)], in_place=True)
        except (jsonpatch.JsonPatchException, jsonpointer.JsonPointerException) as error:
            path = operation.get("path")
            return _apply_failure(
                IssueCode.PATCH_INVALID,
                str(error),
                path if isinstance(path, str) else "",
            )
    expected_revision = cast(int, document["revision"]) + 1
    if updated.get("revision") != expected_revision:
        return _apply_failure(
            IssueCode.SEMANTIC_INVALID,
            "A patch must increment the revision by exactly one",
            "/revision",
        )
    validation = checker(updated)
    if not validation.valid:
        return ApplyResult(False, None, validation.issues)
    return ApplyResult(True, updated, ())


def approve_document(
    document: Mapping[str, object],
    *,
    checker: DocumentChecker,
    content_hasher: ContentHasher,
) -> ApproveResult:
    """Return an immutable approved copy of a valid draft document."""
    validation = checker(document)
    if not validation.valid:
        return ApproveResult(False, None, validation.issues)
    approved = cast(dict[str, object], copy.deepcopy(document))
    approved["status"] = "approved"
    approved["approval"] = {"content_sha256": content_hasher(document)}
    return ApproveResult(True, approved, ())


def _apply_failure(code: IssueCode, message: str, path: str) -> ApplyResult:
    return ApplyResult(False, None, (ValidationIssue(code, message, path),))
