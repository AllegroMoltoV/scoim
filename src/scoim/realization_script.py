"""Shared contract for scripts accepted by the realization pipeline."""

from collections.abc import Mapping

from .script_validation import (
    SCRIPT_DOCUMENT_TYPE,
    SCRIPT_SCHEMA_VERSION,
    check_script_document,
    script_content_sha256,
)
from .validation import CheckResult, IssueCode, ValidationIssue, check, content_sha256

_LEGACY_SCHEMA_VERSION = "0.1.0"


def is_validated_script(document: Mapping[str, object]) -> bool:
    """Return whether a document explicitly declares the current script contract."""

    return (
        document.get("document_type") == SCRIPT_DOCUMENT_TYPE
        and document.get("schema_version") == SCRIPT_SCHEMA_VERSION
    )


def check_realization_script(document: Mapping[str, object]) -> CheckResult:
    """Validate a current script or an explicitly versioned legacy replay script."""

    if is_validated_script(document):
        validation = check_script_document(document)
        required_status = "validated"
    elif "document_type" not in document:
        validation = check(document)
        required_status = "approved"
    else:
        validation = check_script_document(document)
        required_status = "validated"
    if not validation.valid:
        return validation
    if document.get("status") != required_status:
        article = "an" if required_status == "approved" else "a"
        issue = ValidationIssue(
            IssueCode.SEMANTIC_INVALID,
            f"Realization requires {article} {required_status} script",
            "/status",
        )
        return CheckResult(False, (issue,))
    return validation


def realization_script_sha256(document: Mapping[str, object]) -> str:
    """Hash a realization script with the contract declared by that document."""

    if is_validated_script(document):
        return script_content_sha256(document)
    return content_sha256(document)
