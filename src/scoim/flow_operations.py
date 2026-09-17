"""Immutable editing and approval operations for SCoIM flow documents."""

from collections.abc import Mapping, Sequence

from .document_operations import (
    ApplyResult,
    ApproveResult,
    apply_document_patch,
    approve_document,
)
from .flow_validation import check_flow, flow_content_sha256


def apply_flow_patch(
    document: Mapping[str, object],
    operations: Sequence[Mapping[str, object]],
    *,
    new_draft: bool = False,
) -> ApplyResult:
    """Apply an RFC 6902 patch to a flow copy."""
    return apply_document_patch(document, operations, checker=check_flow, new_draft=new_draft)


def approve_flow(document: Mapping[str, object]) -> ApproveResult:
    """Return an approved immutable copy of a valid flow draft."""
    return approve_document(document, checker=check_flow, content_hasher=flow_content_sha256)
