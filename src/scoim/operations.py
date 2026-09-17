"""In-memory operations for SCoIM music-script documents."""

from collections.abc import Mapping, Sequence

from .document_operations import (
    ApplyResult,
    ApproveResult,
    apply_document_patch,
    approve_document,
)
from .validation import check, content_sha256


def apply_patch(
    document: Mapping[str, object],
    operations: Sequence[Mapping[str, object]],
    *,
    new_draft: bool = False,
) -> ApplyResult:
    """Apply a patch to a copy and return only a valid updated document."""
    return apply_document_patch(document, operations, checker=check, new_draft=new_draft)


def approve(document: Mapping[str, object]) -> ApproveResult:
    """Return an immutable approved copy of a valid draft document."""
    return approve_document(document, checker=check, content_hasher=content_sha256)
