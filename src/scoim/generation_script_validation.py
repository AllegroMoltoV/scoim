"""Version routing for script documents accepted by the staged v2 runtime."""

from collections.abc import Mapping

from .script_0_3_validation import check_script_0_3_document
from .script_0_4_validation import check_script_0_4_document
from .validation import CheckResult, IssueCode, ValidationIssue


def check_generation_script_document(document: Mapping[str, object]) -> CheckResult:
    """Validate a current script or a preserved script needed for offline replay."""
    schema_version = document.get("schema_version")
    if schema_version == "0.4.0":
        return check_script_0_4_document(document)
    if schema_version == "0.3.0":
        return check_script_0_3_document(document)
    return CheckResult(
        False,
        (
            ValidationIssue(
                IssueCode.UNSUPPORTED_SCHEMA_VERSION,
                f"Unsupported generation script schema version: {schema_version!r}",
                "/schema_version",
            ),
        ),
    )
