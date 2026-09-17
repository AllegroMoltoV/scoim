"""Validation for immutable SCoIM script 0.2.0 documents."""

import hashlib
import json
from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from typing import cast

import rfc8785
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource

from .validation import CheckResult, IssueCode, ValidationIssue, check_script_semantics

SCRIPT_DOCUMENT_TYPE = "script"
SCRIPT_SCHEMA_VERSION = "0.2.0"
_CONTENT_FIELDS = (
    "document_type",
    "schema_version",
    "document_id",
    "revision",
    "source_flow",
    "script",
)


def check_script_document(document: Mapping[str, object]) -> CheckResult:
    """Check a new validated script without accepting legacy documents."""
    schema_version = document.get("schema_version")
    if isinstance(schema_version, str) and schema_version != SCRIPT_SCHEMA_VERSION:
        return CheckResult(
            False,
            (
                ValidationIssue(
                    IssueCode.UNSUPPORTED_SCHEMA_VERSION,
                    f"Unsupported script schema version: {schema_version!r}",
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
    script = cast(dict[str, object], document["script"])
    semantic_issues = check_script_semantics(script)
    return CheckResult(not semantic_issues, semantic_issues)


def script_content_sha256(document: Mapping[str, object]) -> str:
    """Hash the immutable content and source-flow lineage of a script."""
    content = {field: document[field] for field in _CONTENT_FIELDS}
    return hashlib.sha256(rfc8785.dumps(content)).hexdigest()


@cache
def _validator() -> Draft202012Validator:
    schema_root = files("scoim").joinpath("schemas")
    legacy_schema = json.loads(schema_root.joinpath("script-0.1.0.schema.json").read_text("utf-8"))
    schema = json.loads(schema_root.joinpath("script-0.2.0.schema.json").read_text("utf-8"))
    legacy_resource = Resource.from_contents(legacy_schema)
    registry = Registry().with_resource(cast(str, legacy_schema["$id"]), legacy_resource)
    return Draft202012Validator(schema, registry=registry)


def _error_pointer(error: ValidationError) -> str:
    return "".join(f"/{_escape_pointer_token(part)}" for part in error.absolute_path)


def _escape_pointer_token(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")
