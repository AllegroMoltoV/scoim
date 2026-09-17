"""Finite two-operation compilation from an approved flow to script 0.4.0."""

from pathlib import Path

from .proposal import ProposalRunner
from .script_0_3_compilation import (
    Script03CompilationRequest,
    Script03CompilationResult,
    ScriptCompilationContracts,
    compile_script_with_contracts,
)
from .script_0_4_model_contracts import (
    build_script_document,
    index_structure_candidate,
    relations_prompt,
    relations_response_schema,
    structure_prompt,
    structure_response_schema,
)

Script04CompilationRequest = Script03CompilationRequest
Script04CompilationResult = Script03CompilationResult

_SCRIPT_0_4_CONTRACTS = ScriptCompilationContracts(
    structure_prompt,
    structure_response_schema,
    index_structure_candidate,
    relations_prompt,
    relations_response_schema,
    build_script_document,
)


def compile_script_0_4(
    request: Script04CompilationRequest,
    runner: ProposalRunner,
    run_dir: str | Path,
    *,
    max_new_operations: int | None = None,
) -> Script04CompilationResult:
    """Compile one approved flow while preserving the script 0.4.0 boundary."""
    return compile_script_with_contracts(
        request,
        runner,
        run_dir,
        schema_version="0.4.0",
        contracts=_SCRIPT_0_4_CONTRACTS,
        max_new_operations=max_new_operations,
    )
