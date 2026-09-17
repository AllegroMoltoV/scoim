import scoim
from scoim.codex_proposal import CodexStructuredRunner
from scoim.public_realization import PublicRealizationResult, realize
from scoim.validation import IssueCode, ValidationIssue


def test_public_package_exposes_only_the_realization_boundary() -> None:
    expected_names = {
        "CodexStructuredRunner",
        "IssueCode",
        "PublicRealizationResult",
        "ValidationIssue",
        "realize",
    }

    assert set(scoim.__all__) == expected_names
    assert scoim.CodexStructuredRunner is CodexStructuredRunner
    assert scoim.IssueCode is IssueCode
    assert scoim.PublicRealizationResult is PublicRealizationResult
    assert scoim.ValidationIssue is ValidationIssue
    assert scoim.realize is realize
    assert not hasattr(scoim, "compose_flow")
    assert not hasattr(scoim, "create_model_trial")
    assert not hasattr(scoim, "replay_trial_bundle")
    assert not hasattr(scoim, "realize_solo_piano_3m")
