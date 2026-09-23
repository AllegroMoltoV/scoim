import json
import zipfile
from pathlib import Path

import pytest
from test_scoim_phase3_realization import SequencedRunner, _document, _phase2_ledger, _responses
from test_scoim_phase6_state import _phase6_run
from test_scoim_phase7_realization import _response as _performance_response

from llm_musical_composer.run_state import StateConflictError
from scoim.phase3_realization import Phase3Request, realize_phase3
from scoim.phase4_realization import Phase4Request, realize_phase4
from scoim.phase5_realization import Phase5Request, realize_phase5
from scoim.phase6_realization import Phase6Request, realize_phase6
from scoim.phase7_realization import Phase7Request, realize_phase7
from scoim.phase8_bundle import Phase8BundleRequest, create_phase8_bundle
from scoim.projection_ledger import ProjectionLedgerEntry

_LEGACY_BUNDLE = Path(__file__).parent / "fixtures/scoim/legacy-context-733a97e/bundle.zip"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def legacy_runs(tmp_path: Path) -> Path:
    with zipfile.ZipFile(_LEGACY_BUNDLE) as archive:
        archive.extractall(tmp_path / "legacy")
    return tmp_path / "legacy/model-runs"


@pytest.mark.parametrize("accepted", [True, False], ids=["accepted", "before-acceptance"])
def test_phase3_rejects_legacy_context_before_reusing_a_completed_attempt(
    legacy_runs: Path, accepted: bool
) -> None:
    run_dir = legacy_runs / "phase3"
    accepted_path = run_dir / "events/overall-plan/accepted.json"
    record = _read(accepted_path)
    attempt = run_dir / record["accepted_attempt"]
    assert _read(attempt / "terminal.json")["status"] == "completed"
    assert (attempt / "response.staged.json").is_file()
    if not accepted:
        accepted_path.unlink()
    document = _read(run_dir / "inputs/validated-script.json")
    ledger = tuple(
        ProjectionLedgerEntry(**entry) for entry in _read(run_dir / "inputs/projection-ledger.json")
    )
    runner = SequencedRunner([])

    with pytest.raises(StateConflictError, match="generation context contract"):
        realize_phase3(Phase3Request(document, ledger), runner, run_dir)

    assert runner.prompts == []
    assert accepted_path.exists() is accepted


def test_phase3_current_context_resume_reuses_all_accepted_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scoim.phase3_realization.secrets.randbelow", lambda upper: 0)
    document = _document()
    request = Phase3Request(document, _phase2_ledger(document))
    run_dir = tmp_path / "phase3"
    first = realize_phase3(request, SequencedRunner(_responses()), run_dir)
    assert first.realized
    runner = SequencedRunner([])

    resumed = realize_phase3(request, runner, run_dir)

    assert resumed.realized
    assert runner.prompts == []
    assert resumed.state == first.state


@pytest.mark.parametrize("accepted", [True, False], ids=["accepted", "before-acceptance"])
def test_phase7_rejects_legacy_run_even_with_identical_current_upstream_values(
    tmp_path: Path, accepted: bool
) -> None:
    phase6_dir = _phase6_run(tmp_path)
    run_dir = tmp_path / "phase7"
    generated = realize_phase7(
        Phase7Request(phase6_dir), SequencedRunner([_performance_response()]), run_dir
    )
    assert generated.realized
    spec_path = run_dir / "run-spec.json"
    spec = _read(spec_path)
    spec.pop("generation_context_contract")
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    accepted_paths = list((run_dir / "events").glob("*/accepted.json"))
    assert len(accepted_paths) == 1
    accepted_path = accepted_paths[0]
    attempt = run_dir / _read(accepted_path)["accepted_attempt"]
    assert _read(attempt / "terminal.json")["status"] == "completed"
    assert (attempt / "response.staged.json").is_file()
    if not accepted:
        accepted_path.unlink()
    runner = SequencedRunner([])

    with pytest.raises(StateConflictError, match="run-spec"):
        realize_phase7(Phase7Request(phase6_dir), runner, run_dir)

    assert runner.prompts == []
    assert accepted_path.exists() is accepted


@pytest.mark.parametrize("phase", [4, 5, 6, 7])
def test_later_generation_rejects_genuine_legacy_context_upstream(
    legacy_runs: Path, tmp_path: Path, phase: int
) -> None:
    runner = SequencedRunner([])
    destination = tmp_path / f"new-phase{phase}"
    if phase == 7:
        result = realize_phase7(Phase7Request(legacy_runs / "phase6"), runner, destination)
    else:
        inputs = legacy_runs / f"phase{phase}/inputs"
        values = [
            _read(inputs / "validated-script.json"),
            _read(inputs / "phase3-state.json"),
        ]
        values.extend(_read(inputs / f"phase{number}-state.json") for number in range(4, phase))
        values.append(_read(inputs / "projection-ledger.json"))
        if phase == 4:
            result = realize_phase4(Phase4Request(*values), runner, destination)
        elif phase == 5:
            result = realize_phase5(Phase5Request(*values), runner, destination)
        else:
            result = realize_phase6(Phase6Request(*values), destination)

    assert not result.realized
    assert result.outcome == "request_invalid"
    assert "generation context contract" in result.issues[0].message
    assert runner.prompts == []
    assert not destination.exists()


def test_new_bundle_creation_rejects_genuine_legacy_context_runs(
    legacy_runs: Path, tmp_path: Path
) -> None:
    bundle = legacy_runs.parent
    manifest = _read(bundle / "manifest.json")
    destination = tmp_path / "new-bundle"

    result = create_phase8_bundle(
        Phase8BundleRequest(
            phase7_run_dir=legacy_runs / "phase7",
            phase_run_dirs={f"phase{i}": legacy_runs / f"phase{i}" for i in range(3, 7)},
            composition_id=manifest["composition_id"],
            trial_id="new-trial",
            composition_manifest=(bundle / "lineage/composition-manifest.json").read_bytes(),
        ),
        destination,
    )

    assert not result.created
    assert "generation context contract" in result.issues[0].message
    assert not destination.exists()
