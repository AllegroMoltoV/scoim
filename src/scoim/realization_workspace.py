"""Immutable work-in-progress state for SCoIM realization experiments."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import jsonpatch
import jsonpointer
import rfc8785

from .realization_script import check_realization_script, realization_script_sha256
from .validation import IssueCode, ValidationIssue

WORKSPACE_SCHEMA_VERSION = 4
_ACCOMPANIMENT_WORKSPACE_SCHEMA_VERSION = 3
_MELODY_WORKSPACE_SCHEMA_VERSION = 2
_LEGACY_WORKSPACE_SCHEMA_VERSION = 1
_HARMONY_QUALITIES = frozenset({"major", "minor", "diminished", "major-seventh"})
_ACCOMPANIMENT_DEGREES = frozenset({"root", "third", "fifth", "seventh"})
_REGISTER_ZONES = frozenset({"bass", "low", "middle", "high"})
_ARTICULATIONS = frozenset({"normal", "staccato", "tenuto", "accent"})
_TIMING_PROFILES = frozenset({"neutral", "savor", "flow", "build", "release"})
_TIMING_AMOUNTS = frozenset({"subtle", "moderate"})
_DYNAMICS_PROFILES = frozenset({"steady", "shape", "build", "release"})
_PERFORMANCE_ARTICULATIONS = frozenset({"score", "legato", "light"})
_COORDINATION_PROFILES = frozenset({"score", "rolled", "aligned"})
_PEDAL_PROFILES = frozenset({"none", "phrase_legato", "harmony_legato", "clear"})


@dataclass(frozen=True, slots=True)
class PlanValue:
    """Music-wide choices that downstream material operations may read."""

    tonal_center: int
    mode: str
    contrasts: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class HarmonyChord:
    """One relative-duration harmony event."""

    duration_units: int
    root_pitch_class: int
    quality: str


@dataclass(frozen=True, slots=True)
class MelodyNote:
    """One note in a workspace melody value."""

    at_units: int
    duration_units: int
    pitch: int


@dataclass(frozen=True, slots=True)
class MelodyValue:
    """A foreground line before accompaniment and performance realization."""

    foreground_voice: str
    notes: tuple[MelodyNote, ...]


@dataclass(frozen=True, slots=True)
class AccompanimentNote:
    """One checked piano accompaniment event and its coarse preferences."""

    at_units: int
    preferred_duration_units: int
    duration_units: int
    degree: str
    preferred_register_zone: str
    pitch: int
    articulations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AccompanimentValue:
    """A deterministically placed piano accompaniment for one foreground line."""

    foreground_voice: str
    notes: tuple[AccompanimentNote, ...]


@dataclass(frozen=True, slots=True)
class PerformanceValue:
    """Musical performance choices attached to one structure node."""

    timing_profile: str | None
    timing_amount: str | None
    dynamics_profile: str | None
    articulation_profile: str | None
    coordination_profile: str | None
    pedal_profile: str | None


@dataclass(frozen=True, slots=True)
class ValueProvenance:
    """Exact state inputs and proposal source used to produce one value."""

    read_hashes: tuple[tuple[str, str], ...]
    proposal_sha256: str


@dataclass(frozen=True, slots=True)
class PatchOperation:
    """One Python-generated RFC 6902 operation."""

    op: str
    path: str
    value: object


@dataclass(frozen=True, slots=True)
class CheckedWorkspaceDiff:
    """A proposal bound to the exact workspace values it read."""

    base_revision: int
    base_workspace_record_sha256: str
    read_hashes: tuple[tuple[str, str], ...]
    write_keys: tuple[str, ...]
    patch: tuple[PatchOperation, ...]
    proposal_sha256: str
    validation_issues: tuple[ValidationIssue, ...]
    write_read_hashes: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] | None = None


@dataclass(frozen=True, slots=True)
class RealizationWorkspace:
    """A deterministic, operation-order-independent music workspace."""

    schema_version: int
    approved_script_sha256: str
    revision: int
    plan: PlanValue | None
    harmonies: tuple[tuple[str, tuple[HarmonyChord, ...]], ...]
    melodies: tuple[tuple[str, MelodyValue], ...]
    transition_melodies: tuple[tuple[str, MelodyValue], ...]
    accompaniments: tuple[tuple[str, AccompanimentValue], ...]
    transition_accompaniments: tuple[tuple[str, AccompanimentValue], ...]
    performances: tuple[tuple[str, PerformanceValue], ...]
    provenance: tuple[tuple[str, ValueProvenance], ...]
    music_content_sha256: str
    workspace_record_sha256: str


@dataclass(frozen=True, slots=True)
class WorkspaceApplyResult:
    """Result of atomically applying one checked workspace diff."""

    workspace: RealizationWorkspace | None
    issues: tuple[ValidationIssue, ...]


def create_workspace(document: dict[str, object]) -> RealizationWorkspace:
    """Create an empty workspace for one valid approved SCoIM script."""
    result = check_realization_script(document)
    if not result.valid:
        raise ValueError("A valid realization SCoIM script is required")
    script_hash = realization_script_sha256(document)
    workspace = RealizationWorkspace(
        schema_version=WORKSPACE_SCHEMA_VERSION,
        approved_script_sha256=script_hash,
        revision=0,
        plan=None,
        harmonies=(),
        melodies=(),
        transition_melodies=(),
        accompaniments=(),
        transition_accompaniments=(),
        performances=(),
        provenance=(),
        music_content_sha256="",
        workspace_record_sha256="",
    )
    return _with_current_hashes(workspace)


def workspace_to_dict(workspace: RealizationWorkspace) -> dict[str, object]:
    """Return the canonical JSON-compatible workspace record."""
    return {
        **_record_dict(workspace),
        "workspace_record_sha256": workspace.workspace_record_sha256,
    }


def workspace_from_dict(value: Mapping[str, object]) -> RealizationWorkspace:
    """Restore and verify one canonical workspace record."""
    schema_version = value.get("schema_version")
    expected_fields = {
        "schema_version",
        "approved_script_sha256",
        "revision",
        "plan",
        "harmonies",
        "provenance",
        "music_content_sha256",
        "workspace_record_sha256",
    }
    if schema_version == WORKSPACE_SCHEMA_VERSION:
        expected_fields |= {
            "melodies",
            "transition_melodies",
            "accompaniments",
            "transition_accompaniments",
            "performances",
        }
    elif schema_version == _ACCOMPANIMENT_WORKSPACE_SCHEMA_VERSION:
        expected_fields |= {
            "melodies",
            "transition_melodies",
            "accompaniments",
            "transition_accompaniments",
        }
    elif schema_version == _MELODY_WORKSPACE_SCHEMA_VERSION:
        expected_fields |= {"melodies", "transition_melodies"}
    elif schema_version != _LEGACY_WORKSPACE_SCHEMA_VERSION:
        raise ValueError("Unsupported realization workspace record")
    if set(value) != expected_fields:
        raise ValueError("Unsupported realization workspace record")
    provenance_value = value["provenance"]
    if not isinstance(provenance_value, Mapping):
        raise ValueError("provenance must be an object")
    provenance: list[tuple[str, ValueProvenance]] = []
    for key, raw_source in sorted(provenance_value.items()):
        if not isinstance(key, str) or not isinstance(raw_source, Mapping):
            raise ValueError("Invalid provenance entry")
        raw_reads = raw_source.get("read_hashes")
        if not isinstance(raw_reads, list):
            raise ValueError("read_hashes must be an array")
        reads: list[tuple[str, str]] = []
        for raw_read in raw_reads:
            if (
                not isinstance(raw_read, Mapping)
                or not isinstance(raw_read.get("key"), str)
                or not isinstance(raw_read.get("sha256"), str)
            ):
                raise ValueError("Invalid provenance read hash")
            reads.append((cast(str, raw_read["key"]), cast(str, raw_read["sha256"])))
        proposal_hash = raw_source.get("proposal_sha256")
        if not isinstance(proposal_hash, str):
            raise ValueError("Invalid provenance proposal hash")
        provenance.append((key, ValueProvenance(tuple(sorted(reads)), proposal_hash)))
    script_hash = value["approved_script_sha256"]
    revision = value["revision"]
    music_hash = value["music_content_sha256"]
    record_hash = value["workspace_record_sha256"]
    if (
        not isinstance(script_hash, str)
        or not isinstance(revision, int)
        or revision < 0
        or not isinstance(music_hash, str)
        or not isinstance(record_hash, str)
    ):
        raise ValueError("Invalid workspace identity or revision")
    workspace = RealizationWorkspace(
        schema_version=cast(int, schema_version),
        approved_script_sha256=script_hash,
        revision=revision,
        plan=_parse_plan(value["plan"]),
        harmonies=_parse_harmonies(value["harmonies"]),
        melodies=(
            _parse_melodies(value["melodies"])
            if schema_version
            in {
                _MELODY_WORKSPACE_SCHEMA_VERSION,
                _ACCOMPANIMENT_WORKSPACE_SCHEMA_VERSION,
                WORKSPACE_SCHEMA_VERSION,
            }
            else ()
        ),
        transition_melodies=(
            _parse_melodies(value["transition_melodies"])
            if schema_version
            in {
                _MELODY_WORKSPACE_SCHEMA_VERSION,
                _ACCOMPANIMENT_WORKSPACE_SCHEMA_VERSION,
                WORKSPACE_SCHEMA_VERSION,
            }
            else ()
        ),
        accompaniments=(
            _parse_accompaniments(value["accompaniments"])
            if schema_version in {_ACCOMPANIMENT_WORKSPACE_SCHEMA_VERSION, WORKSPACE_SCHEMA_VERSION}
            else ()
        ),
        transition_accompaniments=(
            _parse_accompaniments(value["transition_accompaniments"])
            if schema_version in {_ACCOMPANIMENT_WORKSPACE_SCHEMA_VERSION, WORKSPACE_SCHEMA_VERSION}
            else ()
        ),
        performances=(
            _parse_performances(value["performances"])
            if schema_version == WORKSPACE_SCHEMA_VERSION
            else ()
        ),
        provenance=tuple(provenance),
        music_content_sha256=music_hash,
        workspace_record_sha256=record_hash,
    )
    expected = _with_current_hashes(workspace)
    if expected.music_content_sha256 != music_hash:
        raise ValueError("music_content_sha256 does not match the workspace")
    if expected.workspace_record_sha256 != record_hash:
        raise ValueError("workspace_record_sha256 does not match the workspace")
    return workspace


def upgrade_workspace_v1_to_v2(workspace: RealizationWorkspace) -> RealizationWorkspace:
    """Upgrade one verified legacy workspace without mutating its saved record."""
    if workspace.schema_version != _LEGACY_WORKSPACE_SCHEMA_VERSION:
        raise ValueError("Only a version 1 workspace can be upgraded")
    upgraded = RealizationWorkspace(
        schema_version=_MELODY_WORKSPACE_SCHEMA_VERSION,
        approved_script_sha256=workspace.approved_script_sha256,
        revision=workspace.revision,
        plan=workspace.plan,
        harmonies=workspace.harmonies,
        melodies=(),
        transition_melodies=(),
        accompaniments=(),
        transition_accompaniments=(),
        performances=(),
        provenance=workspace.provenance,
        music_content_sha256="",
        workspace_record_sha256="",
    )
    return _with_current_hashes(upgraded)


def upgrade_workspace_v2_to_v3(workspace: RealizationWorkspace) -> RealizationWorkspace:
    """Upgrade one verified melody workspace without mutating its saved record."""
    if workspace.schema_version != _MELODY_WORKSPACE_SCHEMA_VERSION:
        raise ValueError("Only a version 2 workspace can be upgraded")
    upgraded = RealizationWorkspace(
        schema_version=_ACCOMPANIMENT_WORKSPACE_SCHEMA_VERSION,
        approved_script_sha256=workspace.approved_script_sha256,
        revision=workspace.revision,
        plan=workspace.plan,
        harmonies=workspace.harmonies,
        melodies=workspace.melodies,
        transition_melodies=workspace.transition_melodies,
        accompaniments=(),
        transition_accompaniments=(),
        performances=(),
        provenance=workspace.provenance,
        music_content_sha256="",
        workspace_record_sha256="",
    )
    return _with_current_hashes(upgraded)


def upgrade_workspace_v3_to_v4(workspace: RealizationWorkspace) -> RealizationWorkspace:
    """Upgrade one verified accompaniment workspace without mutating its saved record."""
    if workspace.schema_version != _ACCOMPANIMENT_WORKSPACE_SCHEMA_VERSION:
        raise ValueError("Only a version 3 workspace can be upgraded")
    upgraded = RealizationWorkspace(
        schema_version=WORKSPACE_SCHEMA_VERSION,
        approved_script_sha256=workspace.approved_script_sha256,
        revision=workspace.revision,
        plan=workspace.plan,
        harmonies=workspace.harmonies,
        melodies=workspace.melodies,
        transition_melodies=workspace.transition_melodies,
        accompaniments=workspace.accompaniments,
        transition_accompaniments=workspace.transition_accompaniments,
        performances=(),
        provenance=workspace.provenance,
        music_content_sha256="",
        workspace_record_sha256="",
    )
    return _with_current_hashes(upgraded)


def checked_diff_to_dict(checked_diff: CheckedWorkspaceDiff) -> dict[str, object]:
    """Return the JSON-compatible form of one checked diff."""
    value: dict[str, object] = {
        "base_revision": checked_diff.base_revision,
        "base_workspace_record_sha256": checked_diff.base_workspace_record_sha256,
        "read_hashes": [{"key": key, "sha256": digest} for key, digest in checked_diff.read_hashes],
        "write_keys": list(checked_diff.write_keys),
        "patch": [
            {"op": operation.op, "path": operation.path, "value": operation.value}
            for operation in checked_diff.patch
        ],
        "proposal_sha256": checked_diff.proposal_sha256,
        "validation_issues": [
            {"code": issue.code.value, "message": issue.message, "path": issue.path}
            for issue in checked_diff.validation_issues
        ],
    }
    if checked_diff.write_read_hashes is not None:
        value["write_read_hashes"] = {
            write_key: [{"key": read_key, "sha256": digest} for read_key, digest in read_hashes]
            for write_key, read_hashes in checked_diff.write_read_hashes
        }
    return value


def checked_diff_from_dict(value: Mapping[str, object]) -> CheckedWorkspaceDiff:
    """Restore one checked diff from its immutable event value."""
    raw_reads = cast(list[Mapping[str, str]], value["read_hashes"])
    raw_writes = cast(list[str], value["write_keys"])
    raw_patch = cast(list[Mapping[str, object]], value["patch"])
    raw_issues = cast(list[Mapping[str, str]], value["validation_issues"])
    raw_write_reads = value.get("write_read_hashes")
    write_read_hashes: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] | None = None
    if raw_write_reads is not None:
        if not isinstance(raw_write_reads, Mapping):
            raise ValueError("write_read_hashes must be an object")
        parsed_write_reads: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        for write_key, raw_dependencies in sorted(raw_write_reads.items()):
            if not isinstance(write_key, str) or not isinstance(raw_dependencies, list):
                raise ValueError("Invalid write dependency entry")
            dependencies: list[tuple[str, str]] = []
            for raw_dependency in raw_dependencies:
                if (
                    not isinstance(raw_dependency, Mapping)
                    or not isinstance(raw_dependency.get("key"), str)
                    or not isinstance(raw_dependency.get("sha256"), str)
                ):
                    raise ValueError("Invalid write dependency hash")
                dependencies.append(
                    (
                        cast(str, raw_dependency["key"]),
                        cast(str, raw_dependency["sha256"]),
                    )
                )
            parsed_write_reads.append((write_key, tuple(sorted(dependencies))))
        write_read_hashes = tuple(parsed_write_reads)
    return CheckedWorkspaceDiff(
        base_revision=cast(int, value["base_revision"]),
        base_workspace_record_sha256=cast(str, value["base_workspace_record_sha256"]),
        read_hashes=tuple((item["key"], item["sha256"]) for item in raw_reads),
        write_keys=tuple(raw_writes),
        patch=tuple(
            PatchOperation(
                op=cast(str, item["op"]),
                path=cast(str, item["path"]),
                value=item.get("value"),
            )
            for item in raw_patch
        ),
        proposal_sha256=cast(str, value["proposal_sha256"]),
        validation_issues=tuple(
            ValidationIssue(
                code=IssueCode(item["code"]),
                message=item["message"],
                path=item["path"],
            )
            for item in raw_issues
        ),
        write_read_hashes=write_read_hashes,
    )


def state_key_sha256(workspace: RealizationWorkspace, key: str) -> str:
    """Hash one concrete RFC 6901 workspace value."""
    try:
        value = jsonpointer.resolve_pointer(_music_dict(workspace), key)
    except jsonpointer.JsonPointerException as error:
        raise ValueError(f"Unknown workspace key: {key}") from error
    return _sha256(value)


def apply_checked_diff(
    workspace: RealizationWorkspace,
    checked_diff: CheckedWorkspaceDiff,
) -> WorkspaceApplyResult:
    """Apply a validated diff only when its exact inputs are still current."""
    if checked_diff.validation_issues:
        return WorkspaceApplyResult(None, checked_diff.validation_issues)
    if checked_diff.base_revision != workspace.revision:
        return _apply_failure(
            IssueCode.STALE_REVISION,
            "The workspace revision has changed",
            "/revision",
        )
    if checked_diff.base_workspace_record_sha256 != workspace.workspace_record_sha256:
        return _apply_failure(
            IssueCode.LINEAGE_MISMATCH,
            "The workspace record hash has changed",
            "/workspace_record_sha256",
        )
    for key, expected_hash in checked_diff.read_hashes:
        try:
            actual_hash = state_key_sha256(workspace, key)
        except ValueError:
            return _apply_failure(
                IssueCode.NOT_FOUND,
                "A read key does not exist",
                key,
            )
        if actual_hash != expected_hash:
            return _apply_failure(
                IssueCode.LINEAGE_MISMATCH,
                "A value read by the proposal has changed",
                key,
            )
    patch_paths = tuple(operation.path for operation in checked_diff.patch)
    if patch_paths != checked_diff.write_keys or len(set(patch_paths)) != len(patch_paths):
        return _apply_failure(
            IssueCode.PATCH_INVALID,
            "Patch paths must exactly match the declared write keys",
            "/patch",
        )
    write_dependencies = _write_dependencies(checked_diff)
    if write_dependencies is None:
        return _apply_failure(
            IssueCode.PATCH_INVALID,
            "Write dependencies must exactly cover the write keys and read current inputs",
            "/write_read_hashes",
        )
    write_key_set = set(checked_diff.write_keys)
    if any(
        dependency_key in write_key_set and dependency_key != write_key
        for write_key, dependencies in write_dependencies.items()
        for dependency_key, _ in dependencies
    ):
        return _apply_failure(
            IssueCode.PATCH_INVALID,
            "A diff cannot change a dependency and its dependent value together",
            "/write_read_hashes",
        )
    original_music = _music_dict(workspace)
    updated_music = original_music
    operations = [
        {"op": operation.op, "path": operation.path, "value": operation.value}
        for operation in checked_diff.patch
    ]
    try:
        updated_music = cast(
            dict[str, object],
            jsonpatch.apply_patch(updated_music, operations, in_place=False),
        )
    except (KeyError, TypeError, ValueError, jsonpatch.JsonPatchException) as error:
        return _apply_failure(IssueCode.PATCH_INVALID, str(error), "/patch")
    provenance = dict(workspace.provenance)
    pending_invalidations: list[tuple[str, str]] = []
    for key in checked_diff.write_keys:
        old_value = _resolve_optional(original_music, key)
        new_value = _resolve_optional(updated_music, key)
        if old_value is not _MISSING and old_value != new_value:
            pending_invalidations.append((key, _sha256(old_value)))
    handled_invalidations: set[tuple[str, str]] = set()
    while pending_invalidations:
        changed = pending_invalidations.pop()
        if changed in handled_invalidations:
            continue
        handled_invalidations.add(changed)
        for key, source in tuple(provenance.items()):
            if key in write_key_set or changed not in source.read_hashes:
                continue
            old_value = _resolve_optional(updated_music, key)
            if old_value is _MISSING:
                provenance.pop(key)
                continue
            try:
                updated_music = cast(
                    dict[str, object],
                    jsonpatch.apply_patch(
                        updated_music,
                        [{"op": "remove", "path": key}],
                        in_place=False,
                    ),
                )
            except jsonpatch.JsonPatchException as error:
                return _apply_failure(IssueCode.PATCH_INVALID, str(error), key)
            provenance.pop(key)
            pending_invalidations.append((key, _sha256(old_value)))
    try:
        plan = _parse_plan(updated_music["plan"])
        harmonies = _parse_harmonies(updated_music["harmonies"])
        melodies = (
            _parse_melodies(updated_music["melodies"])
            if workspace.schema_version
            in {
                _MELODY_WORKSPACE_SCHEMA_VERSION,
                _ACCOMPANIMENT_WORKSPACE_SCHEMA_VERSION,
                WORKSPACE_SCHEMA_VERSION,
            }
            else ()
        )
        transition_melodies = (
            _parse_melodies(updated_music["transition_melodies"])
            if workspace.schema_version
            in {
                _MELODY_WORKSPACE_SCHEMA_VERSION,
                _ACCOMPANIMENT_WORKSPACE_SCHEMA_VERSION,
                WORKSPACE_SCHEMA_VERSION,
            }
            else ()
        )
        accompaniments = (
            _parse_accompaniments(updated_music["accompaniments"])
            if workspace.schema_version
            in {_ACCOMPANIMENT_WORKSPACE_SCHEMA_VERSION, WORKSPACE_SCHEMA_VERSION}
            else ()
        )
        transition_accompaniments = (
            _parse_accompaniments(updated_music["transition_accompaniments"])
            if workspace.schema_version
            in {_ACCOMPANIMENT_WORKSPACE_SCHEMA_VERSION, WORKSPACE_SCHEMA_VERSION}
            else ()
        )
        performances = (
            _parse_performances(updated_music["performances"])
            if workspace.schema_version == WORKSPACE_SCHEMA_VERSION
            else ()
        )
    except (KeyError, TypeError, ValueError) as error:
        return _apply_failure(IssueCode.PATCH_INVALID, str(error), "/patch")
    for key in checked_diff.write_keys:
        provenance[key] = ValueProvenance(
            read_hashes=write_dependencies[key],
            proposal_sha256=checked_diff.proposal_sha256,
        )
    updated = RealizationWorkspace(
        schema_version=workspace.schema_version,
        approved_script_sha256=workspace.approved_script_sha256,
        revision=workspace.revision + 1,
        plan=plan,
        harmonies=harmonies,
        melodies=melodies,
        transition_melodies=transition_melodies,
        accompaniments=accompaniments,
        transition_accompaniments=transition_accompaniments,
        performances=performances,
        provenance=tuple(sorted(provenance.items())),
        music_content_sha256="",
        workspace_record_sha256="",
    )
    return WorkspaceApplyResult(_with_current_hashes(updated), ())


def replay_checked_diffs(
    initial: RealizationWorkspace,
    checked_diffs: tuple[CheckedWorkspaceDiff, ...],
) -> WorkspaceApplyResult:
    """Rebuild a workspace by applying its checked diffs in recorded order."""
    current = initial
    for checked_diff in checked_diffs:
        result = apply_checked_diff(current, checked_diff)
        if result.workspace is None:
            return result
        current = result.workspace
    return WorkspaceApplyResult(current, ())


def _apply_failure(code: IssueCode, message: str, path: str) -> WorkspaceApplyResult:
    return WorkspaceApplyResult(
        workspace=None,
        issues=(ValidationIssue(code=code, message=message, path=path),),
    )


def _write_dependencies(
    checked_diff: CheckedWorkspaceDiff,
) -> dict[str, tuple[tuple[str, str], ...]] | None:
    if checked_diff.write_read_hashes is None:
        return {key: tuple(sorted(checked_diff.read_hashes)) for key in checked_diff.write_keys}
    dependencies = dict(checked_diff.write_read_hashes)
    if len(dependencies) != len(checked_diff.write_read_hashes) or set(dependencies) != set(
        checked_diff.write_keys
    ):
        return None
    allowed_reads = set(checked_diff.read_hashes)
    for reads in dependencies.values():
        if len(set(reads)) != len(reads) or not set(reads).issubset(allowed_reads):
            return None
    return {key: tuple(sorted(reads)) for key, reads in dependencies.items()}


_MISSING = object()


def _resolve_optional(value: Mapping[str, object], key: str) -> object:
    try:
        return jsonpointer.resolve_pointer(value, key)
    except jsonpointer.JsonPointerException:
        return _MISSING


def _parse_plan(value: object) -> PlanValue | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "tonal_center",
        "mode",
        "contrasts",
    }:
        raise ValueError("Plan must contain tonal_center, mode, and contrasts")
    tonal_center = value["tonal_center"]
    mode = value["mode"]
    contrasts = value["contrasts"]
    if (
        isinstance(tonal_center, bool)
        or not isinstance(tonal_center, int)
        or not 0 <= tonal_center <= 11
    ):
        raise ValueError("tonal_center must be an integer from 0 through 11")
    if mode not in {"major", "minor"}:
        raise ValueError("mode must be major or minor")
    if not isinstance(contrasts, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in contrasts.items()
    ):
        raise ValueError("contrasts must map section keys to descriptions")
    return PlanValue(
        tonal_center=tonal_center,
        mode=cast(str, mode),
        contrasts=tuple(sorted(cast(Mapping[str, str], contrasts).items())),
    )


def _parse_harmonies(
    value: object,
) -> tuple[tuple[str, tuple[HarmonyChord, ...]], ...]:
    if not isinstance(value, Mapping):
        raise ValueError("harmonies must be an object")
    parsed: list[tuple[str, tuple[HarmonyChord, ...]]] = []
    for material_key, raw_chords in sorted(value.items()):
        if not isinstance(material_key, str) or not isinstance(raw_chords, list):
            raise ValueError("harmonies must map material keys to chord arrays")
        chords: list[HarmonyChord] = []
        for raw_chord in raw_chords:
            if not isinstance(raw_chord, Mapping) or set(raw_chord) != {
                "duration_units",
                "root_pitch_class",
                "quality",
            }:
                raise ValueError("A harmony chord must be an object")
            duration = raw_chord["duration_units"]
            root = raw_chord["root_pitch_class"]
            quality = raw_chord["quality"]
            if isinstance(duration, bool) or not isinstance(duration, int) or duration < 1:
                raise ValueError("Harmony duration_units must be a positive integer")
            if isinstance(root, bool) or not isinstance(root, int) or not 0 <= root <= 11:
                raise ValueError("Harmony root_pitch_class must be from 0 through 11")
            if not isinstance(quality, str) or quality not in _HARMONY_QUALITIES:
                raise ValueError("Harmony quality is unsupported")
            chords.append(
                HarmonyChord(
                    duration_units=duration,
                    root_pitch_class=root,
                    quality=quality,
                )
            )
        if not chords:
            raise ValueError("A harmony sequence must not be empty")
        parsed.append((material_key, tuple(chords)))
    return tuple(parsed)


def _plan_dict(plan: PlanValue | None) -> dict[str, object] | None:
    if plan is None:
        return None
    return {
        "tonal_center": plan.tonal_center,
        "mode": plan.mode,
        "contrasts": dict(plan.contrasts),
    }


def _harmony_dict(chord: HarmonyChord) -> dict[str, object]:
    return {
        "duration_units": chord.duration_units,
        "root_pitch_class": chord.root_pitch_class,
        "quality": chord.quality,
    }


def _parse_melodies(value: object) -> tuple[tuple[str, MelodyValue], ...]:
    if not isinstance(value, Mapping):
        raise ValueError("melodies must be an object")
    parsed: list[tuple[str, MelodyValue]] = []
    for key, raw_melody in sorted(value.items()):
        if not isinstance(key, str) or not isinstance(raw_melody, Mapping):
            raise ValueError("melodies must map keys to melody objects")
        if set(raw_melody) != {"foreground_voice", "notes"}:
            raise ValueError("A melody must contain foreground_voice and notes")
        voice = raw_melody["foreground_voice"]
        raw_notes = raw_melody["notes"]
        if voice not in {"upper", "lower"} or not isinstance(raw_notes, list) or not raw_notes:
            raise ValueError("A melody must have a supported voice and non-empty notes")
        notes: list[MelodyNote] = []
        for raw_note in raw_notes:
            if not isinstance(raw_note, Mapping) or set(raw_note) != {
                "at_units",
                "duration_units",
                "pitch",
            }:
                raise ValueError("A melody note must contain timing and pitch")
            at_units = raw_note["at_units"]
            duration_units = raw_note["duration_units"]
            pitch = raw_note["pitch"]
            if (
                isinstance(at_units, bool)
                or not isinstance(at_units, int)
                or at_units < 0
                or isinstance(duration_units, bool)
                or not isinstance(duration_units, int)
                or duration_units < 1
                or isinstance(pitch, bool)
                or not isinstance(pitch, int)
                or not 21 <= pitch <= 108
            ):
                raise ValueError("A melody note is outside the supported range")
            notes.append(MelodyNote(at_units, duration_units, pitch))
        parsed.append((key, MelodyValue(cast(str, voice), tuple(notes))))
    return tuple(parsed)


def _melody_dict(melody: MelodyValue) -> dict[str, object]:
    return {
        "foreground_voice": melody.foreground_voice,
        "notes": [
            {
                "at_units": note.at_units,
                "duration_units": note.duration_units,
                "pitch": note.pitch,
            }
            for note in melody.notes
        ],
    }


def _parse_accompaniments(value: object) -> tuple[tuple[str, AccompanimentValue], ...]:
    if not isinstance(value, Mapping):
        raise ValueError("accompaniments must be an object")
    parsed: list[tuple[str, AccompanimentValue]] = []
    note_fields = {
        "at_units",
        "preferred_duration_units",
        "duration_units",
        "degree",
        "preferred_register_zone",
        "pitch",
        "articulations",
    }
    for key, raw_accompaniment in sorted(value.items()):
        if not isinstance(key, str) or not isinstance(raw_accompaniment, Mapping):
            raise ValueError("accompaniments must map keys to accompaniment objects")
        if set(raw_accompaniment) != {"foreground_voice", "notes"}:
            raise ValueError("An accompaniment must contain foreground_voice and notes")
        voice = raw_accompaniment["foreground_voice"]
        raw_notes = raw_accompaniment["notes"]
        if voice not in {"upper", "lower"} or not isinstance(raw_notes, list) or not raw_notes:
            raise ValueError("An accompaniment must have a supported voice and non-empty notes")
        notes: list[AccompanimentNote] = []
        for raw_note in raw_notes:
            if not isinstance(raw_note, Mapping) or set(raw_note) != note_fields:
                raise ValueError("An accompaniment note has unsupported fields")
            at_units = raw_note["at_units"]
            preferred_duration = raw_note["preferred_duration_units"]
            duration = raw_note["duration_units"]
            degree = raw_note["degree"]
            zone = raw_note["preferred_register_zone"]
            pitch = raw_note["pitch"]
            articulations = raw_note["articulations"]
            if (
                isinstance(at_units, bool)
                or not isinstance(at_units, int)
                or at_units < 0
                or isinstance(preferred_duration, bool)
                or not isinstance(preferred_duration, int)
                or preferred_duration < 1
                or isinstance(duration, bool)
                or not isinstance(duration, int)
                or not 1 <= duration <= preferred_duration
                or degree not in _ACCOMPANIMENT_DEGREES
                or zone not in _REGISTER_ZONES
                or isinstance(pitch, bool)
                or not isinstance(pitch, int)
                or not 21 <= pitch <= 108
                or not isinstance(articulations, list)
                or len(set(articulations)) != len(articulations)
                or any(item not in _ARTICULATIONS for item in articulations)
            ):
                raise ValueError("An accompaniment note is outside the supported range")
            notes.append(
                AccompanimentNote(
                    at_units,
                    preferred_duration,
                    duration,
                    cast(str, degree),
                    cast(str, zone),
                    pitch,
                    tuple(cast(list[str], articulations)),
                )
            )
        parsed.append((key, AccompanimentValue(cast(str, voice), tuple(notes))))
    return tuple(parsed)


def _accompaniment_dict(accompaniment: AccompanimentValue) -> dict[str, object]:
    return {
        "foreground_voice": accompaniment.foreground_voice,
        "notes": [
            {
                "at_units": note.at_units,
                "preferred_duration_units": note.preferred_duration_units,
                "duration_units": note.duration_units,
                "degree": note.degree,
                "preferred_register_zone": note.preferred_register_zone,
                "pitch": note.pitch,
                "articulations": list(note.articulations),
            }
            for note in accompaniment.notes
        ],
    }


def _parse_performances(value: object) -> tuple[tuple[str, PerformanceValue], ...]:
    if not isinstance(value, Mapping):
        raise ValueError("performances must be an object")
    parsed: list[tuple[str, PerformanceValue]] = []
    fields = {
        "timing_profile",
        "timing_amount",
        "dynamics_profile",
        "articulation_profile",
        "coordination_profile",
        "pedal_profile",
    }
    supported = {
        "timing_profile": _TIMING_PROFILES,
        "timing_amount": _TIMING_AMOUNTS,
        "dynamics_profile": _DYNAMICS_PROFILES,
        "articulation_profile": _PERFORMANCE_ARTICULATIONS,
        "coordination_profile": _COORDINATION_PROFILES,
        "pedal_profile": _PEDAL_PROFILES,
    }
    for node_id, raw_performance in sorted(value.items()):
        if not isinstance(node_id, str) or not isinstance(raw_performance, Mapping):
            raise ValueError("performances must map node IDs to performance objects")
        if set(raw_performance) != fields:
            raise ValueError("A performance must contain all supported fields")
        for field, choices in supported.items():
            item = raw_performance[field]
            if item is not None and item not in choices:
                raise ValueError(f"Performance {field} is unsupported")
        if (
            raw_performance["timing_profile"] is None
            and raw_performance["timing_amount"] is not None
        ):
            raise ValueError("Performance timing amount requires a timing profile")
        parsed.append(
            (
                node_id,
                PerformanceValue(
                    timing_profile=cast(str | None, raw_performance["timing_profile"]),
                    timing_amount=cast(str | None, raw_performance["timing_amount"]),
                    dynamics_profile=cast(str | None, raw_performance["dynamics_profile"]),
                    articulation_profile=cast(str | None, raw_performance["articulation_profile"]),
                    coordination_profile=cast(str | None, raw_performance["coordination_profile"]),
                    pedal_profile=cast(str | None, raw_performance["pedal_profile"]),
                ),
            )
        )
    return tuple(parsed)


def _performance_dict(performance: PerformanceValue) -> dict[str, object]:
    return {
        "timing_profile": performance.timing_profile,
        "timing_amount": performance.timing_amount,
        "dynamics_profile": performance.dynamics_profile,
        "articulation_profile": performance.articulation_profile,
        "coordination_profile": performance.coordination_profile,
        "pedal_profile": performance.pedal_profile,
    }


def _music_dict(workspace: RealizationWorkspace) -> dict[str, object]:
    value: dict[str, object] = {
        "approved_script_sha256": workspace.approved_script_sha256,
        "plan": _plan_dict(workspace.plan),
        "harmonies": {
            key: [_harmony_dict(chord) for chord in chords] for key, chords in workspace.harmonies
        },
    }
    if workspace.schema_version in {
        _MELODY_WORKSPACE_SCHEMA_VERSION,
        _ACCOMPANIMENT_WORKSPACE_SCHEMA_VERSION,
        WORKSPACE_SCHEMA_VERSION,
    }:
        value["melodies"] = {key: _melody_dict(melody) for key, melody in workspace.melodies}
        value["transition_melodies"] = {
            key: _melody_dict(melody) for key, melody in workspace.transition_melodies
        }
    if workspace.schema_version in {
        _ACCOMPANIMENT_WORKSPACE_SCHEMA_VERSION,
        WORKSPACE_SCHEMA_VERSION,
    }:
        value["accompaniments"] = {
            key: _accompaniment_dict(accompaniment)
            for key, accompaniment in workspace.accompaniments
        }
        value["transition_accompaniments"] = {
            key: _accompaniment_dict(accompaniment)
            for key, accompaniment in workspace.transition_accompaniments
        }
    if workspace.schema_version == WORKSPACE_SCHEMA_VERSION:
        value["performances"] = {
            key: _performance_dict(performance) for key, performance in workspace.performances
        }
    return value


def _record_dict(workspace: RealizationWorkspace) -> dict[str, object]:
    return {
        "schema_version": workspace.schema_version,
        **_music_dict(workspace),
        "revision": workspace.revision,
        "provenance": {
            key: {
                "read_hashes": [
                    {"key": read_key, "sha256": digest}
                    for read_key, digest in provenance.read_hashes
                ],
                "proposal_sha256": provenance.proposal_sha256,
            }
            for key, provenance in workspace.provenance
        },
        "music_content_sha256": workspace.music_content_sha256,
    }


def _with_current_hashes(workspace: RealizationWorkspace) -> RealizationWorkspace:
    music_hash = _sha256(_music_dict(workspace))
    with_music_hash = RealizationWorkspace(
        schema_version=workspace.schema_version,
        approved_script_sha256=workspace.approved_script_sha256,
        revision=workspace.revision,
        plan=workspace.plan,
        harmonies=workspace.harmonies,
        melodies=workspace.melodies,
        transition_melodies=workspace.transition_melodies,
        accompaniments=workspace.accompaniments,
        transition_accompaniments=workspace.transition_accompaniments,
        performances=workspace.performances,
        provenance=workspace.provenance,
        music_content_sha256=music_hash,
        workspace_record_sha256="",
    )
    return RealizationWorkspace(
        schema_version=with_music_hash.schema_version,
        approved_script_sha256=with_music_hash.approved_script_sha256,
        revision=with_music_hash.revision,
        plan=with_music_hash.plan,
        harmonies=with_music_hash.harmonies,
        melodies=with_music_hash.melodies,
        transition_melodies=with_music_hash.transition_melodies,
        accompaniments=with_music_hash.accompaniments,
        transition_accompaniments=with_music_hash.transition_accompaniments,
        performances=with_music_hash.performances,
        provenance=with_music_hash.provenance,
        music_content_sha256=music_hash,
        workspace_record_sha256=_sha256(_record_dict(with_music_hash)),
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(rfc8785.dumps(value)).hexdigest()
