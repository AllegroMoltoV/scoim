"""通常の Python を実行せず、制限付き作曲記法だけを解析する。"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from typing import Any

from llm_musical_composer.composition_ir import (
    Composition,
    Material,
    Note,
    Part,
    Pedal,
    Phrase,
    TonicEnding,
    Use,
)


class DslError(ValueError):
    """作曲記法が構文または意味上の制約に違反した場合の例外。"""


@dataclass(frozen=True)
class ValidationPolicy:
    minimum_duration_ms: int
    maximum_duration_ms: int
    maximum_note_count: int
    minimum_part_count: int = 0


SHORT_FORM_POLICY = ValidationPolicy(30_000, 60_000, 512)
THREE_MINUTE_POLICY = ValidationPolicy(180_000, 180_000, 2_048, 3)


def _literal(node: ast.AST, expected: type[int] | type[str]) -> int | str:
    if not isinstance(node, ast.Constant) or type(node.value) is not expected:
        raise DslError(f"expected a literal {expected.__name__}")
    return node.value


def _call(node: ast.AST, name: str) -> ast.Call:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise DslError("only direct calls to allowed DSL functions are accepted")
    if node.func.id != name:
        raise DslError(f"unknown DSL function: {node.func.id}")
    if any(keyword.arg is None for keyword in node.keywords):
        raise DslError("keyword expansion is not accepted")
    return node


def _arguments(
    call: ast.Call,
    *,
    positional: tuple[str, ...],
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> dict[str, ast.AST]:
    if len(call.args) > len(positional):
        raise DslError("too many positional arguments")
    result = dict(zip(positional, call.args, strict=False))
    for keyword in call.keywords:
        assert keyword.arg is not None
        if keyword.arg in result:
            raise DslError(f"duplicate argument: {keyword.arg}")
        result[keyword.arg] = keyword.value
    accepted = set(positional) | required | optional
    unknown = set(result) - accepted
    if unknown:
        raise DslError(f"unknown argument: {sorted(unknown)[0]}")
    missing = required - result.keys()
    if missing:
        raise DslError(f"missing argument: {sorted(missing)[0]}")
    return result


def _list(node: ast.AST, builder: Any) -> tuple[Any, ...]:
    if not isinstance(node, ast.List):
        raise DslError("expected a list literal")
    return tuple(builder(item) for item in node.elts)


def _parse_use(node: ast.AST) -> Use:
    args = _arguments(
        _call(node, "use"),
        positional=("material_id",),
        required=frozenset(),
        optional=frozenset({"role", "energy", "attack_style"}),
    )
    role = args.get("role")
    energy = args.get("energy")
    attack_style = args.get("attack_style")
    return Use(
        material_id=str(_literal(args["material_id"], str)),
        role=None if role is None else str(_literal(role, str)),
        energy=None if energy is None else int(_literal(energy, int)),
        attack_style=(None if attack_style is None else str(_literal(attack_style, str))),
    )


def _parse_note(node: ast.AST) -> Note:
    args = _arguments(
        _call(node, "note"),
        positional=("event_id", "pitch", "velocity", "at_ms", "duration_ms"),
        required=frozenset({"at_ms", "duration_ms", "pitch", "velocity"}),
        optional=frozenset({"voice"}),
    )
    voice = args.get("voice")
    return Note(
        event_id=str(_literal(args["event_id"], str)),
        at_ms=int(_literal(args["at_ms"], int)),
        duration_ms=int(_literal(args["duration_ms"], int)),
        pitch=int(_literal(args["pitch"], int)),
        velocity=int(_literal(args["velocity"], int)),
        voice=None if voice is None else str(_literal(voice, str)),
    )


def _parse_pedal(node: ast.AST) -> Pedal:
    try:
        call = _call(node, "pedal")
    except DslError:
        call = _call(node, "cc64")
    args = _arguments(
        call,
        positional=("event_id",),
        required=frozenset({"at_ms", "value"}),
    )
    return Pedal(
        event_id=str(_literal(args["event_id"], str)),
        at_ms=int(_literal(args["at_ms"], int)),
        value=int(_literal(args["value"], int)),
    )


def _parse_material(node: ast.AST) -> Material:
    args = _arguments(
        _call(node, "material"),
        positional=("material_id",),
        required=frozenset({"duration_ms", "notes"}),
        optional=frozenset({"pedals", "derived_from"}),
    )
    pedals_node = args.get("pedals")
    derived_from_node = args.get("derived_from")
    return Material(
        material_id=str(_literal(args["material_id"], str)),
        duration_ms=int(_literal(args["duration_ms"], int)),
        notes=_list(args["notes"], _parse_note),
        pedals=() if pedals_node is None else _list(pedals_node, _parse_pedal),
        derived_from=(None if derived_from_node is None else str(_literal(derived_from_node, str))),
    )


def _parse_tonic_ending(node: ast.AST) -> TonicEnding:
    args = _arguments(
        _call(node, "tonic_hold"),
        positional=(),
        required=frozenset({"duration_ms"}),
    )
    return TonicEnding(duration_ms=int(_literal(args["duration_ms"], int)))


def _parse_phrase(
    node: ast.AST, *, part_id: str, start_use_index: int
) -> tuple[Phrase, tuple[Use, ...]]:
    args = _arguments(
        _call(node, "phrase"),
        positional=("phrase_id",),
        required=frozenset({"role", "uses"}),
        optional=frozenset({"derived_from", "variation_kind"}),
    )
    uses = _list(args["uses"], _parse_use)
    if not uses:
        raise DslError("phrase uses must not be empty")
    derived_from_node = args.get("derived_from")
    variation_kind_node = args.get("variation_kind")
    phrase = Phrase(
        phrase_id=str(_literal(args["phrase_id"], str)),
        part_id=part_id,
        role=str(_literal(args["role"], str)),
        derived_from=(None if derived_from_node is None else str(_literal(derived_from_node, str))),
        start_use_index=start_use_index,
        end_use_index=start_use_index + len(uses),
        variation_kind=(
            None if variation_kind_node is None else str(_literal(variation_kind_node, str))
        ),
    )
    return phrase, uses


def _parse_part(
    node: ast.AST, start_use_index: int
) -> tuple[Part, tuple[Phrase, ...], tuple[Use, ...]]:
    args = _arguments(
        _call(node, "part"),
        positional=("part_id",),
        required=frozenset({"role", "energy"}),
        optional=frozenset({"uses", "phrases"}),
    )
    uses_node = args.get("uses")
    phrases_node = args.get("phrases")
    if (uses_node is None) == (phrases_node is None):
        raise DslError("exactly one of uses or phrases must be specified")
    part_id = str(_literal(args["part_id"], str))
    phrases: list[Phrase] = []
    if phrases_node is not None:
        if not isinstance(phrases_node, ast.List):
            raise DslError("expected a list literal")
        nested_uses: list[Use] = []
        for phrase_node in phrases_node.elts:
            phrase, phrase_uses = _parse_phrase(
                phrase_node,
                part_id=part_id,
                start_use_index=start_use_index + len(nested_uses),
            )
            phrases.append(phrase)
            nested_uses.extend(phrase_uses)
        uses = tuple(nested_uses)
    else:
        assert uses_node is not None
        uses = _list(uses_node, _parse_use)
    if not uses:
        raise DslError("part uses must not be empty")
    part = Part(
        part_id=part_id,
        role=str(_literal(args["role"], str)),
        energy=int(_literal(args["energy"], int)),
        start_use_index=start_use_index,
        end_use_index=start_use_index + len(uses),
    )
    return part, tuple(phrases), uses


def _parse_parts(
    node: ast.AST,
) -> tuple[tuple[Part, ...], tuple[Phrase, ...], tuple[Use, ...]]:
    if not isinstance(node, ast.List):
        raise DslError("expected a list literal")
    parts: list[Part] = []
    phrases: list[Phrase] = []
    form: list[Use] = []
    for item in node.elts:
        part, part_phrases, uses = _parse_part(item, len(form))
        parts.append(part)
        phrases.extend(part_phrases)
        form.extend(uses)
    return tuple(parts), tuple(phrases), tuple(form)


def _validate_section_contract(composition: Composition, *, require_section_contract: bool) -> None:
    contract_fields = [(section.role, section.energy) for section in composition.form]
    for values in contract_fields:
        if any(value is not None for value in values) and not all(
            value is not None for value in values
        ):
            raise DslError("role and energy must be specified together")
    complete = [all(value is not None for value in values) for values in contract_fields]
    if not any(complete):
        if require_section_contract:
            raise DslError("section contract is required for every use")
        return
    if not all(complete):
        raise DslError("section contract is required for every use")

    allowed_roles = {"opening", "contrast", "transition", "return", "climax", "release"}
    allowed_attack_styles = {"clustered", "distributed"}
    legacy_attack_styles = [section.attack_style is not None for section in composition.form]
    if any(legacy_attack_styles) and not all(legacy_attack_styles):
        raise DslError("attack_style must be specified for every use or omitted from every use")
    for section in composition.form:
        if section.role not in allowed_roles:
            raise DslError("role is not allowed")
        if section.energy is None or not 1 <= section.energy <= 5:
            raise DslError("energy must be between 1 and 5")
        if section.attack_style is not None and section.attack_style not in allowed_attack_styles:
            raise DslError("attack_style is not allowed")
    if composition.form[0].role != "opening":
        raise DslError("the first section role must be opening")
    climaxes = [section for section in composition.form if section.role == "climax"]
    if len(climaxes) != 1:
        raise DslError("section contract must contain exactly one climax")
    climax = climaxes[0]
    if composition.parts:
        climax_parts = [part for part in composition.parts if part.role == "climax"]
        if len(climax_parts) == 1:
            climax_part = climax_parts[0]
            if any(
                index < climax_part.start_use_index or index >= climax_part.end_use_index
                for index, section in enumerate(composition.form)
                if section.energy is not None and section.energy >= climax.energy
            ):
                raise DslError("highest energy uses must stay inside the climax part")
    elif any(
        section is not climax and section.energy is not None and section.energy >= climax.energy
        for section in composition.form
    ):
        raise DslError("climax must have the unique highest energy")
    declared_by_material: dict[str, tuple[int | None, str | None]] = {}
    for section in composition.form:
        declaration = (section.energy, section.attack_style)
        previous = declared_by_material.setdefault(section.material_id, declaration)
        if previous != declaration:
            raise DslError("reused material must preserve energy and attack_style")


def _validate_parts(composition: Composition, *, minimum_part_count: int) -> None:
    if not composition.parts:
        if minimum_part_count:
            raise DslError("parts are required by this validation policy")
        return
    if len(composition.parts) < minimum_part_count:
        raise DslError(f"parts must contain at least {minimum_part_count} entries")
    part_ids = [part.part_id for part in composition.parts]
    if any(not part_id for part_id in part_ids) or len(set(part_ids)) != len(part_ids):
        raise DslError("part IDs must be non-empty and unique")
    allowed_roles = {"opening", "development", "climax", "return", "release"}
    for part in composition.parts:
        if part.start_use_index >= part.end_use_index:
            raise DslError("part uses must not be empty")
        if part.role not in allowed_roles:
            raise DslError("part role is not allowed")
        if not 1 <= part.energy <= 5:
            raise DslError("part energy must be between 1 and 5")
    if composition.parts[0].role != "opening":
        raise DslError("the first part role must be opening")
    climaxes = [part for part in composition.parts if part.role == "climax"]
    if len(climaxes) != 1:
        raise DslError("parts must contain exactly one climax")
    climax = climaxes[0]
    if climax is composition.parts[0] or climax is composition.parts[-1]:
        raise DslError("the climax part must not be first or last")
    if any(part is not climax and part.energy >= climax.energy for part in composition.parts):
        raise DslError("climax part must have the unique highest energy")
    following_parts = composition.parts[composition.parts.index(climax) + 1 :]
    if not any(part.energy < climax.energy for part in following_parts):
        raise DslError("a lower-energy part must follow the climax")
    climax_use_indices = [
        index for index, use in enumerate(composition.form) if use.role == "climax"
    ]
    if climax_use_indices and not all(
        climax.start_use_index <= index < climax.end_use_index for index in climax_use_indices
    ):
        raise DslError("climax use must be inside the climax part")


def _validate_phrases(composition: Composition) -> None:
    if not composition.phrases:
        return
    phrase_ids = [phrase.phrase_id for phrase in composition.phrases]
    if any(not phrase_id for phrase_id in phrase_ids) or len(set(phrase_ids)) != len(phrase_ids):
        raise DslError("phrase IDs must be non-empty and unique")
    allowed_roles = {"statement", "variation", "contrast", "return"}
    allowed_variation_kinds = {"rhythmic", "textural", "registral"}
    earlier: set[str] = set()
    parts_by_id = {part.part_id: part for part in composition.parts}
    for phrase in composition.phrases:
        if phrase.role not in allowed_roles:
            raise DslError("phrase role is not allowed")
        if phrase.variation_kind is not None:
            if phrase.role != "variation":
                raise DslError("only a variation phrase may specify variation_kind")
            if phrase.variation_kind not in allowed_variation_kinds:
                raise DslError("variation_kind is not allowed")
        if phrase.start_use_index >= phrase.end_use_index:
            raise DslError("phrase uses must not be empty")
        part = parts_by_id.get(phrase.part_id)
        if part is None or not (
            part.start_use_index
            <= phrase.start_use_index
            < phrase.end_use_index
            <= part.end_use_index
        ):
            raise DslError("phrase use range must stay inside its part")
        if phrase.role in {"variation", "return"}:
            if phrase.derived_from is None:
                raise DslError(f"{phrase.role} phrase requires derived_from")
            if phrase.derived_from not in earlier:
                raise DslError("derived_from must name an earlier phrase")
        elif phrase.derived_from is not None:
            raise DslError("only variation and return phrases may specify derived_from")
        earlier.add(phrase.phrase_id)

    phrases_by_id = {phrase.phrase_id: phrase for phrase in composition.phrases}
    material_map = composition.material_by_id
    for phrase in composition.phrases:
        if phrase.derived_from is None:
            continue
        source = phrases_by_id[phrase.derived_from]
        source_materials = {
            use.material_id
            for use in composition.form[source.start_use_index : source.end_use_index]
            if use.role != "transition"
        }
        related_count = 0
        for use in composition.form[phrase.start_use_index : phrase.end_use_index]:
            if use.role == "transition":
                continue
            if use.material_id in source_materials:
                if phrase.role == "variation":
                    raise DslError(
                        "variation phrase must use derived material instead of an exact return"
                    )
                related_count += 1
                continue
            material = material_map[use.material_id]
            if material.derived_from not in source_materials:
                raise DslError("derived phrase material must come from its source phrase")
            related_count += 1
        if not related_count:
            raise DslError("derived phrase must contain a related material")


def _validate(
    composition: Composition,
    *,
    require_section_contract: bool,
    policy: ValidationPolicy,
) -> None:
    if not composition.title.strip():
        raise DslError("title must not be empty")
    if len(composition.form) < 3:
        raise DslError("form must contain at least three uses")
    material_ids = [material.material_id for material in composition.materials]
    if len(set(material_ids)) != len(material_ids):
        raise DslError("material IDs must be unique")
    material_map = composition.material_by_id
    material_positions = {material_id: index for index, material_id in enumerate(material_ids)}
    for index, material in enumerate(composition.materials):
        if material.derived_from is None:
            continue
        if material.derived_from not in material_map:
            raise DslError(f"unknown material in derived_from: {material.derived_from}")
        if material_positions[material.derived_from] >= index:
            raise DslError("derived_from must name an earlier material")
    for use in composition.form:
        if use.material_id not in material_map:
            raise DslError(f"unknown material in form: {use.material_id}")
    if max(Counter(use.material_id for use in composition.form).values(), default=0) < 2:
        raise DslError("form must reuse at least one material")
    _validate_section_contract(composition, require_section_contract=require_section_contract)
    _validate_parts(composition, minimum_part_count=policy.minimum_part_count)
    _validate_phrases(composition)

    contract_values = (composition.tonal_center, composition.mode, composition.ending)
    if any(value is not None for value in contract_values) and not all(
        value is not None for value in contract_values
    ):
        raise DslError("tonal_center, mode, and ending must be specified together")
    if composition.tonal_center is not None and not 0 <= composition.tonal_center <= 11:
        raise DslError("tonal_center must be between 0 and 11")
    if composition.mode is not None and composition.mode not in {"major", "minor"}:
        raise DslError("mode must be major or minor")
    if composition.ending is not None and not 3000 <= composition.ending.duration_ms <= 6000:
        raise DslError("ending duration must be between 3000 and 6000 ms")

    event_ids: set[str] = set()
    for material in composition.materials:
        if not material.material_id or material.duration_ms <= 0:
            raise DslError("material ID and duration must be valid")
        intervals_by_pitch: dict[int, list[tuple[int, int]]] = {}
        for event in (*material.notes, *material.pedals):
            if not event.event_id or event.event_id in event_ids:
                raise DslError("event IDs must be unique")
            event_ids.add(event.event_id)
        for note in material.notes:
            if note.at_ms < 0 or note.duration_ms <= 0:
                raise DslError("note timing must be positive")
            if note.at_ms + note.duration_ms > material.duration_ms:
                raise DslError("note must fit inside its material")
            if not 21 <= note.pitch <= 108:
                raise DslError("pitch must be between 21 and 108")
            if not 0 <= note.velocity <= 127:
                raise DslError("velocity must be between 0 and 127")
            if note.voice is not None and note.voice not in {"upper", "lower"}:
                raise DslError("voice must be upper or lower")
            intervals_by_pitch.setdefault(note.pitch, []).append(
                (note.at_ms, note.at_ms + note.duration_ms)
            )
        for intervals in intervals_by_pitch.values():
            for previous, current in zip(sorted(intervals), sorted(intervals)[1:], strict=False):
                if current[0] < previous[1]:
                    raise DslError("overlapping notes of the same pitch are ambiguous")
        for pedal in material.pedals:
            if not 0 <= pedal.at_ms <= material.duration_ms:
                raise DslError("pedal must fit inside its material")
            if not 0 <= pedal.value <= 127:
                raise DslError("pedal value must be between 0 and 127")

    if not policy.minimum_duration_ms <= composition.duration_ms <= policy.maximum_duration_ms:
        raise DslError(
            "expanded duration must be between "
            f"{policy.minimum_duration_ms} and {policy.maximum_duration_ms} ms"
        )
    if composition.note_count > policy.maximum_note_count:
        raise DslError(f"expanded note count must not exceed {policy.maximum_note_count}")


def parse_composition(
    source: str,
    *,
    require_section_contract: bool = False,
    require_parts: bool = False,
    policy: ValidationPolicy = SHORT_FORM_POLICY,
) -> Composition:
    """許可した式だけを型付き内部表現へ変換する。"""
    try:
        root = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise DslError(f"invalid syntax at line {error.lineno}: {error.msg}") from error
    call = _call(root.body, "composition")
    args = _arguments(
        call,
        positional=(),
        required=frozenset({"title", "materials"}),
        optional=frozenset({"form", "parts", "tonal_center", "mode", "ending"}),
    )
    form_node = args.get("form")
    parts_node = args.get("parts")
    if (form_node is None) == (parts_node is None):
        raise DslError("exactly one of form or parts must be specified")
    if parts_node is not None:
        parts, phrases, form = _parse_parts(parts_node)
    else:
        assert form_node is not None
        parts = ()
        phrases = ()
        form = _list(form_node, _parse_use)
    if require_parts and not parts:
        raise DslError("parts are required")
    tonal_center_node = args.get("tonal_center")
    mode_node = args.get("mode")
    ending_node = args.get("ending")
    composition = Composition(
        title=str(_literal(args["title"], str)),
        form=form,
        materials=_list(args["materials"], _parse_material),
        tonal_center=None if tonal_center_node is None else int(_literal(tonal_center_node, int)),
        mode=None if mode_node is None else str(_literal(mode_node, str)),
        ending=None if ending_node is None else _parse_tonic_ending(ending_node),
        parts=parts,
        phrases=phrases,
    )
    _validate(
        composition,
        require_section_contract=require_section_contract,
        policy=policy,
    )
    return composition
