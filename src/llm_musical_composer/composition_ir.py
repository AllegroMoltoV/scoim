"""制限付き作曲記法を受け取るための型付き内部表現。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Note:
    event_id: str
    at_ms: int
    duration_ms: int
    pitch: int
    velocity: int
    voice: str | None = None


@dataclass(frozen=True)
class Pedal:
    event_id: str
    at_ms: int
    value: int


@dataclass(frozen=True)
class Material:
    material_id: str
    duration_ms: int
    notes: tuple[Note, ...]
    pedals: tuple[Pedal, ...] = ()
    derived_from: str | None = None


@dataclass(frozen=True)
class Use:
    material_id: str
    role: str | None = None
    energy: int | None = None
    attack_style: str | None = None


@dataclass(frozen=True)
class TonicEnding:
    duration_ms: int


@dataclass(frozen=True)
class Part:
    part_id: str
    role: str
    energy: int
    start_use_index: int
    end_use_index: int


@dataclass(frozen=True)
class Phrase:
    phrase_id: str
    part_id: str
    role: str
    derived_from: str | None
    start_use_index: int
    end_use_index: int
    variation_kind: str | None = None


@dataclass(frozen=True)
class Composition:
    title: str
    form: tuple[Use, ...]
    materials: tuple[Material, ...]
    tonal_center: int | None = None
    mode: str | None = None
    ending: TonicEnding | None = None
    parts: tuple[Part, ...] = ()
    phrases: tuple[Phrase, ...] = ()

    @property
    def material_by_id(self) -> dict[str, Material]:
        return {material.material_id: material for material in self.materials}

    @property
    def body_duration_ms(self) -> int:
        materials = self.material_by_id
        return sum(materials[section.material_id].duration_ms for section in self.form)

    @property
    def duration_ms(self) -> int:
        ending_duration = self.ending.duration_ms if self.ending is not None else 0
        return self.body_duration_ms + ending_duration

    @property
    def body_note_count(self) -> int:
        materials = self.material_by_id
        return sum(len(materials[section.material_id].notes) for section in self.form)

    @property
    def note_count(self) -> int:
        ending_note_count = 4 if self.ending is not None else 0
        return self.body_note_count + ending_note_count
