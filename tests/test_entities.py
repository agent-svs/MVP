"""Entity resolution — must not confidently guess the wrong doctor."""

from __future__ import annotations

import pytest

from core.nlu.entities import entity_store


@pytest.mark.parametrize("surface", ["Bones", "Jonas", "Smyth", "Dr. Steve", "Priya Sharma"])
def test_near_miss_names_do_not_resolve(surface):
    """These all resolved to a real doctor under the old 0.70 ratio rule."""
    assert entity_store.resolve_doctor(surface) is None


@pytest.mark.parametrize("surface,expected", [
    ("Dr. Jones", "Dr. Jones"),
    ("jones", "Dr. Jones"),
    ("Smith", "Dr. Smith"),
    ("dr kulkarni", "Dr. Kulkarni"),
])
def test_real_doctors_resolve(surface, expected):
    assert entity_store.resolve_doctor(surface)["canonical_name"] == expected


def test_close_typo_still_resolves():
    assert entity_store.resolve_doctor("Kulkarnix")["canonical_name"] == "Dr. Kulkarni"


@pytest.mark.parametrize("surface,expected", [
    ("Cardiology", "Cardiology"),
    ("heart", "Cardiology"),
    ("checkup", "General Checkup"),
    ("dental cleaning", "Dental"),
])
def test_services_resolve(surface, expected):
    assert entity_store.resolve_service(surface)["canonical_name"] == expected


@pytest.mark.parametrize("surface", ["Pilates", "crypto mining", "time travel"])
def test_unknown_services_rejected(surface):
    assert entity_store.resolve_service(surface) is None


def test_empty_input_is_safe():
    assert entity_store.resolve_doctor("") is None
    assert entity_store.resolve_entity("date", "2026-08-24") is None


# ── the practitioner-noun regression ────────────────────────
@pytest.mark.parametrize("surface,expected", [
    ("cardiologist", "Cardiology"),
    ("neurologist", "Neurology"),
    ("dermatologist", "Dermatology"),
    ("dentist", "Dental"),
    ("pediatrician", "Pediatrics"),
    ("gynaecologist", "Gynecology"),
    ("optometrist", "Ophthalmology"),
])
def test_practitioner_nouns_resolve_to_their_specialty(surface, expected):
    """People name a specialty by its practitioner. "cardiologist" scores only
    0.82 against "cardiology", so pure typo tolerance rejected all of these —
    which cleared the slot and derailed the whole booking."""
    result = entity_store.resolve_service(surface)
    assert result is not None, f"{surface} must resolve"
    assert result["canonical_name"] == expected


def test_stem_matching_does_not_reintroduce_surname_collisions():
    """The morphological rule must not undo the near-miss protection."""
    for surface in ("Bones", "Jonas", "Smyth"):
        assert entity_store.resolve_doctor(surface) is None
