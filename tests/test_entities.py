"""
Tests for Canonical Entity Resolution.
"""

from core.nlu.entities import entity_store


def test_resolve_exact_doctor():
    """Exact doctor names should resolve to the correct ID."""
    res = entity_store.resolve_doctor("Dr. Kulkarni")
    assert res is not None
    assert res["id"] == "DOC_001"
    assert res["canonical_name"] == "Dr. Kulkarni"


def test_resolve_fuzzy_doctor():
    """Fuzzy doctor names (misspellings/variants) should resolve."""
    # Substring / simple misspelling
    res1 = entity_store.resolve_doctor("dr kulikarmi")
    assert res1 is not None
    assert res1["id"] == "DOC_001"
    
    # Substring of an alias
    res2 = entity_store.resolve_doctor("kulkarni")
    assert res2 is not None
    assert res2["id"] == "DOC_001"


def test_resolve_unknown_doctor():
    """Unknown doctors should return None."""
    assert entity_store.resolve_doctor("Dr. Who") is None
    assert entity_store.resolve_doctor("Frank") is None


def test_resolve_exact_service():
    """Exact service names should resolve to the correct ID."""
    res = entity_store.resolve_service("General Checkup")
    assert res is not None
    assert res["id"] == "SRV_001"


def test_resolve_fuzzy_service():
    """Fuzzy service names should resolve."""
    res1 = entity_store.resolve_service("check up")
    assert res1 is not None
    assert res1["id"] == "SRV_001"
    
    res2 = entity_store.resolve_service("consult")
    assert res2 is not None
    assert res2["id"] == "SRV_002"


def test_resolve_general_entity():
    """General resolve_entity method should route correctly."""
    dr_res = entity_store.resolve_entity("doctor_name", "SMITH")
    assert dr_res is not None
    assert dr_res["id"] == "DOC_002"
    
    srv_res = entity_store.resolve_entity("service_type", "shot")
    assert srv_res is not None
    assert srv_res["id"] == "SRV_004"
    
    # Unsupported slot type
    assert entity_store.resolve_entity("time", "5 PM") is None
