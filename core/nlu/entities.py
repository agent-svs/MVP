import re
from typing import Optional, Dict, Any, List
# We can use simple Levenshtein distance for fuzzy matching.
# Python's difflib provides SequenceMatcher which is built-in and sufficient.
from difflib import SequenceMatcher

class EntityStore:
    """Canonical entity store for resolving surface text to backend IDs."""
    
    def __init__(self):
        # In a real system, these would be loaded from a database or API.
        # For this prototype, we'll hardcode some logical canonical entities.
        
        self.doctors: List[Dict[str, Any]] = [
            {"id": "DOC_001", "name": "Dr. Kulkarni", "specialty": "General Practice", "aliases": ["kulkarni", "kulikarmi", "dr k"]},
            {"id": "DOC_002", "name": "Dr. Smith", "specialty": "Pediatrics", "aliases": ["smith", "dr s"]},
            {"id": "DOC_003", "name": "Dr. Jones", "specialty": "Cardiology", "aliases": ["jones", "dr j"]},
        ]
        
        self.services: List[Dict[str, Any]] = [
            {"id": "SRV_001", "name": "General Checkup", "aliases": ["checkup", "check up", "check-up", "physical"]},
            {"id": "SRV_002", "name": "Consultation", "aliases": ["consult", "consultation", "advice"]},
            {"id": "SRV_003", "name": "Follow-up", "aliases": ["follow up", "followup", "review"]},
            {"id": "SRV_004", "name": "Vaccination", "aliases": ["vaccine", "shot", "immunization"]},
        ]

    def _fuzzy_match(self, surface_text: str, candidates: List[str], threshold: float = 0.7) -> bool:
        """Check if surface text closely matches any of the candidates."""
        surface_text = surface_text.lower().strip()
        for candidate in candidates:
            # Exact match is always true
            if surface_text == candidate.lower():
                return True
            
            # Simple substring match (e.g. "kulkarni" inside "dr kulkarni")
            if candidate.lower() in surface_text or surface_text in candidate.lower():
                # Only accept substring if it's a significant portion
                if len(surface_text) >= 4 and len(candidate) >= 4:
                    return True
                    
            # Levenshtein / Sequence matching for typos
            ratio = SequenceMatcher(None, surface_text, candidate.lower()).ratio()
            if ratio >= threshold:
                return True
                
        return False

    def resolve_doctor(self, surface_text: str) -> Optional[Dict[str, Any]]:
        """Resolve a raw doctor name to a canonical doctor entity."""
        if not surface_text:
            return None
            
        # Clean up common prefixes for better matching against aliases
        clean_text = re.sub(r'^(dr\.?|doctor)\s+', '', surface_text, flags=re.IGNORECASE).strip()
        
        for doc in self.doctors:
            candidates = [doc["name"]] + doc.get("aliases", [])
            if self._fuzzy_match(surface_text, candidates) or self._fuzzy_match(clean_text, candidates):
                return {"id": doc["id"], "canonical_name": doc["name"]}
                
        return None

    def resolve_service(self, surface_text: str) -> Optional[Dict[str, Any]]:
        """Resolve a raw service request to a canonical service entity."""
        if not surface_text:
            return None
            
        for srv in self.services:
            candidates = [srv["name"]] + srv.get("aliases", [])
            if self._fuzzy_match(surface_text, candidates):
                return {"id": srv["id"], "canonical_name": srv["name"]}
                
        return None

    def resolve_entity(self, slot_type: str, surface_text: str) -> Optional[Dict[str, Any]]:
        """Main entry point to resolve any supported entity type."""
        if slot_type == "doctor_name":
            return self.resolve_doctor(surface_text)
        elif slot_type == "service_type":
            return self.resolve_service(surface_text)
        # Dates and times are already canonicalized by dateutil in slots.py
        # Names and phone numbers are free-form strings
        return None

# Global instance for easy access
entity_store = EntityStore()
