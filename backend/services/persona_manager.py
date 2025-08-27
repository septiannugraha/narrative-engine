"""
Persona Manager - Save and switch between player characters
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class Persona:
    """A saved player character persona"""
    
    def __init__(self, name: str, description: str, tags: List[str] = None):
        self.name = name
        self.description = description
        self.tags = tags or []
        self.created_at = datetime.now()
        self.last_used = datetime.now()
        self.play_count = 0
        self.favorite = False
        
        # Extended attributes for rich personas
        self.background = ""
        self.personality_traits = []
        self.physical_details = {}
        self.preferred_actions = []  # Common actions this persona takes
        self.speech_style = ""  # How they talk
        self.relationships = {}  # Remembered relationships
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving"""
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "play_count": self.play_count,
            "favorite": self.favorite,
            "background": self.background,
            "personality_traits": self.personality_traits,
            "physical_details": self.physical_details,
            "preferred_actions": self.preferred_actions,
            "speech_style": self.speech_style,
            "relationships": self.relationships
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Persona':
        """Create from dictionary"""
        persona = cls(
            name=data["name"],
            description=data["description"],
            tags=data.get("tags", [])
        )
        persona.created_at = datetime.fromisoformat(data["created_at"])
        persona.last_used = datetime.fromisoformat(data["last_used"])
        persona.play_count = data.get("play_count", 0)
        persona.favorite = data.get("favorite", False)
        persona.background = data.get("background", "")
        persona.personality_traits = data.get("personality_traits", [])
        persona.physical_details = data.get("physical_details", {})
        persona.preferred_actions = data.get("preferred_actions", [])
        persona.speech_style = data.get("speech_style", "")
        persona.relationships = data.get("relationships", {})
        return persona


class PersonaManager:
    """Manages saved personas for quick character switching"""
    
    def __init__(self, save_dir: str = "saved_personas"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.personas: Dict[str, Persona] = {}
        self.current_persona: Optional[Persona] = None
        self.load_all_personas()
        
        # Default personas for new users
        self._create_default_personas()
    
    def _create_default_personas(self):
        """Create some default personas if none exist"""
        if len(self.personas) == 0:
            # Charming Rogue
            self.create_persona(
                name="Cassian",
                description="A charming rogue with quick wit and quicker hands. Dark hair, green eyes, and a devil-may-care smile.",
                tags=["male", "rogue", "charming", "witty"],
                extended_attrs={
                    "background": "Former noble turned adventurer after a family scandal",
                    "personality_traits": ["flirtatious", "clever", "risk-taker", "loyal to friends"],
                    "physical_details": {
                        "height": "5'10\"",
                        "build": "lean and agile",
                        "hair": "dark brown, slightly messy",
                        "eyes": "emerald green",
                        "notable_features": "scar through left eyebrow"
                    },
                    "speech_style": "Playful and teasing, with occasional formal language from noble upbringing",
                    "preferred_actions": ["flirt", "sneak", "charm", "gamble"]
                }
            )
            
            # Mysterious Sorceress
            self.create_persona(
                name="Lyralei",
                description="A mysterious sorceress with flowing silver hair and violet eyes that seem to see through souls.",
                tags=["female", "mage", "mysterious", "intelligent"],
                extended_attrs={
                    "background": "Trained at the Celestial Academy, seeking forbidden knowledge",
                    "personality_traits": ["intelligent", "curious", "secretly passionate", "guarded"],
                    "physical_details": {
                        "height": "5'6\"",
                        "build": "slender and graceful",
                        "hair": "long silver hair",
                        "eyes": "violet, almost luminescent",
                        "notable_features": "arcane tattoos on arms"
                    },
                    "speech_style": "Elegant and precise, occasionally slips into ancient languages",
                    "preferred_actions": ["study", "cast spells", "seduce intellectually", "explore mysteries"]
                }
            )
            
            # Strong Warrior Woman
            self.create_persona(
                name="Valeria",
                description="A fierce warrior woman with sun-kissed skin and muscles earned through countless battles.",
                tags=["female", "warrior", "strong", "direct"],
                extended_attrs={
                    "background": "Gladiator champion seeking freedom and purpose",
                    "personality_traits": ["brave", "direct", "protective", "surprisingly gentle"],
                    "physical_details": {
                        "height": "5'9\"",
                        "build": "athletic and muscular",
                        "hair": "long red hair in warrior braids",
                        "eyes": "steel gray",
                        "notable_features": "battle scars, tribal tattoos"
                    },
                    "speech_style": "Direct and honest, actions speak louder than words",
                    "preferred_actions": ["fight", "protect", "challenge", "passionate encounters"]
                }
            )
    
    def create_persona(self, name: str, description: str, tags: List[str] = None,
                      extended_attrs: Dict = None) -> Persona:
        """Create a new persona"""
        if name in self.personas:
            logger.warning(f"Persona {name} already exists, updating instead")
        
        persona = Persona(name, description, tags)
        
        # Add extended attributes if provided
        if extended_attrs:
            persona.background = extended_attrs.get("background", "")
            persona.personality_traits = extended_attrs.get("personality_traits", [])
            persona.physical_details = extended_attrs.get("physical_details", {})
            persona.preferred_actions = extended_attrs.get("preferred_actions", [])
            persona.speech_style = extended_attrs.get("speech_style", "")
            persona.relationships = extended_attrs.get("relationships", {})
        
        self.personas[name] = persona
        self.save_persona(persona)
        logger.info(f"Created persona: {name}")
        return persona
    
    def save_persona(self, persona: Persona):
        """Save a persona to disk"""
        file_path = self.save_dir / f"{persona.name}.json"
        with open(file_path, 'w') as f:
            json.dump(persona.to_dict(), f, indent=2)
    
    def load_persona(self, name: str) -> Optional[Persona]:
        """Load a specific persona"""
        file_path = self.save_dir / f"{name}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                persona = Persona.from_dict(data)
                self.personas[name] = persona
                return persona
        return None
    
    def load_all_personas(self):
        """Load all saved personas"""
        for file_path in self.save_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    persona = Persona.from_dict(data)
                    self.personas[persona.name] = persona
            except Exception as e:
                logger.error(f"Failed to load persona from {file_path}: {e}")
    
    def switch_persona(self, name: str) -> Optional[Persona]:
        """Switch to a different persona"""
        if name in self.personas:
            self.current_persona = self.personas[name]
            self.current_persona.last_used = datetime.now()
            self.current_persona.play_count += 1
            self.save_persona(self.current_persona)
            logger.info(f"Switched to persona: {name}")
            return self.current_persona
        return None
    
    def get_persona(self, name: str) -> Optional[Persona]:
        """Get a persona without switching"""
        return self.personas.get(name)
    
    def list_personas(self, tags: List[str] = None) -> List[Persona]:
        """List all personas, optionally filtered by tags"""
        personas = list(self.personas.values())
        
        if tags:
            personas = [p for p in personas 
                       if any(tag in p.tags for tag in tags)]
        
        # Sort by favorites first, then last used
        personas.sort(key=lambda p: (not p.favorite, p.last_used), reverse=True)
        return personas
    
    def delete_persona(self, name: str) -> bool:
        """Delete a persona"""
        if name in self.personas:
            file_path = self.save_dir / f"{name}.json"
            if file_path.exists():
                file_path.unlink()
            del self.personas[name]
            if self.current_persona and self.current_persona.name == name:
                self.current_persona = None
            logger.info(f"Deleted persona: {name}")
            return True
        return False
    
    def toggle_favorite(self, name: str) -> bool:
        """Toggle favorite status of a persona"""
        if name in self.personas:
            persona = self.personas[name]
            persona.favorite = not persona.favorite
            self.save_persona(persona)
            return persona.favorite
        return False
    
    def update_persona_relationship(self, persona_name: str, npc_name: str, 
                                   relationship_type: str, notes: str = ""):
        """Update a persona's remembered relationship with an NPC"""
        if persona_name in self.personas:
            persona = self.personas[persona_name]
            persona.relationships[npc_name] = {
                "type": relationship_type,
                "notes": notes,
                "last_interaction": datetime.now().isoformat()
            }
            self.save_persona(persona)
    
    def get_persona_for_world(self, world_tags: List[str]) -> Optional[Persona]:
        """Suggest a persona based on world tags"""
        # Find personas with matching tags
        matching_personas = []
        for persona in self.personas.values():
            match_score = sum(1 for tag in persona.tags if tag in world_tags)
            if match_score > 0:
                matching_personas.append((persona, match_score))
        
        if matching_personas:
            # Return the best match
            matching_personas.sort(key=lambda x: x[1], reverse=True)
            return matching_personas[0][0]
        
        return None

# Global persona manager instance
persona_manager = PersonaManager()