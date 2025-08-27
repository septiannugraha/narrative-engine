"""
Scene and Location models for spatial narrative management
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
from datetime import datetime
import json

from .character import Character
from .position import Position


@dataclass
class Location:
    """Physical location in the narrative world"""
    
    name: str
    description: str
    
    # Physical properties
    size: str = "medium"  # tiny, small, medium, large, huge
    indoor: bool = True
    lighting: str = "normal"  # dark, dim, normal, bright
    temperature: str = "comfortable"  # cold, cool, comfortable, warm, hot
    
    # Environmental details
    props: List[str] = field(default_factory=list)  # Interactive objects
    exits: Dict[str, str] = field(default_factory=dict)  # direction -> location_name
    areas: List[str] = field(default_factory=list)  # Sub-areas within location
    
    # Atmosphere
    mood: str = "neutral"  # neutral, intimate, tense, festive, mysterious
    sounds: List[str] = field(default_factory=list)
    scents: List[str] = field(default_factory=list)
    
    # Accessibility
    is_private: bool = False
    is_locked: bool = False
    required_item: Optional[str] = None  # Item needed to enter
    
    # State flags
    visited: bool = False
    discovered_secrets: List[str] = field(default_factory=list)
    
    def describe(self, time_of_day: str = "day", weather: str = "clear") -> str:
        """Generate atmospheric description"""
        desc = self.description
        
        # Add time-based variations
        if time_of_day == "night" and self.indoor:
            desc += " Shadows dance in the corners."
        elif time_of_day == "dawn":
            desc += " Early morning light filters through."
            
        # Add weather effects if outdoor
        if not self.indoor:
            if weather == "rain":
                desc += " Rain patters steadily."
            elif weather == "storm":
                desc += " Thunder rumbles overhead."
                
        return desc
    
    def get_ambient_text(self) -> str:
        """Get ambient atmosphere text"""
        ambient = []
        
        if self.sounds:
            ambient.append(f"You hear {', '.join(self.sounds)}.")
        if self.scents:
            ambient.append(f"The air smells of {', '.join(self.scents)}.")
            
        return " ".join(ambient)
    
    def can_enter(self, character: Character) -> tuple[bool, str]:
        """Check if character can enter location"""
        if self.is_locked:
            if self.required_item and self.required_item in character.inventory:
                return True, f"{character.name} uses {self.required_item} to enter."
            return False, f"The {self.name} is locked."
            
        if self.is_private:
            # Could check relationships, permissions, etc.
            return True, f"{character.name} enters the private {self.name}."
            
        return True, ""
    
    def to_dict(self) -> dict:
        """Export to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'size': self.size,
            'indoor': self.indoor,
            'lighting': self.lighting,
            'temperature': self.temperature,
            'props': self.props,
            'exits': self.exits,
            'areas': self.areas,
            'mood': self.mood,
            'sounds': self.sounds,
            'scents': self.scents,
            'is_private': self.is_private,
            'is_locked': self.is_locked,
            'visited': self.visited
        }


@dataclass
class Scene:
    """Active scene with characters and state"""
    
    location: Location
    present_characters: Set[Character] = field(default_factory=set)
    
    # Scene properties
    scene_type: str = "exploration"  # exploration, dialogue, action, intimate, transition
    intensity: int = 5  # 1-10 dramatic intensity
    pacing: str = "normal"  # slow, normal, fast, frantic
    
    # Active elements
    active_props: Dict[str, str] = field(default_factory=dict)  # prop -> state
    environmental_effects: List[str] = field(default_factory=list)
    ongoing_actions: List[str] = field(default_factory=list)
    
    # Narrative tracking
    beat_count: int = 0  # Number of narrative beats in scene
    tension_level: int = 0  # Current dramatic tension 0-100
    last_speaker: Optional[str] = None
    
    # Scene goals
    objectives: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    reveals: List[str] = field(default_factory=list)  # Information to reveal
    
    def add_character(self, character: Character):
        """Add character to scene"""
        self.present_characters.add(character)
        character.position.location = self.location.name
        
    def remove_character(self, character: Character):
        """Remove character from scene"""
        self.present_characters.discard(character)
        
    def get_proximity_groups(self, threshold: float = 2.0) -> List[List[Character]]:
        """Group characters by proximity"""
        groups = []
        processed = set()
        
        for char in self.present_characters:
            if char in processed:
                continue
                
            group = [char]
            processed.add(char)
            
            for other in self.present_characters:
                if other in processed:
                    continue
                if char.position.distance_to(other.position) <= threshold:
                    group.append(other)
                    processed.add(other)
                    
            groups.append(group)
            
        return groups
    
    def get_character_by_name(self, name: str) -> Optional[Character]:
        """Find character in scene by name"""
        for char in self.present_characters:
            if char.name.lower() == name.lower():
                return char
        return None
    
    def describe_positions(self) -> str:
        """Describe where everyone is"""
        if not self.present_characters:
            return f"The {self.location.name} is empty."
            
        descriptions = []
        groups = self.get_proximity_groups()
        
        for group in groups:
            if len(group) == 1:
                char = group[0]
                desc = f"{char.name} is {char.position.posture}"
                if char.position.posture == "standing":
                    desc = f"{char.name} stands"
                descriptions.append(desc)
            else:
                names = [c.name for c in group]
                descriptions.append(f"{', '.join(names[:-1])} and {names[-1]} are together")
                
        return f"In the {self.location.name}: " + ". ".join(descriptions) + "."
    
    def escalate_tension(self, amount: int = 10):
        """Increase dramatic tension"""
        self.tension_level = min(100, self.tension_level + amount)
        
        # Adjust intensity based on tension
        if self.tension_level > 80:
            self.intensity = 9
        elif self.tension_level > 60:
            self.intensity = 7
        elif self.tension_level > 40:
            self.intensity = 5
            
    def resolve_tension(self, amount: int = 20):
        """Decrease dramatic tension"""
        self.tension_level = max(0, self.tension_level - amount)
        
    def advance_beat(self):
        """Mark narrative beat progression"""
        self.beat_count += 1
        
        # Natural tension flow
        if self.beat_count % 5 == 0:  # Every 5 beats
            if self.tension_level > 50:
                self.resolve_tension(10)  # Natural ebb
            else:
                self.escalate_tension(5)  # Build up
                
    def should_transition(self) -> bool:
        """Check if scene should transition"""
        # Scene naturally ends after enough beats with low tension
        if self.beat_count > 20 and self.tension_level < 20:
            return True
            
        # Or if all objectives are complete
        if self.objectives and all(obj in self.reveals for obj in self.objectives):
            return True
            
        return False
    
    def get_mood_modifiers(self) -> Dict[str, float]:
        """Get mood-based behavior modifiers"""
        modifiers = {
            'intimacy': 1.0,
            'aggression': 1.0,
            'playfulness': 1.0,
            'formality': 1.0
        }
        
        # Adjust based on location mood
        if self.location.mood == "intimate":
            modifiers['intimacy'] = 1.5
            modifiers['formality'] = 0.5
        elif self.location.mood == "tense":
            modifiers['aggression'] = 1.3
            modifiers['playfulness'] = 0.7
        elif self.location.mood == "festive":
            modifiers['playfulness'] = 1.5
            modifiers['formality'] = 0.3
            
        # Adjust based on scene intensity
        intensity_factor = self.intensity / 10.0
        modifiers['aggression'] *= intensity_factor
        modifiers['intimacy'] *= (2.0 - intensity_factor)  # Inverse
        
        return modifiers
    
    def to_dict(self) -> dict:
        """Export scene state"""
        return {
            'location': self.location.name,
            'characters': [c.name for c in self.present_characters],
            'scene_type': self.scene_type,
            'intensity': self.intensity,
            'pacing': self.pacing,
            'tension_level': self.tension_level,
            'beat_count': self.beat_count,
            'active_props': self.active_props,
            'environmental_effects': self.environmental_effects,
            'ongoing_actions': self.ongoing_actions,
            'objectives': self.objectives,
            'position_summary': self.describe_positions()
        }


@dataclass 
class SceneTemplate:
    """Template for common scene setups"""
    
    name: str
    scene_type: str
    required_props: List[str] = field(default_factory=list)
    
    # Initial setup
    starting_positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    starting_clothing: Dict[str, str] = field(default_factory=dict)
    starting_mood: str = "neutral"
    
    # Scene flow
    opening_description: str = ""
    key_beats: List[str] = field(default_factory=list)
    potential_conflicts: List[str] = field(default_factory=list)
    
    def apply_to_scene(self, scene: Scene, characters: Dict[str, Character]):
        """Apply template to a scene"""
        scene.scene_type = self.scene_type
        scene.location.mood = self.starting_mood
        
        # Position characters
        for char_name, pos_data in self.starting_positions.items():
            if char_name in characters:
                char = characters[char_name]
                char.position = Position(**pos_data)
                scene.add_character(char)
                
        # Set clothing if specified
        for char_name, clothing in self.starting_clothing.items():
            if char_name in characters:
                characters[char_name].clothing_description = clothing
                
        # Add scene objectives from beats
        scene.objectives = self.key_beats.copy()
        scene.conflicts = self.potential_conflicts.copy()


# Common scene templates
SCENE_TEMPLATES = {
    "hot_springs_encounter": SceneTemplate(
        name="Hot Springs Encounter",
        scene_type="intimate",
        required_props=["pools", "rocks", "towels"],
        starting_positions={
            "player": {"x": 0, "y": 0, "z": 0, "posture": "sitting"},
            "npc": {"x": 2, "y": 0, "z": 0, "posture": "sitting"}
        },
        starting_clothing={
            "player": "towel",
            "npc": "towel"
        },
        starting_mood="intimate",
        opening_description="Steam rises from the natural hot springs, creating an intimate atmosphere.",
        key_beats=[
            "Initial surprise/acknowledgment",
            "Tension builds from proximity",
            "Conversation or action choice",
            "Resolution or escalation"
        ],
        potential_conflicts=["embarrassment", "attraction", "privacy"]
    ),
    
    "tavern_meeting": SceneTemplate(
        name="Tavern Meeting",
        scene_type="dialogue",
        required_props=["bar", "tables", "fireplace"],
        starting_positions={
            "player": {"x": 5, "y": 5, "z": 0, "posture": "sitting"},
            "npc": {"x": 5, "y": 6, "z": 0, "posture": "sitting"}
        },
        starting_mood="warm",
        opening_description="The tavern buzzes with conversation and the clink of mugs.",
        key_beats=[
            "Introductions or greetings",
            "Information exchange",
            "Quest hook or relationship development"
        ],
        potential_conflicts=["misunderstanding", "competing goals", "hidden agenda"]
    ),
    
    "combat_arena": SceneTemplate(
        name="Combat Arena", 
        scene_type="action",
        required_props=["weapons", "barriers", "audience"],
        starting_positions={
            "player": {"x": -10, "y": 0, "z": 0, "posture": "standing", "facing": "east"},
            "opponent": {"x": 10, "y": 0, "z": 0, "posture": "standing", "facing": "west"}
        },
        starting_mood="tense",
        opening_description="The crowd roars as combatants face each other across the arena.",
        key_beats=[
            "Initial clash",
            "Momentum shift",
            "Decisive moment",
            "Victory or defeat"
        ],
        potential_conflicts=["survival", "honor", "rivalry"]
    )
}