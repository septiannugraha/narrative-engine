"""
Character model with full state tracking
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import json

from .position import Position
from .relationship import Relationship
from .anatomy import AnatomicalModel, PhysicalMeasurements, BodySide


class ClothingState(Enum):
    """Tracks character clothing state"""
    FULL = "fully dressed"
    PARTIAL = "partially dressed" 
    UNDERWEAR = "underwear only"
    TOWEL = "towel only"
    NAKED = "naked"
    ARMOR = "armored"
    FORMAL = "formal attire"
    CASUAL = "casual wear"
    SLEEP = "sleepwear"
    
    def coverage_level(self) -> int:
        """Return coverage level 0-100"""
        coverage = {
            self.NAKED: 0,
            self.TOWEL: 20,
            self.UNDERWEAR: 30,
            self.SLEEP: 40,
            self.PARTIAL: 50,
            self.CASUAL: 80,
            self.FULL: 85,
            self.FORMAL: 90,
            self.ARMOR: 100
        }
        return coverage.get(self, 50)


class EmotionalState(Enum):
    """Character emotional states"""
    CALM = "calm"
    HAPPY = "happy"
    EXCITED = "excited"
    NERVOUS = "nervous"
    EMBARRASSED = "embarrassed"
    AROUSED = "aroused"
    ANGRY = "angry"
    SAD = "sad"
    FEARFUL = "fearful"
    CONFIDENT = "confident"
    PLAYFUL = "playful"
    LOVING = "loving"
    
    def intensity_modifier(self) -> float:
        """Return intensity modifier for actions"""
        intensity = {
            self.CALM: 0.8,
            self.NERVOUS: 0.9,
            self.HAPPY: 1.1,
            self.EXCITED: 1.3,
            self.AROUSED: 1.4,
            self.ANGRY: 1.5,
            self.LOVING: 1.2,
            self.PLAYFUL: 1.2
        }
        return intensity.get(self, 1.0)


class PhysicalState(Enum):
    """Physical condition tracking"""
    HEALTHY = "healthy"
    TIRED = "tired"
    EXHAUSTED = "exhausted"
    INJURED = "injured"
    AROUSED = "physically aroused"
    RELAXED = "relaxed"
    TENSE = "tense"


@dataclass
class Memory:
    """Single memory entry"""
    timestamp: datetime
    event: str
    emotional_impact: int  # -100 to 100
    participants: List[str]
    location: str
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'event': self.event,
            'emotional_impact': self.emotional_impact,
            'participants': self.participants,
            'location': self.location,
            'tags': self.tags
        }


@dataclass
class Character:
    """Complete character with state, memory, and agency"""
    
    # Identity
    name: str
    description: str
    age: Optional[int] = None
    personality_traits: List[str] = field(default_factory=list)
    
    # Physical State
    position: Position = field(default_factory=Position)
    clothing: ClothingState = ClothingState.FULL
    clothing_description: str = ""
    physical_state: PhysicalState = PhysicalState.HEALTHY
    
    # Anatomical Details (NEW)
    anatomy: Optional[AnatomicalModel] = None
    measurements: Optional[PhysicalMeasurements] = None
    
    # Mental State
    emotional_state: EmotionalState = EmotionalState.CALM
    arousal_level: int = 0  # 0-100
    stress_level: int = 0  # 0-100
    
    # Inventory & Status
    inventory: List[str] = field(default_factory=list)
    status_effects: List[str] = field(default_factory=list)
    
    # Memory & Relationships
    memories: List[Memory] = field(default_factory=list)
    relationships: Dict[str, Relationship] = field(default_factory=dict)
    
    # Stats (Optional RPG elements)
    stats: Dict[str, int] = field(default_factory=lambda: {
        'health': 100,
        'stamina': 100,
        'mana': 100,
        'strength': 10,
        'dexterity': 10,
        'intelligence': 10,
        'charisma': 10
    })
    
    # Internal state for AI
    current_goal: Optional[str] = None
    hidden_thoughts: List[str] = field(default_factory=list)
    
    def __hash__(self):
        """Make character hashable by name"""
        return hash(self.name)
    
    def __eq__(self, other):
        """Characters equal if same name"""
        if isinstance(other, Character):
            return self.name == other.name
        return False
    
    def move_to(self, position: Position, description: str = "") -> str:
        """Move character to new position"""
        old_pos = str(self.position)
        self.position = position
        
        if description:
            return description
        return f"{self.name} moves from {old_pos} to {position}"
    
    def change_clothing(self, new_state: ClothingState, description: str = "") -> Optional[str]:
        """Change clothing state"""
        if new_state == self.clothing:
            return None
            
        old_state = self.clothing
        self.clothing = new_state
        self.clothing_description = description
        
        # Affect emotional state based on exposure
        if new_state.coverage_level() < 30:
            if self.emotional_state == EmotionalState.CALM:
                self.emotional_state = EmotionalState.EMBARRASSED
                
        return f"{self.name}'s clothing changes from {old_state.value} to {new_state.value}"
    
    def modify_arousal(self, change: int) -> str:
        """Modify arousal level with bounds"""
        old_level = self.arousal_level
        self.arousal_level = max(0, min(100, self.arousal_level + change))
        
        # Update emotional state based on arousal
        if self.arousal_level > 70:
            self.emotional_state = EmotionalState.AROUSED
        elif self.arousal_level > 40 and self.emotional_state == EmotionalState.CALM:
            self.emotional_state = EmotionalState.EXCITED
            
        if self.arousal_level != old_level:
            return f"{self.name}'s arousal changes from {old_level} to {self.arousal_level}"
        return ""
    
    def add_memory(self, event: str, emotional_impact: int, 
                   participants: List[str], location: str, tags: List[str] = None):
        """Add memory to character's history"""
        memory = Memory(
            timestamp=datetime.now(),
            event=event,
            emotional_impact=emotional_impact,
            participants=participants,
            location=location,
            tags=tags or []
        )
        self.memories.append(memory)
        
        # Limit memory size
        if len(self.memories) > 100:
            self.memories = self.memories[-100:]
    
    def get_relationship(self, other_name: str) -> Relationship:
        """Get or create relationship with another character"""
        if other_name not in self.relationships:
            self.relationships[other_name] = Relationship(other_name)
        return self.relationships[other_name]
    
    def interact_with(self, other: 'Character', interaction_type: str, 
                     description: str = "") -> str:
        """Record interaction with another character"""
        relationship = self.get_relationship(other.name)
        
        # Modify relationship based on interaction
        if interaction_type == "compliment":
            relationship.modify(affection=5, attraction=3)
            other.modify_arousal(5)
        elif interaction_type == "touch":
            relationship.modify(affection=3, attraction=5, trust=2)
            other.modify_arousal(10)
            self.modify_arousal(8)
        elif interaction_type == "kiss":
            relationship.modify(affection=10, attraction=10, trust=5)
            other.modify_arousal(20)
            self.modify_arousal(20)
        elif interaction_type == "argue":
            relationship.modify(affection=-5, trust=-3)
            
        # Record in history
        relationship.add_interaction(interaction_type, description)
        
        # Create memory
        self.add_memory(
            event=f"{interaction_type} with {other.name}",
            emotional_impact=relationship.get_impact(interaction_type),
            participants=[self.name, other.name],
            location=str(self.position.location),
            tags=[interaction_type, "social"]
        )
        
        return description or f"{self.name} {interaction_type}s {other.name}"
    
    def to_dict(self) -> dict:
        """Export character state as dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'position': self.position.to_dict(),
            'clothing': self.clothing.value,
            'clothing_description': self.clothing_description,
            'emotional_state': self.emotional_state.value,
            'physical_state': self.physical_state.value,
            'arousal_level': self.arousal_level,
            'stress_level': self.stress_level,
            'inventory': self.inventory,
            'status_effects': self.status_effects,
            'stats': self.stats,
            'relationships': {
                name: rel.to_dict() 
                for name, rel in self.relationships.items()
            },
            'memories': [m.to_dict() for m in self.memories[-10:]],  # Last 10
            'current_goal': self.current_goal
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Character':
        """Create character from dictionary"""
        char = cls(
            name=data['name'],
            description=data['description']
        )
        
        # Restore position
        char.position = Position.from_dict(data.get('position', {}))
        
        # Restore states
        char.clothing = ClothingState(data.get('clothing', 'fully dressed'))
        char.clothing_description = data.get('clothing_description', '')
        char.emotional_state = EmotionalState(data.get('emotional_state', 'calm'))
        char.physical_state = PhysicalState(data.get('physical_state', 'healthy'))
        
        # Restore levels
        char.arousal_level = data.get('arousal_level', 0)
        char.stress_level = data.get('stress_level', 0)
        
        # Restore inventory and stats
        char.inventory = data.get('inventory', [])
        char.status_effects = data.get('status_effects', [])
        char.stats = data.get('stats', char.stats)
        
        return char