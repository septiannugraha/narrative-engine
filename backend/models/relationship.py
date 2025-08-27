"""
Relationship tracking between characters
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum


class RelationshipType(Enum):
    """Types of relationships"""
    STRANGER = "stranger"
    ACQUAINTANCE = "acquaintance"
    FRIEND = "friend"
    CLOSE_FRIEND = "close friend"
    ROMANTIC_INTEREST = "romantic interest"
    LOVER = "lover"
    PARTNER = "partner"
    RIVAL = "rival"
    ENEMY = "enemy"
    FAMILY = "family"
    ALLY = "ally"
    
    def intimacy_level(self) -> int:
        """Return intimacy level 0-100"""
        levels = {
            self.STRANGER: 0,
            self.ENEMY: 0,
            self.RIVAL: 10,
            self.ACQUAINTANCE: 20,
            self.ALLY: 30,
            self.FRIEND: 40,
            self.CLOSE_FRIEND: 60,
            self.ROMANTIC_INTEREST: 70,
            self.FAMILY: 80,
            self.LOVER: 90,
            self.PARTNER: 100
        }
        return levels.get(self, 0)


@dataclass
class Interaction:
    """Single interaction record"""
    timestamp: datetime
    interaction_type: str
    description: str
    location: str = ""
    emotional_impact: int = 0  # -100 to 100


@dataclass
class Relationship:
    """Track relationship between two characters"""
    
    target_name: str
    relationship_type: RelationshipType = RelationshipType.STRANGER
    
    # Relationship metrics (0-100)
    affection: int = 50
    trust: int = 50
    attraction: int = 50
    respect: int = 50
    
    # Romance tracking
    romantic_tension: int = 0
    intimacy_level: int = 0
    jealousy: int = 0
    
    # History
    first_meeting: Optional[datetime] = None
    interactions: List[Interaction] = field(default_factory=list)
    shared_memories: List[str] = field(default_factory=list)
    
    # Flags
    is_romantic: bool = False
    is_sexual: bool = False
    is_exclusive: bool = False
    has_confessed: bool = False
    has_kissed: bool = False
    has_been_intimate: bool = False
    
    def modify(self, affection: int = 0, trust: int = 0, 
              attraction: int = 0, respect: int = 0,
              romantic_tension: int = 0, jealousy: int = 0):
        """Modify relationship values with bounds"""
        self.affection = max(0, min(100, self.affection + affection))
        self.trust = max(0, min(100, self.trust + trust))
        self.attraction = max(0, min(100, self.attraction + attraction))
        self.respect = max(0, min(100, self.respect + respect))
        self.romantic_tension = max(0, min(100, self.romantic_tension + romantic_tension))
        self.jealousy = max(0, min(100, self.jealousy + jealousy))
        
        # Update relationship type based on metrics
        self._update_relationship_type()
    
    def _update_relationship_type(self):
        """Update relationship type based on current metrics"""
        avg_positive = (self.affection + self.trust + self.respect) / 3
        
        if avg_positive < 20:
            if self.respect < 30:
                self.relationship_type = RelationshipType.ENEMY
            else:
                self.relationship_type = RelationshipType.STRANGER
        elif avg_positive < 40:
            if self.respect < 40:
                self.relationship_type = RelationshipType.RIVAL
            else:
                self.relationship_type = RelationshipType.ACQUAINTANCE
        elif self.is_romantic or self.attraction > 70:
            if self.intimacy_level > 80:
                self.relationship_type = RelationshipType.PARTNER
            elif self.intimacy_level > 50:
                self.relationship_type = RelationshipType.LOVER
            else:
                self.relationship_type = RelationshipType.ROMANTIC_INTEREST
        elif avg_positive < 60:
            self.relationship_type = RelationshipType.FRIEND
        else:
            self.relationship_type = RelationshipType.CLOSE_FRIEND
    
    def add_interaction(self, interaction_type: str, description: str = "",
                       location: str = "", emotional_impact: int = 0):
        """Add interaction to history"""
        if not self.first_meeting:
            self.first_meeting = datetime.now()
            
        interaction = Interaction(
            timestamp=datetime.now(),
            interaction_type=interaction_type,
            description=description,
            location=location,
            emotional_impact=emotional_impact
        )
        self.interactions.append(interaction)
        
        # Update flags based on interaction type
        if interaction_type == "confession":
            self.has_confessed = True
            self.is_romantic = True
        elif interaction_type == "kiss":
            self.has_kissed = True
            self.is_romantic = True
            self.intimacy_level = max(50, self.intimacy_level)
        elif interaction_type in ["intimate", "sexual"]:
            self.has_been_intimate = True
            self.is_sexual = True
            self.intimacy_level = max(80, self.intimacy_level)
        
        # Limit history size
        if len(self.interactions) > 50:
            self.interactions = self.interactions[-50:]
    
    def get_impact(self, interaction_type: str) -> int:
        """Get emotional impact of interaction type"""
        impacts = {
            "compliment": 10,
            "touch": 15,
            "hug": 20,
            "kiss": 30,
            "intimate": 40,
            "confession": 35,
            "argue": -20,
            "insult": -30,
            "betray": -50
        }
        return impacts.get(interaction_type, 0)
    
    def describe(self) -> str:
        """Get relationship description"""
        desc = f"{self.relationship_type.value}"
        
        if self.is_romantic:
            desc += " (romantic)"
        if self.is_exclusive:
            desc += " (exclusive)"
        
        # Add emotional descriptors
        emotions = []
        if self.affection > 80:
            emotions.append("deeply affectionate")
        elif self.affection > 60:
            emotions.append("affectionate")
            
        if self.attraction > 80:
            emotions.append("intensely attracted")
        elif self.attraction > 60:
            emotions.append("attracted")
            
        if self.trust > 80:
            emotions.append("completely trusting")
        elif self.trust < 30:
            emotions.append("distrustful")
            
        if self.jealousy > 50:
            emotions.append("jealous")
            
        if emotions:
            desc += f" - {', '.join(emotions)}"
            
        return desc
    
    def to_dict(self) -> dict:
        """Export to dictionary"""
        return {
            'target_name': self.target_name,
            'relationship_type': self.relationship_type.value,
            'affection': self.affection,
            'trust': self.trust,
            'attraction': self.attraction,
            'respect': self.respect,
            'romantic_tension': self.romantic_tension,
            'intimacy_level': self.intimacy_level,
            'jealousy': self.jealousy,
            'is_romantic': self.is_romantic,
            'is_sexual': self.is_sexual,
            'is_exclusive': self.is_exclusive,
            'has_confessed': self.has_confessed,
            'has_kissed': self.has_kissed,
            'has_been_intimate': self.has_been_intimate,
            'interactions_count': len(self.interactions),
            'first_meeting': self.first_meeting.isoformat() if self.first_meeting else None
        }