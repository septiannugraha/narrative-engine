"""
Position tracking in 3D space with location context
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class Position:
    """Track character position in 3D space within a location"""
    
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    location: str = "unknown"
    
    # Optional details
    facing: str = "north"  # Direction character is facing
    posture: str = "standing"  # standing, sitting, lying, kneeling
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate distance to another position"""
        if self.location != other.location:
            return float('inf')  # Different locations = infinite distance
            
        return math.sqrt(
            (self.x - other.x) ** 2 + 
            (self.y - other.y) ** 2 + 
            (self.z - other.z) ** 2
        )
    
    def is_adjacent_to(self, other: 'Position', threshold: float = 1.0) -> bool:
        """Check if positions are adjacent"""
        return self.distance_to(other) <= threshold
    
    def move_toward(self, target: 'Position', distance: float) -> 'Position':
        """Create new position moved toward target"""
        if self.location != target.location:
            return self  # Can't move toward different location
            
        total_distance = self.distance_to(target)
        if total_distance == 0:
            return Position(self.x, self.y, self.z, self.location)
            
        # Calculate unit vector
        ratio = min(distance / total_distance, 1.0)
        new_x = self.x + (target.x - self.x) * ratio
        new_y = self.y + (target.y - self.y) * ratio
        new_z = self.z + (target.z - self.z) * ratio
        
        return Position(new_x, new_y, new_z, self.location, self.facing, self.posture)
    
    def get_grid_position(self, grid_size: float = 1.0) -> Tuple[int, int, int]:
        """Get position as grid coordinates"""
        return (
            int(self.x / grid_size),
            int(self.y / grid_size),
            int(self.z / grid_size)
        )
    
    def describe_relative_to(self, other: 'Position') -> str:
        """Describe position relative to another"""
        if self.location != other.location:
            return f"in {self.location} (while other is in {other.location})"
            
        distance = self.distance_to(other)
        
        if distance < 0.5:
            return "right next to"
        elif distance < 1.0:
            return "very close to"
        elif distance < 2.0:
            return "close to"
        elif distance < 5.0:
            return "near"
        elif distance < 10.0:
            return "across from"
        else:
            return "far from"
    
    def __str__(self) -> str:
        """String representation"""
        pos_str = f"{self.location}: ({self.x:.1f}, {self.y:.1f}, {self.z:.1f})"
        if self.posture != "standing":
            pos_str += f" [{self.posture}]"
        return pos_str
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'location': self.location,
            'facing': self.facing,
            'posture': self.posture
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Position':
        """Create from dictionary"""
        return cls(
            x=data.get('x', 0.0),
            y=data.get('y', 0.0),
            z=data.get('z', 0.0),
            location=data.get('location', 'unknown'),
            facing=data.get('facing', 'north'),
            posture=data.get('posture', 'standing')
        )