"""
World state management - the single source of truth
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import copy

from .character import Character
from .scene import Scene, Location
from .position import Position


@dataclass
class Checkpoint:
    """Saved world state snapshot"""
    name: str
    timestamp: datetime
    world_data: dict
    description: str = ""
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'timestamp': self.timestamp.isoformat(),
            'description': self.description,
            'data_size': len(str(self.world_data))
        }


class WorldState:
    """
    The single source of truth for the entire narrative world.
    Tracks all characters, locations, scenes, and global state.
    """
    
    def __init__(self, world_id: str = "default"):
        self.world_id = world_id
        self.created_at = datetime.now()
        
        # Core collections
        self.locations: Dict[str, Location] = {}
        self.characters: Dict[str, Character] = {}
        
        # Active state
        self.active_scene: Optional[Scene] = None
        self.current_time = datetime.now()
        self.day_count = 0
        self.weather = "clear"
        self.season = "spring"
        
        # Global flags and variables
        self.global_flags: Dict[str, Any] = {}
        self.quest_states: Dict[str, str] = {}
        
        # History and checkpoints
        self.narrative_log: List[str] = []
        self.event_history: List[dict] = []
        self.checkpoints: Dict[str, Checkpoint] = {}
        
        # Statistics
        self.stats = {
            'total_interactions': 0,
            'scenes_visited': [],
            'relationships_formed': 0,
            'checkpoints_created': 0
        }
    
    # ========================================================================
    # LOCATION MANAGEMENT
    # ========================================================================
    
    def add_location(self, location: Location):
        """Add a location to the world"""
        self.locations[location.name] = location
        self.log_event(f"Location '{location.name}' added to world")
    
    def get_location(self, name: str) -> Optional[Location]:
        """Get location by name"""
        return self.locations.get(name)
    
    def list_locations(self) -> List[str]:
        """List all location names"""
        return list(self.locations.keys())
    
    # ========================================================================
    # CHARACTER MANAGEMENT
    # ========================================================================
    
    def add_character(self, character: Character):
        """Add a character to the world"""
        self.characters[character.name] = character
        self.log_event(f"Character '{character.name}' joined the world")
        
    def get_character(self, name: str) -> Optional[Character]:
        """Get character by name"""
        return self.characters.get(name)
    
    def list_characters(self) -> List[str]:
        """List all character names"""
        return list(self.characters.keys())
    
    def get_characters_at_location(self, location_name: str) -> List[Character]:
        """Get all characters at a specific location"""
        return [
            char for char in self.characters.values()
            if char.position.location == location_name
        ]
    
    def move_character(self, char_name: str, 
                      to_location: Optional[str] = None,
                      to_position: Optional[Position] = None) -> str:
        """Move a character to a new location or position"""
        if char_name not in self.characters:
            return f"Character {char_name} not found"
        
        character = self.characters[char_name]
        old_location = character.position.location
        
        # Handle location change
        if to_location and to_location != old_location:
            if to_location not in self.locations:
                return f"Location {to_location} not found"
            
            # Remove from current scene if active
            if self.active_scene and character in self.active_scene.present_characters:
                self.active_scene.remove_character(character)
            
            # Update position
            character.position.location = to_location
            
            # Add to new scene if it's the active one
            if self.active_scene and self.active_scene.location.name == to_location:
                self.active_scene.add_character(character)
            
            message = f"{char_name} moves from {old_location} to {to_location}"
            
        # Handle position change within same location
        elif to_position:
            old_pos = str(character.position)
            character.position = to_position
            message = f"{char_name} moves to {to_position}"
        else:
            return "No movement specified"
        
        self.log_event(message)
        return message
    
    # ========================================================================
    # SCENE MANAGEMENT
    # ========================================================================
    
    def set_scene(self, location_name: str) -> Scene:
        """Set the active scene to a location"""
        if location_name not in self.locations:
            raise ValueError(f"Location {location_name} not found")
        
        location = self.locations[location_name]
        self.active_scene = Scene(location)
        
        # Add all characters at this location to the scene
        for char in self.get_characters_at_location(location_name):
            self.active_scene.add_character(char)
        
        # Track visited scenes
        if location_name not in self.stats['scenes_visited']:
            self.stats['scenes_visited'].append(location_name)
        
        self.log_event(f"Scene set to {location_name}")
        return self.active_scene
    
    def get_active_scene(self) -> Optional[Scene]:
        """Get the currently active scene"""
        return self.active_scene
    
    # ========================================================================
    # TIME AND ENVIRONMENT
    # ========================================================================
    
    def advance_time(self, hours: int = 1):
        """Advance world time"""
        from datetime import timedelta
        self.current_time += timedelta(hours=hours)
        
        # Check for day change
        if self.current_time.hour < hours:
            self.day_count += 1
            self.log_event(f"Day {self.day_count} begins")
        
        # Update time of day
        hour = self.current_time.hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 20:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        
        self.global_flags['time_of_day'] = time_of_day
    
    def set_weather(self, weather: str):
        """Set current weather"""
        self.weather = weather
        self.log_event(f"Weather changes to {weather}")
    
    def set_season(self, season: str):
        """Set current season"""
        self.season = season
        self.log_event(f"Season changes to {season}")
    
    # ========================================================================
    # CHECKPOINTS
    # ========================================================================
    
    def create_checkpoint(self, name: str, description: str = "") -> str:
        """Save current world state as checkpoint"""
        checkpoint_data = {
            'characters': {
                name: char.to_dict() 
                for name, char in self.characters.items()
            },
            'locations': {
                name: loc.to_dict()
                for name, loc in self.locations.items()
            },
            'global_flags': copy.deepcopy(self.global_flags),
            'quest_states': copy.deepcopy(self.quest_states),
            'current_time': self.current_time.isoformat(),
            'day_count': self.day_count,
            'weather': self.weather,
            'season': self.season,
            'stats': copy.deepcopy(self.stats)
        }
        
        checkpoint = Checkpoint(
            name=name,
            timestamp=datetime.now(),
            world_data=checkpoint_data,
            description=description
        )
        
        self.checkpoints[name] = checkpoint
        self.stats['checkpoints_created'] += 1
        
        message = f"Checkpoint '{name}' created"
        self.log_event(message)
        return message
    
    def restore_checkpoint(self, name: str) -> str:
        """Restore world state from checkpoint"""
        if name not in self.checkpoints:
            return f"Checkpoint '{name}' not found"
        
        checkpoint = self.checkpoints[name]
        data = checkpoint.world_data
        
        # Restore characters
        for char_name, char_data in data['characters'].items():
            if char_name in self.characters:
                # Update existing character
                char = Character.from_dict(char_data)
                self.characters[char_name] = char
        
        # Restore global state
        self.global_flags = copy.deepcopy(data['global_flags'])
        self.quest_states = copy.deepcopy(data['quest_states'])
        self.current_time = datetime.fromisoformat(data['current_time'])
        self.day_count = data['day_count']
        self.weather = data['weather']
        self.season = data['season']
        self.stats = copy.deepcopy(data['stats'])
        
        message = f"World restored to checkpoint '{name}'"
        self.log_event(message)
        return message
    
    def list_checkpoints(self) -> List[dict]:
        """List all available checkpoints"""
        return [cp.to_dict() for cp in self.checkpoints.values()]
    
    # ========================================================================
    # EVENT LOGGING
    # ========================================================================
    
    def log_event(self, event: str, event_type: str = "system"):
        """Log an event to history"""
        self.narrative_log.append(event)
        
        self.event_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'event': event
        })
        
        # Limit log size
        if len(self.narrative_log) > 1000:
            self.narrative_log = self.narrative_log[-1000:]
        if len(self.event_history) > 500:
            self.event_history = self.event_history[-500:]
    
    def get_recent_events(self, count: int = 10) -> List[str]:
        """Get recent narrative events"""
        return self.narrative_log[-count:]
    
    # ========================================================================
    # RELATIONSHIP TRACKING
    # ========================================================================
    
    def get_all_relationships(self) -> Dict[str, Dict[str, str]]:
        """Get all character relationships"""
        relationships = {}
        
        for char_name, character in self.characters.items():
            relationships[char_name] = {}
            for target, rel in character.relationships.items():
                relationships[char_name][target] = rel.describe()
        
        return relationships
    
    def get_relationship_graph(self) -> List[dict]:
        """Get relationship data for visualization"""
        edges = []
        
        for char_name, character in self.characters.items():
            for target, rel in character.relationships.items():
                edges.append({
                    'source': char_name,
                    'target': target,
                    'type': rel.relationship_type.value,
                    'strength': (rel.affection + rel.trust) / 2,
                    'romantic': rel.is_romantic
                })
        
        return edges
    
    # ========================================================================
    # EXPORT/IMPORT
    # ========================================================================
    
    def to_dict(self) -> dict:
        """Export world state as dictionary"""
        return {
            'world_id': self.world_id,
            'created_at': self.created_at.isoformat(),
            'current_time': self.current_time.isoformat(),
            'day_count': self.day_count,
            'weather': self.weather,
            'season': self.season,
            'locations': list(self.locations.keys()),
            'characters': {
                name: char.to_dict() 
                for name, char in self.characters.items()
            },
            'active_scene': self.active_scene.to_dict() if self.active_scene else None,
            'global_flags': self.global_flags,
            'quest_states': self.quest_states,
            'recent_events': self.get_recent_events(20),
            'stats': self.stats,
            'checkpoints': list(self.checkpoints.keys()),
            'relationships': self.get_all_relationships()
        }
    
    def to_json(self) -> str:
        """Export world state as JSON"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WorldState':
        """Import world state from JSON"""
        data = json.loads(json_str)
        
        world = cls(world_id=data.get('world_id', 'imported'))
        
        # Restore times
        world.current_time = datetime.fromisoformat(data['current_time'])
        world.day_count = data['day_count']
        world.weather = data['weather']
        world.season = data['season']
        
        # Restore flags and states
        world.global_flags = data.get('global_flags', {})
        world.quest_states = data.get('quest_states', {})
        world.stats = data.get('stats', world.stats)
        
        # Restore characters
        for char_name, char_data in data.get('characters', {}).items():
            character = Character.from_dict(char_data)
            world.add_character(character)
        
        return world