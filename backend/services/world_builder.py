"""
World Builder - Various methods to initialize worlds
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..models import (
    WorldState, Character, Location, Position,
    ClothingState, EmotionalState, Relationship, RelationshipType
)
from .structured_director import StructuredDirector
from .director import DirectorConfig


class WorldTemplate:
    """Load and apply world templates"""
    
    def __init__(self, template_path: str = None, template_data: Dict = None):
        if template_path:
            with open(template_path, 'r') as f:
                self.data = json.load(f)
        elif template_data:
            self.data = template_data
        else:
            raise ValueError("Must provide either template_path or template_data")
    
    def to_world_state(self, world_name: str = None) -> WorldState:
        """Convert template to WorldState"""
        world = WorldState(world_name or self.data.get("world_name", "template_world"))
        
        # Set world properties
        world_state = self.data.get("world_state", {})
        world.time_of_day = world_state.get("time_of_day", "day")
        world.weather = world_state.get("weather", "clear")
        world.season = world_state.get("season", "spring")
        
        # Create locations
        for loc_data in self.data.get("locations", []):
            location = Location(
                name=loc_data["name"],
                description=loc_data.get("description", ""),
                props=loc_data.get("props", []),
                mood=loc_data.get("mood", "neutral"),
                lighting=loc_data.get("lighting", "normal"),
                temperature=loc_data.get("temperature", "comfortable"),
                sounds=loc_data.get("ambient_sounds", []),
                scents=loc_data.get("scents", [])
            )
            world.add_location(location)
        
        # Create NPCs
        for npc_data in self.data.get("npcs", []):
            npc = Character(
                name=npc_data["name"],
                description=npc_data.get("description", ""),
                age=npc_data.get("age", 30),
                personality_traits=npc_data.get("personality", [])
            )
            
            # Set position
            pos_desc = npc_data.get("initial_position", "center")
            x, y = self._parse_position(pos_desc)
            npc.position = Position(
                x=x, y=y, z=0,
                location=self.data["locations"][0]["name"] if self.data.get("locations") else "Unknown"
            )
            
            # Set clothing
            clothing = npc_data.get("clothing", "dressed")
            npc.change_clothing(ClothingState.FULL, clothing)
            
            world.add_character(npc)
        
        return world
    
    def _parse_position(self, position_desc: str) -> tuple:
        """Parse position description to coordinates"""
        pos_lower = position_desc.lower()
        
        if "behind bar" in pos_lower or "bartender" in pos_lower:
            return (0, 5)
        elif "corner" in pos_lower:
            return (-7, -7)
        elif "door" in pos_lower or "entrance" in pos_lower:
            return (0, -8)
        elif "center" in pos_lower:
            return (0, 0)
        else:
            # Random position
            import random
            return (random.uniform(-5, 5), random.uniform(-5, 5))


class WorldBuilder:
    """Advanced world building utilities"""
    
    def __init__(self):
        self.templates_dir = Path("worlds/templates")
        self.templates_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_from_prompt(self, prompt: str, player_name: str = "Player") -> WorldState:
        """Create world entirely from prompt (current method)"""
        world = WorldState("prompt_world")
        director = StructuredDirector(world)
        
        result = await director.establish_scene_from_prompt(
            initial_prompt=prompt,
            player_name=player_name
        )
        
        if result.get("success"):
            return world
        else:
            raise ValueError(f"Failed to create world: {result.get('error')}")
    
    async def create_hybrid(self, 
                          template_name: str, 
                          enhancement_prompt: str = None,
                          player_name: str = "Player") -> WorldState:
        """Start with template, enhance with LLM"""
        
        # Load base template
        template_path = self.templates_dir / f"{template_name}.json"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_name}")
        
        template = WorldTemplate(str(template_path))
        world = template.to_world_state()
        
        # Enhance with LLM if requested
        if enhancement_prompt:
            director = StructuredDirector(world)
            
            # Add the enhancements
            enhanced_prompt = f"""
Current world: {template_name}

Enhancements to add:
{enhancement_prompt}

Keep all existing characters and locations, but add the requested changes.
"""
            await director.enhance_scene(enhanced_prompt, player_name)
        
        return world
    
    async def create_progressive(self, director: StructuredDirector) -> WorldState:
        """Build world step by step with user input"""
        
        steps = []
        
        # Step 1: Location
        location_prompt = input("Describe the location: ")
        steps.append(f"Location: {location_prompt}")
        
        # Step 2: NPCs
        npc_prompt = input("Describe the NPCs present: ")
        steps.append(f"NPCs: {npc_prompt}")
        
        # Step 3: Atmosphere
        atmosphere_prompt = input("Describe the atmosphere/situation: ")
        steps.append(f"Atmosphere: {atmosphere_prompt}")
        
        # Step 4: Player
        player_prompt = input("Describe your character: ")
        player_name = input("Your character's name: ")
        steps.append(f"Player: {player_prompt}")
        
        # Combine into full prompt
        full_prompt = "\n".join(steps)
        
        result = await director.establish_scene_from_prompt(
            initial_prompt=full_prompt,
            player_name=player_name
        )
        
        return director.world if result.get("success") else None
    
    def save_world(self, world: WorldState, filename: str):
        """Save world state to file"""
        save_path = Path("saved_worlds") / f"{filename}.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(world.to_dict(), f, indent=2)
        
        print(f"World saved to {save_path}")
    
    def load_world(self, filename: str) -> WorldState:
        """Load world state from file"""
        load_path = Path("saved_worlds") / f"{filename}.json"
        
        if not load_path.exists():
            raise FileNotFoundError(f"Save file not found: {filename}")
        
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        # For now, create a basic world from the saved data
        # In production, would use WorldState.from_json()
        world = WorldState(data.get("world_id", "loaded_world"))
        
        # Load basic world properties
        world.weather = data.get("weather", "clear")
        world.season = data.get("season", "spring")
        world.day_count = data.get("day_count", 0)
        
        # Note: Full reconstruction would need to handle all the complex objects
        # For demo purposes, we're showing the save/load concept
        print(f"  Loaded world ID: {world.world_id}")
        print(f"  Note: Full deserialization would restore all characters and locations")
        
        return world


class LocationManager:
    """Manage multiple locations and travel"""
    
    def __init__(self, world: WorldState):
        self.world = world
        self.connections = {}  # Graph of location connections
    
    def add_connection(self, loc1: str, loc2: str, 
                      travel_time: int = 0, 
                      bidirectional: bool = True):
        """Connect two locations"""
        if loc1 not in self.connections:
            self.connections[loc1] = []
        
        self.connections[loc1].append({
            "destination": loc2,
            "travel_time": travel_time
        })
        
        if bidirectional:
            if loc2 not in self.connections:
                self.connections[loc2] = []
            
            self.connections[loc2].append({
                "destination": loc1,
                "travel_time": travel_time
            })
    
    def get_connected_locations(self, location: str) -> List[str]:
        """Get all locations connected to the given one"""
        if location not in self.connections:
            return []
        
        return [conn["destination"] for conn in self.connections[location]]
    
    async def move_character_to(self, 
                              character_name: str, 
                              destination: str,
                              director: Optional[StructuredDirector] = None) -> bool:
        """Move a character to a new location"""
        
        if character_name not in self.world.characters:
            print(f"Character {character_name} not found")
            return False
        
        if destination not in self.world.locations:
            print(f"Location {destination} not found")
            return False
        
        character = self.world.characters[character_name]
        current_loc = character.position.location
        
        # Check if locations are connected
        if destination not in self.get_connected_locations(current_loc):
            print(f"No path from {current_loc} to {destination}")
            return False
        
        # Move character
        character.position.location = destination
        
        # Generate narrative if director provided
        if director:
            travel_prompt = f"{character_name} travels from {current_loc} to {destination}"
            await director.generate_travel_narrative(travel_prompt)
        
        return True
    
    def create_location_graph(self, locations_data: Dict[str, List[str]]):
        """Create a graph from location connection data"""
        for location, connections in locations_data.items():
            for connected_loc in connections:
                self.add_connection(location, connected_loc)


# Preset world templates
PRESET_WORLDS = {
    "fantasy_tavern": {
        "world_name": "The Dragon's Rest Tavern",
        "locations": [{
            "name": "The Dragon's Rest",
            "description": "A cozy tavern with wooden beams and a roaring fireplace",
            "mood": "welcoming",
            "props": ["bar", "tables", "fireplace", "stairs to rooms"],
            "ambient_sounds": ["fire crackling", "mugs clinking", "laughter"]
        }],
        "npcs": [
            {
                "name": "Martha",
                "role": "Innkeeper",
                "description": "A plump, cheerful woman with graying hair",
                "personality": ["friendly", "gossipy", "protective"],
                "initial_position": "behind bar"
            },
            {
                "name": "Hooded Stranger",
                "role": "Mysterious Guest",
                "description": "A figure in dark robes sitting in the corner",
                "personality": ["quiet", "observant", "dangerous"],
                "initial_position": "corner table"
            }
        ],
        "world_state": {
            "time_of_day": "evening",
            "weather": "stormy",
            "tension_level": 40
        }
    },
    
    "cyberpunk_bar": {
        "world_name": "Night City - The Chrome Coffin",
        "locations": [{
            "name": "The Chrome Coffin",
            "description": "A neon-lit dive bar with chrome fixtures and holographic displays",
            "mood": "edgy",
            "props": ["bar", "booths", "dance floor", "VIP section"],
            "ambient_sounds": ["synthwave", "crowd noise", "occasional gunfire outside"]
        }],
        "npcs": [
            {
                "name": "Rex",
                "role": "Bartender",
                "description": "Ex-military with chrome arms and facial scars",
                "personality": ["gruff", "loyal", "streetwise"],
                "initial_position": "behind bar"
            },
            {
                "name": "Nyx",
                "role": "Netrunner",
                "description": "Young hacker with glowing cybernetic eyes",
                "personality": ["paranoid", "brilliant", "antisocial"],
                "initial_position": "corner booth"
            }
        ],
        "world_state": {
            "time_of_day": "night",
            "weather": "acid rain",
            "tension_level": 70
        }
    }
}