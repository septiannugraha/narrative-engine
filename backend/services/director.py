"""
The Director - Orchestrates narrative flow and manages the story
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import random
import json

from ..models import (
    WorldState, Character, Scene, Location,
    ClothingState, EmotionalState, Position
)
from .llm_service import LLMService, NarrativeContext, LLMFactory


@dataclass
class DirectorConfig:
    """Configuration for the Director"""
    
    # Pacing controls
    min_beats_per_scene: int = 5
    max_beats_per_scene: int = 30
    
    # NPC behavior
    npc_action_chance: float = 0.3  # Chance NPC acts autonomously each turn
    npc_interaction_radius: float = 3.0  # Distance for NPC interactions
    
    # Drama controls  
    tension_escalation_rate: float = 0.1
    tension_decay_rate: float = 0.05
    chaos_injection_chance: float = 0.15  # Chance of random events
    
    # Content settings
    content_rating: str = "mature"
    enable_romance: bool = True
    enable_combat: bool = True
    
    # Style preferences
    narrative_style: List[str] = None  # e.g., ["immersive", "sensory", "emotional"]
    
    def __post_init__(self):
        if self.narrative_style is None:
            self.narrative_style = ["immersive", "atmospheric", "character-driven"]


class Director:
    """The Director orchestrates the narrative experience"""
    
    def __init__(self, 
                 world: WorldState,
                 llm_service: Optional[LLMService] = None,
                 config: Optional[DirectorConfig] = None):
        self.world = world
        self.llm_service = llm_service or LLMFactory.create_from_env()
        self.config = config or DirectorConfig()
        
        # Track narrative state
        self.current_scene: Optional[Scene] = None
        self.narrative_log: List[Dict[str, Any]] = []
        self.pending_events: List[Dict[str, Any]] = []
        
    async def process_player_action(self, 
                                   player_name: str,
                                   action: str) -> Dict[str, Any]:
        """Process a player's action and generate narrative response"""
        
        # Get or create current scene
        if not self.current_scene:
            self.current_scene = self._initialize_scene(player_name)
        
        # Build narrative context
        context = self._build_context(player_name)
        
        # Generate narrative response
        narrative = await self.llm_service.generate_narrative(action, context)
        
        # Update world state based on narrative
        self._update_world_from_narrative(narrative, player_name, action)
        
        # Advance scene
        self.current_scene.advance_beat()
        
        # Check for NPC actions
        npc_actions = await self._process_npc_turns(player_name)
        
        # Check for random events
        random_events = self._check_random_events()
        
        # Log the action
        self._log_narrative({
            "type": "player_action",
            "player": player_name,
            "action": action,
            "narrative": narrative,
            "npc_actions": npc_actions,
            "events": random_events,
            "timestamp": datetime.now().isoformat()
        })
        
        # Check if scene should transition
        if self.current_scene.should_transition():
            transition = self._plan_scene_transition()
        else:
            transition = None
        
        return {
            "narrative": narrative,
            "npc_actions": npc_actions,
            "events": random_events,
            "scene_transition": transition,
            "world_state": self.world.to_dict()
        }
    
    def _initialize_scene(self, player_name: str) -> Scene:
        """Initialize a new scene"""
        player = self.world.characters.get(player_name)
        if not player:
            raise ValueError(f"Player {player_name} not found")
        
        # Get location
        location_name = player.position.location
        location = self.world.locations.get(location_name)
        if not location:
            # Create default location
            location = Location(
                name=location_name,
                description="A mysterious place"
            )
            self.world.add_location(location)
        
        # Create scene
        scene = Scene(location=location)
        
        # Add characters in same location
        for char in self.world.characters.values():
            if char.position.location == location_name:
                scene.add_character(char)
        
        # Set scene type based on context
        if len(scene.present_characters) > 1:
            scene.scene_type = "dialogue"
        else:
            scene.scene_type = "exploration"
        
        self.world.active_scene = scene
        return scene
    
    def _build_context(self, active_character: str) -> NarrativeContext:
        """Build context for narrative generation"""
        scene = self.current_scene
        
        # Get character states
        characters = []
        for char in scene.present_characters:
            char_info = {
                "name": char.name,
                "description": char.description,  # Include description for gender clarity
                "position": f"{char.position.posture} at ({char.position.x}, {char.position.y})",
                "clothing": char.clothing_description or char.clothing.value,
                "emotional_state": char.emotional_state.value,
                "arousal": char.arousal_level if self.config.enable_romance else None
            }
            characters.append(char_info)
        
        # Get recent history
        recent_actions = []
        recent_dialogue = []
        for entry in self.narrative_log[-5:]:
            if entry["type"] == "player_action":
                recent_actions.append(f"{entry['player']}: {entry['action']}")
            elif entry.get("dialogue"):
                recent_dialogue.append(entry["dialogue"])
        
        # Determine time of day from current_time
        hour = self.world.current_time.hour
        if hour < 6:
            time_of_day = "night"
        elif hour < 12:
            time_of_day = "morning"
        elif hour < 17:
            time_of_day = "afternoon"
        elif hour < 20:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        
        return NarrativeContext(
            world_summary=f"Time: {time_of_day}, Season: {self.world.season}",
            location_description=scene.location.describe(
                time_of_day,
                self.world.weather
            ),
            time_of_day=time_of_day,
            weather=self.world.weather,
            characters=characters,
            active_character=active_character,
            recent_actions=recent_actions,
            recent_dialogue=recent_dialogue,
            scene_type=scene.scene_type,
            tension_level=scene.tension_level,
            objectives=scene.objectives,
            style_hints=self.config.narrative_style,
            content_rating=self.config.content_rating
        )
    
    async def _process_npc_turns(self, player_name: str) -> List[Dict[str, Any]]:
        """Process autonomous NPC actions"""
        npc_actions = []
        
        player = self.world.characters.get(player_name)
        if not player:
            return npc_actions
        
        for npc in self.current_scene.present_characters:
            if npc.name == player_name:
                continue
            
            # Check if NPC should act
            if random.random() > self.config.npc_action_chance:
                continue
            
            # Check proximity to player
            distance = npc.position.distance_to(player.position)
            if distance > self.config.npc_interaction_radius:
                continue
            
            # Generate NPC action
            context = self._build_context(npc.name)
            action_data = await self.llm_service.generate_npc_action(npc.name, context)
            
            # Process the action
            if action_data.get("action"):
                self._process_npc_action(npc, action_data)
                npc_actions.append({
                    "character": npc.name,
                    **action_data
                })
        
        return npc_actions
    
    def _process_npc_action(self, npc: Character, action_data: Dict[str, Any]):
        """Process an NPC's action"""
        action = action_data.get("action", "")
        
        # Update emotional state based on action
        if "angry" in action.lower() or "shout" in action.lower():
            npc.emotional_state = EmotionalState.ANGRY
        elif "laugh" in action.lower() or "smile" in action.lower():
            npc.emotional_state = EmotionalState.HAPPY
        elif "cry" in action.lower() or "sob" in action.lower():
            npc.emotional_state = EmotionalState.SAD
        
        # Update position if movement detected
        if any(word in action.lower() for word in ["walk", "move", "approach", "back away"]):
            if "approach" in action.lower():
                # Move closer to player
                if self.current_scene.present_characters:
                    target = list(self.current_scene.present_characters)[0]
                    npc.move_toward(target.position, 1.0)
            elif "back away" in action.lower():
                # Move away
                npc.position.x += random.uniform(-2, 2)
                npc.position.y += random.uniform(-2, 2)
        
        # Store the dialogue as memory
        if action_data.get("dialogue"):
            npc.add_memory(f"Said: {action_data['dialogue']}")
    
    def _check_random_events(self) -> List[Dict[str, Any]]:
        """Check for and generate random events"""
        events = []
        
        if random.random() < self.config.chaos_injection_chance:
            event_type = random.choice([
                "environmental",
                "npc_arrival", 
                "discovery",
                "complication"
            ])
            
            if event_type == "environmental":
                events.append({
                    "type": "environmental",
                    "description": self._generate_environmental_event()
                })
            elif event_type == "npc_arrival":
                events.append({
                    "type": "npc_arrival",
                    "description": "Someone new arrives at the scene"
                })
            elif event_type == "discovery":
                events.append({
                    "type": "discovery",
                    "description": "You notice something interesting"
                })
            elif event_type == "complication":
                self.current_scene.escalate_tension(20)
                events.append({
                    "type": "complication",
                    "description": "The situation becomes more complex"
                })
        
        return events
    
    def _generate_environmental_event(self) -> str:
        """Generate a random environmental event"""
        if self.current_scene.location.indoor:
            events = [
                "A door creaks open in the distance",
                "The lights flicker momentarily",
                "You hear footsteps from above",
                "A window rattles in its frame"
            ]
        else:
            events = [
                "The wind picks up suddenly",
                "A bird cries out overhead",
                "Clouds begin to gather",
                "The temperature drops noticeably"
            ]
        
        return random.choice(events)
    
    def _update_world_from_narrative(self, 
                                    narrative: str,
                                    player_name: str,
                                    action: str):
        """Update world state based on generated narrative"""
        
        # Simple heuristic updates based on keywords
        narrative_lower = narrative.lower()
        
        # Update tension
        if any(word in narrative_lower for word in ["tense", "danger", "threat", "conflict"]):
            self.current_scene.escalate_tension(10)
        elif any(word in narrative_lower for word in ["calm", "peaceful", "relax"]):
            self.current_scene.resolve_tension(10)
        
        # Update mood
        if "romantic" in narrative_lower or "intimate" in narrative_lower:
            self.current_scene.location.mood = "intimate"
        elif "festive" in narrative_lower or "celebrat" in narrative_lower:
            self.current_scene.location.mood = "festive"
        elif "mysterious" in narrative_lower or "eerie" in narrative_lower:
            self.current_scene.location.mood = "mysterious"
    
    def _plan_scene_transition(self) -> Dict[str, Any]:
        """Plan a transition to a new scene"""
        suggestions = []
        
        # Suggest location changes
        for location_name in self.world.locations:
            if location_name != self.current_scene.location.name:
                suggestions.append({
                    "type": "location_change",
                    "target": location_name,
                    "reason": "Explore new area"
                })
        
        # Suggest time advancement
        suggestions.append({
            "type": "time_advance",
            "duration": "1 hour",
            "reason": "Let time pass"
        })
        
        return {
            "ready_for_transition": True,
            "suggestions": suggestions
        }
    
    def _log_narrative(self, entry: Dict[str, Any]):
        """Log narrative entry"""
        self.narrative_log.append(entry)
        
        # Keep log size manageable
        if len(self.narrative_log) > 100:
            self.narrative_log = self.narrative_log[-50:]
    
    async def generate_scene_description(self, location_name: str) -> str:
        """Generate a rich description of a scene"""
        location = self.world.locations.get(location_name)
        if not location:
            return "An unknown location."
        
        # Build context for scene description
        characters_present = [
            char for char in self.world.characters.values()
            if char.position.location == location_name
        ]
        
        context = NarrativeContext(
            world_summary=f"Time: {self.world.current_time.strftime('%H:%M')}, Season: {self.world.season}",
            location_description=location.description,
            time_of_day="day",
            weather=self.world.weather,
            characters=[{"name": c.name} for c in characters_present],
            active_character="Observer",
            style_hints=["descriptive", "atmospheric", "immersive"]
        )
        
        prompt = "Describe the scene in rich, sensory detail."
        return await self.llm_service.generate_narrative(prompt, context)
    
    def get_story_summary(self) -> str:
        """Get a summary of the story so far"""
        if not self.narrative_log:
            return "The story has just begun."
        
        summary_parts = []
        
        # Count actions by character
        action_counts = {}
        for entry in self.narrative_log:
            if entry["type"] == "player_action":
                player = entry["player"]
                action_counts[player] = action_counts.get(player, 0) + 1
        
        # Get key events
        key_events = [
            entry for entry in self.narrative_log
            if entry.get("events") or entry.get("scene_transition")
        ]
        
        summary_parts.append(f"Story beats: {len(self.narrative_log)}")
        summary_parts.append(f"Active characters: {', '.join(action_counts.keys())}")
        
        if key_events:
            summary_parts.append(f"Key events: {len(key_events)}")
        
        if self.current_scene:
            summary_parts.append(f"Current location: {self.current_scene.location.name}")
            summary_parts.append(f"Scene tension: {self.current_scene.tension_level}/100")
        
        return " | ".join(summary_parts)


class StoryRecorder:
    """Records and formats the story for export"""
    
    def __init__(self, director: Director):
        self.director = director
        self.chapters: List[Dict[str, Any]] = []
        self.current_chapter: Dict[str, Any] = None
    
    def start_chapter(self, title: str):
        """Start a new chapter"""
        if self.current_chapter:
            self.chapters.append(self.current_chapter)
        
        self.current_chapter = {
            "title": title,
            "started": datetime.now().isoformat(),
            "entries": []
        }
    
    def add_entry(self, entry_type: str, content: str, metadata: Dict = None):
        """Add an entry to the current chapter"""
        if not self.current_chapter:
            self.start_chapter("Chapter 1")
        
        self.current_chapter["entries"].append({
            "type": entry_type,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def export_story(self, format: str = "markdown") -> str:
        """Export the story in various formats"""
        if format == "markdown":
            return self._export_markdown()
        elif format == "json":
            return self._export_json()
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _export_markdown(self) -> str:
        """Export story as markdown"""
        lines = ["# Story Transcript\n"]
        
        for chapter in self.chapters:
            lines.append(f"\n## {chapter['title']}\n")
            lines.append(f"*Started: {chapter['started']}*\n")
            
            for entry in chapter["entries"]:
                if entry["type"] == "narrative":
                    lines.append(f"\n{entry['content']}\n")
                elif entry["type"] == "dialogue":
                    speaker = entry["metadata"].get("speaker", "Unknown")
                    lines.append(f'\n**{speaker}:** "{entry["content"]}"\n')
                elif entry["type"] == "action":
                    actor = entry["metadata"].get("actor", "Someone")
                    lines.append(f"\n*{actor} {entry['content']}*\n")
        
        return "\n".join(lines)
    
    def _export_json(self) -> str:
        """Export story as JSON"""
        return json.dumps({
            "story": self.chapters,
            "metadata": {
                "total_chapters": len(self.chapters),
                "exported": datetime.now().isoformat()
            }
        }, indent=2)