"""
Director Agent - Autonomous scene establishment and management
"""

import asyncio
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..models import WorldState, Character, Location, Scene, Position
from .llm_service import LLMService, NarrativeContext, LLMFactory
from .scene_parser import SceneParser
from .director import Director, DirectorConfig


class DirectorAgent(Director):
    """
    Enhanced Director that can autonomously establish scenes from narratives
    Acts as an agent that interprets and builds the world
    """
    
    def __init__(self, world: WorldState, llm_service: Optional[LLMService] = None, 
                 config: Optional[DirectorConfig] = None):
        super().__init__(world, llm_service, config)
        self.scene_parser = SceneParser(llm_service)
        self.established_locations = {}
        self.established_characters = {}
        
    async def establish_scene_from_prompt(self, 
                                         initial_prompt: str,
                                         player_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Establish an entire scene from a narrative prompt
        This is the main agent function that builds the world
        """
        
        print("🎬 Director Agent: Establishing scene from narrative...")
        
        # Step 1: Generate initial narrative from prompt
        context = NarrativeContext(
            world_summary="A new scene is being established",
            location_description="",
            time_of_day="day",
            weather="clear",
            characters=[],
            active_character=player_name or "Observer",
            style_hints=["descriptive", "clear character positions", "atmospheric"],
            content_rating=self.config.content_rating
        )
        
        # Generate rich narrative description
        narrative_prompt = f"""{initial_prompt}

Describe this scene in rich detail, including:
- The exact location and its atmosphere
- Each character present, their appearance, position, and current state
- The relationships between characters
- What each character is doing right now
- The mood and tension of the scene"""

        narrative = await self.llm_service.generate_narrative(
            action=narrative_prompt,
            context=context
        )
        
        print(f"📝 Generated narrative ({len(narrative)} chars)")
        
        # Step 2: Parse the narrative to extract structured data
        print("🔍 Parsing narrative to extract scene elements...")
        parsed_scene = await self.scene_parser.parse_initial_scene(narrative)
        
        # Step 3: Create location from parsed data
        location_data = parsed_scene.get("location", {})
        location = self.scene_parser.create_location_from_parse(location_data)
        self.world.add_location(location)
        print(f"📍 Created location: {location.name}")
        
        # Step 4: Create player character if specified
        player_char = None
        if player_name and parsed_scene.get("player"):
            player_data = parsed_scene["player"]
            player_char = Character(
                name=player_name,
                description=player_data.get("role", "the protagonist"),
                age=30,  # Default age
                personality_traits=["decisive", "observant"]
            )
            
            # Set player position
            x, y = self.scene_parser._parse_position_to_coords(
                player_data.get("position", "center of scene")
            )
            player_char.position = Position(x=x, y=y, z=0, location=location.name)
            
            # Set player clothing
            clothing_state = self.scene_parser._parse_clothing_state(
                player_data.get("clothing", "dressed")
            )
            player_char.change_clothing(clothing_state, player_data.get("clothing", ""))
            
            self.world.add_character(player_char)
            print(f"👤 Created player character: {player_name}")
        
        # Step 5: Create all NPCs from parsed data
        characters_created = []
        for char_data in parsed_scene.get("characters", []):
            character = self.scene_parser.create_character_from_parse(
                char_data, 
                location.name
            )
            
            # Establish relationship with player if applicable
            if player_char:
                relationship = self.scene_parser.parse_relationships_from_context(
                    char_data,
                    player_name
                )
                if relationship:
                    character.relationships[player_name] = relationship
                    # Reciprocal relationship
                    player_char.relationships[character.name] = relationship
            
            self.world.add_character(character)
            characters_created.append(character.name)
            
            print(f"  • Created {character.name}: {char_data.get('description', 'no description')[:50]}...")
        
        # Step 6: Set up the scene
        scene = self.world.set_scene(location.name)
        
        # Apply scene context
        scene_context = parsed_scene.get("scene_context", {})
        tension = scene_context.get("tension_level", "medium")
        if tension == "high":
            scene.tension_level = 70
        elif tension == "low":
            scene.tension_level = 20
        else:
            scene.tension_level = 40
            
        # Step 7: Generate scene summary
        summary = await self._generate_scene_summary()
        
        return {
            "success": True,
            "narrative": narrative,
            "location": location.to_dict(),
            "characters_created": characters_created,
            "player": player_name,
            "parsed_data": parsed_scene,
            "scene_summary": summary,
            "world_state": self.world.to_dict()
        }
    
    async def _generate_scene_summary(self) -> str:
        """Generate a summary of the established scene"""
        if not self.current_scene:
            return "No active scene"
            
        scene = self.current_scene
        location = scene.location
        
        summary_parts = []
        summary_parts.append(f"📍 Location: {location.name}")
        summary_parts.append(f"   {location.description}")
        summary_parts.append(f"   Mood: {location.mood}, Lighting: {location.lighting}")
        
        if scene.present_characters:
            summary_parts.append(f"\n👥 Characters Present ({len(scene.present_characters)}):")
            for char in scene.present_characters:
                summary_parts.append(f"   • {char.name} - {char.position.posture} at ({char.position.x:.1f}, {char.position.y:.1f})")
                summary_parts.append(f"     {char.clothing_description}")
                if char.relationships:
                    for other, rel in char.relationships.items():
                        summary_parts.append(f"     → {other}: {rel.describe()}")
        
        summary_parts.append(f"\n🎭 Scene: {scene.scene_type}, Tension: {scene.tension_level}/100")
        
        return "\n".join(summary_parts)
    
    async def interpret_action_effects(self, 
                                      action: str, 
                                      narrative_response: str) -> Dict[str, Any]:
        """
        Interpret the effects of an action from the narrative response
        Updates world state based on what happened in the narrative
        """
        
        # Create prompt to interpret effects
        interpretation_prompt = f"""Analyze this narrative response and identify what changed:

ACTION TAKEN: {action}

NARRATIVE RESPONSE: {narrative_response}

Identify and return as JSON:
{{
    "position_changes": [
        {{"character": "name", "new_position": "description of where they moved"}}
    ],
    "clothing_changes": [
        {{"character": "name", "new_clothing": "what they're wearing now"}}
    ],
    "emotional_changes": [
        {{"character": "name", "new_emotion": "their emotional state"}}
    ],
    "relationship_changes": [
        {{"character1": "name", "character2": "name", "change": "improved/worsened/romantic tension"}}
    ],
    "new_characters_appeared": [
        {{"name": "name", "description": "who they are", "position": "where they appeared"}}
    ],
    "scene_changes": {{
        "tension_delta": 0,  // How much tension increased/decreased (-20 to +20)
        "mood_change": "new mood if changed",
        "time_passed": "how much time passed if any"
    }}
}}"""

        try:
            # Use the model directly for interpretation
            import google.generativeai as genai
            
            if hasattr(self.llm_service, 'model'):
                model = self.llm_service.model
                response_obj = await model.generate_content_async(interpretation_prompt)
                response = response_obj.text
            else:
                # Fallback
                genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
                model = genai.GenerativeModel('gemini-2.5-flash')
                response_obj = await model.generate_content_async(interpretation_prompt)
                response = response_obj.text
            
            # Parse JSON response
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                effects = json.loads(json_match.group())
                
                # Apply the interpreted changes
                await self._apply_interpreted_changes(effects)
                
                return effects
            
        except Exception as e:
            print(f"Error interpreting effects: {e}")
            
        return {}
    
    async def _apply_interpreted_changes(self, effects: Dict[str, Any]):
        """Apply interpreted changes to the world state"""
        
        # Apply position changes
        for change in effects.get("position_changes", []):
            char_name = change.get("character")
            if char_name in self.world.characters:
                char = self.world.characters[char_name]
                # Parse new position
                new_x, new_y = self.scene_parser._parse_position_to_coords(
                    change.get("new_position", "")
                )
                char.position.x = new_x
                char.position.y = new_y
        
        # Apply clothing changes
        for change in effects.get("clothing_changes", []):
            char_name = change.get("character")
            if char_name in self.world.characters:
                char = self.world.characters[char_name]
                new_clothing = change.get("new_clothing", "")
                clothing_state = self.scene_parser._parse_clothing_state(new_clothing)
                char.change_clothing(clothing_state, new_clothing)
        
        # Apply emotional changes
        for change in effects.get("emotional_changes", []):
            char_name = change.get("character")
            if char_name in self.world.characters:
                char = self.world.characters[char_name]
                char.emotional_state = self.scene_parser._parse_emotional_state(
                    change.get("new_emotion", "calm")
                )
        
        # Apply relationship changes
        for change in effects.get("relationship_changes", []):
            char1 = self.world.characters.get(change.get("character1"))
            char2 = self.world.characters.get(change.get("character2"))
            if char1 and char2:
                rel = char1.get_relationship(char2.name)
                change_type = change.get("change", "")
                if "improved" in change_type:
                    rel.improve_relationship(10, 5)
                elif "worsened" in change_type:
                    rel.damage_relationship(10, 5)
                elif "romantic" in change_type:
                    rel.attraction = min(100, rel.attraction + 15)
                    if rel.attraction > 60:
                        rel.is_romantic = True
        
        # Handle new characters
        for new_char_data in effects.get("new_characters_appeared", []):
            if new_char_data.get("name") not in self.world.characters:
                # Create new character
                new_char = self.scene_parser.create_character_from_parse(
                    new_char_data,
                    self.current_scene.location.name if self.current_scene else "Unknown"
                )
                self.world.add_character(new_char)
                if self.current_scene:
                    self.current_scene.add_character(new_char)
        
        # Apply scene changes
        if self.current_scene:
            scene_changes = effects.get("scene_changes", {})
            tension_delta = scene_changes.get("tension_delta", 0)
            if tension_delta > 0:
                self.current_scene.escalate_tension(tension_delta)
            elif tension_delta < 0:
                self.current_scene.resolve_tension(abs(tension_delta))
            
            if scene_changes.get("mood_change"):
                self.current_scene.location.mood = scene_changes["mood_change"]
    
    async def run_autonomous_scene(self, 
                                  initial_prompt: str,
                                  player_name: str,
                                  num_turns: int = 5) -> List[Dict[str, Any]]:
        """
        Run a complete autonomous scene
        The Director establishes everything and manages the narrative
        """
        
        print("\n🎭 DIRECTOR AGENT: Autonomous Scene Management")
        print("="*60)
        
        # Establish the scene
        setup_result = await self.establish_scene_from_prompt(initial_prompt, player_name)
        
        if not setup_result.get("success"):
            return [{"error": "Failed to establish scene"}]
        
        results = [setup_result]
        
        # Run narrative turns
        for turn in range(num_turns):
            print(f"\n📖 Turn {turn + 1}/{num_turns}")
            
            # Let NPCs act autonomously
            if self.current_scene:
                for npc in self.current_scene.present_characters:
                    if npc.name != player_name:
                        # NPC might act
                        import random
                        if random.random() < self.config.npc_action_chance:
                            context = self._build_context(npc.name)
                            npc_action = await self.llm_service.generate_npc_action(
                                npc.name, context
                            )
                            print(f"  • {npc.name}: {npc_action.get('action', 'observes quietly')}")
            
            await asyncio.sleep(0.5)
        
        return results