"""
Scene Parser - Extracts structured data from narrative descriptions
"""

import json
import re
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..models import (
    Character, Location, Position, ClothingState, 
    EmotionalState, Relationship, RelationshipType
)
from .llm_service import NarrativeContext


@dataclass
class ParsedCharacter:
    """Parsed character information from narrative"""
    name: str
    description: str
    gender: str
    age_estimate: str
    position_description: str
    clothing_description: str
    emotional_state: str
    relationship_to_player: str
    personality_hints: List[str]


@dataclass
class ParsedLocation:
    """Parsed location information from narrative"""
    name: str
    description: str
    props: List[str]
    mood: str
    adjacent_locations: List[str]
    environmental_details: Dict[str, Any]


class SceneParser:
    """Parses narrative text to extract structured scene data"""
    
    def __init__(self, llm_service):
        self.llm_service = llm_service
    
    async def parse_initial_scene(self, narrative_text: str) -> Dict[str, Any]:
        """
        Parse an initial scene narrative to extract all scene elements
        Uses LLM to understand and structure the narrative
        """
        
        # Create a structured prompt for the LLM to parse the narrative
        parsing_prompt = f"""Analyze this narrative scene and extract structured information.

NARRATIVE:
{narrative_text}

Extract and return a JSON object with this EXACT structure:
{{
    "location": {{
        "name": "Name of the current location",
        "description": "Brief description",
        "props": ["list", "of", "objects", "in", "scene"],
        "mood": "emotional atmosphere (intimate/tense/cheerful/mysterious/etc)",
        "lighting": "lighting conditions",
        "sounds": ["ambient", "sounds"],
        "adjacent_areas": ["connected", "locations"]
    }},
    "characters": [
        {{
            "name": "Character name",
            "description": "Physical appearance and notable features",
            "gender": "male/female/other",
            "approximate_age": "young/adult/middle-aged/elderly or number",
            "position": "where they are in the scene",
            "posture": "standing/sitting/lying/etc",
            "clothing": "what they're wearing",
            "emotional_state": "calm/excited/nervous/angry/etc",
            "relationship_to_player": "stranger/acquaintance/friend/lover/rival/etc",
            "personality_hints": ["traits", "shown", "in", "scene"]
        }}
    ],
    "player": {{
        "name": "Player character name",
        "position": "where the player is",
        "clothing": "what player is wearing",
        "role": "player's role/title if mentioned"
    }},
    "scene_context": {{
        "time": "time of day/night",
        "weather": "if applicable",
        "tension_level": "low/medium/high",
        "immediate_situation": "what's happening right now"
    }}
}}

Be precise and extract ONLY information explicitly stated or strongly implied in the narrative."""

        try:
            # Use the LLM service model directly for parsing
            import google.generativeai as genai
            
            # Get the model from the service if it's Gemini
            if hasattr(self.llm_service, 'model'):
                model = self.llm_service.model
                response_obj = await model.generate_content_async(parsing_prompt)
                response = response_obj.text
            else:
                # Fallback - create a temporary model for parsing
                genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
                model = genai.GenerativeModel('gemini-2.5-flash')
                response_obj = await model.generate_content_async(parsing_prompt)
                response = response_obj.text
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
                return parsed_data
            else:
                # Fallback: try to parse the whole response
                return json.loads(response)
                
        except Exception as e:
            print(f"Error parsing scene: {e}")
            # Return a basic structure if parsing fails
            return self._create_fallback_parse(narrative_text)
    
    def _create_fallback_parse(self, narrative_text: str) -> Dict[str, Any]:
        """Create a basic parse if LLM parsing fails"""
        return {
            "location": {
                "name": "Unknown Location",
                "description": "A location described in the narrative",
                "props": [],
                "mood": "neutral"
            },
            "characters": [],
            "player": {
                "name": "Player",
                "position": "present in scene"
            },
            "scene_context": {
                "time": "unknown",
                "tension_level": "medium"
            }
        }
    
    def create_location_from_parse(self, location_data: Dict[str, Any]) -> Location:
        """Create a Location object from parsed data"""
        return Location(
            name=location_data.get("name", "Unknown Location"),
            description=location_data.get("description", ""),
            props=location_data.get("props", []),
            mood=location_data.get("mood", "neutral"),
            lighting=location_data.get("lighting", "normal"),
            sounds=location_data.get("sounds", []),
            scents=location_data.get("scents", []),
            indoor=location_data.get("indoor", True),
            temperature=location_data.get("temperature", "comfortable")
        )
    
    def create_character_from_parse(self, char_data: Dict[str, Any], 
                                   location_name: str) -> Character:
        """Create a Character object from parsed data"""
        
        # Determine age from description
        age_str = char_data.get("approximate_age", "adult")
        if isinstance(age_str, int):
            age = age_str
        elif "young" in age_str.lower():
            age = 22
        elif "elderly" in age_str.lower():
            age = 65
        elif "middle" in age_str.lower():
            age = 45
        else:
            age = 30  # Default adult age
        
        # Create character
        character = Character(
            name=char_data.get("name", "Unknown"),
            description=char_data.get("description", ""),
            age=age,
            personality_traits=char_data.get("personality_hints", [])
        )
        
        # Set position
        position_desc = char_data.get("position", "in the scene")
        posture = char_data.get("posture", "standing")
        
        # Assign coordinates based on position description
        x, y = self._parse_position_to_coords(position_desc)
        character.position = Position(
            x=x, y=y, z=0,
            location=location_name,
            posture=posture
        )
        
        # Set clothing
        clothing_desc = char_data.get("clothing", "dressed")
        clothing_state = self._parse_clothing_state(clothing_desc)
        character.change_clothing(clothing_state, clothing_desc)
        
        # Set emotional state
        emotion_str = char_data.get("emotional_state", "calm")
        character.emotional_state = self._parse_emotional_state(emotion_str)
        
        return character
    
    def _parse_position_to_coords(self, position_desc: str) -> Tuple[float, float]:
        """Convert position description to coordinates"""
        position_lower = position_desc.lower()
        
        # Parse relative positions
        if "center" in position_lower or "middle" in position_lower:
            return (0, 0)
        elif "left" in position_lower or "port" in position_lower:
            x = -3
        elif "right" in position_lower or "starboard" in position_lower:
            x = 3
        else:
            x = 0
            
        if "front" in position_lower or "forward" in position_lower:
            y = 3
        elif "back" in position_lower or "rear" in position_lower or "aft" in position_lower:
            y = -3
        else:
            y = 0
            
        # Add some variation if multiple characters are in same general area
        import random
        x += random.uniform(-1, 1)
        y += random.uniform(-1, 1)
        
        return (x, y)
    
    def _parse_clothing_state(self, clothing_desc: str) -> ClothingState:
        """Parse clothing description to ClothingState enum"""
        clothing_lower = clothing_desc.lower()
        
        if "armor" in clothing_lower:
            return ClothingState.ARMOR
        elif "formal" in clothing_lower or "uniform" in clothing_lower:
            return ClothingState.FORMAL
        elif "towel" in clothing_lower:
            return ClothingState.TOWEL
        elif "naked" in clothing_lower or "nude" in clothing_lower:
            return ClothingState.NAKED
        elif "underwear" in clothing_lower:
            return ClothingState.UNDERWEAR
        elif "pajama" in clothing_lower or "sleepwear" in clothing_lower:
            return ClothingState.SLEEP
        elif "casual" in clothing_lower:
            return ClothingState.CASUAL
        else:
            return ClothingState.FULL
    
    def _parse_emotional_state(self, emotion_str: str) -> EmotionalState:
        """Parse emotion string to EmotionalState enum"""
        emotion_lower = emotion_str.lower()
        
        if "calm" in emotion_lower or "serene" in emotion_lower:
            return EmotionalState.CALM
        elif "happy" in emotion_lower or "cheerful" in emotion_lower:
            return EmotionalState.HAPPY
        elif "excited" in emotion_lower or "energetic" in emotion_lower:
            return EmotionalState.EXCITED
        elif "nervous" in emotion_lower or "anxious" in emotion_lower:
            return EmotionalState.NERVOUS
        elif "embarrassed" in emotion_lower or "flustered" in emotion_lower:
            return EmotionalState.EMBARRASSED
        elif "aroused" in emotion_lower or "lustful" in emotion_lower:
            return EmotionalState.AROUSED
        elif "angry" in emotion_lower or "furious" in emotion_lower:
            return EmotionalState.ANGRY
        elif "sad" in emotion_lower or "melancholy" in emotion_lower:
            return EmotionalState.SAD
        elif "fearful" in emotion_lower or "afraid" in emotion_lower:
            return EmotionalState.FEARFUL
        else:
            return EmotionalState.CALM
    
    def parse_relationships_from_context(self, 
                                        char_data: Dict[str, Any],
                                        player_name: str) -> Optional[Relationship]:
        """Parse relationship between character and player"""
        rel_str = char_data.get("relationship_to_player", "stranger")
        rel_lower = rel_str.lower()
        
        relationship = Relationship()
        
        if "stranger" in rel_lower:
            relationship.affection = 0
            relationship.trust = 0
            relationship.relationship_type = RelationshipType.STRANGER
        elif "acquaintance" in rel_lower:
            relationship.affection = 20
            relationship.trust = 20
            relationship.relationship_type = RelationshipType.ACQUAINTANCE
        elif "friend" in rel_lower:
            relationship.affection = 60
            relationship.trust = 60
            relationship.relationship_type = RelationshipType.FRIEND
        elif "lover" in rel_lower or "romantic" in rel_lower:
            relationship.affection = 80
            relationship.trust = 70
            relationship.attraction = 80
            relationship.is_romantic = True
            relationship.relationship_type = RelationshipType.LOVER
        elif "rival" in rel_lower:
            relationship.affection = 10
            relationship.trust = 20
            relationship.relationship_type = RelationshipType.RIVAL
        elif "ally" in rel_lower:
            relationship.affection = 50
            relationship.trust = 70
            relationship.relationship_type = RelationshipType.ALLY
            
        return relationship