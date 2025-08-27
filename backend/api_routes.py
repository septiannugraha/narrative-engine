"""
API Routes for Narrative Engine with Structured Director Integration
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import json
import asyncio
from pathlib import Path

from .models import WorldState
from .services.structured_director import StructuredDirector
from .services.world_builder import WorldTemplate, PRESET_WORLDS
from .services.director import DirectorConfig
from .services.persistence import persistence_service, GameEvent
from .services.character_manager import character_manager
from .services.session_manager import session_manager

router = APIRouter()

# Store active game sessions - now managed by session_manager
active_sessions: Dict[str, Dict[str, Any]] = {}


class CharacterCreation(BaseModel):
    name: str
    description: str
    world_type: str = "fantasy_tavern"  # or "cyberpunk_bar" or "custom"
    custom_world_prompt: Optional[str] = None


class PlayerAction(BaseModel):
    action: str
    session_id: str


class GameSession:
    """Manages a single game session"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.world: Optional[WorldState] = None
        self.director: Optional[StructuredDirector] = None
        self.player_name: str = ""
        self.player_description: str = ""
        self.websocket: Optional[WebSocket] = None
        
    async def initialize(self, character: CharacterCreation):
        """Initialize game session with character and world"""
        self.player_name = character.name
        self.player_description = character.description
        
        # Initialize persistence
        await persistence_service.initialize()
        await persistence_service.manager.create_session(
            self.session_id,
            self.player_name,
            character.world_type,
            {"description": self.player_description}
        )
        
        # Create world based on selection
        if character.world_type in ["fantasy_tavern", "cyberpunk_bar"]:
            # Use built-in template
            template = WorldTemplate(template_data=PRESET_WORLDS[character.world_type])
            self.world = template.to_world_state()
        elif character.world_type != "custom":
            # Try to load template from file
            import json
            from pathlib import Path
            template_filename = character.world_type if character.world_type.endswith('.json') else f"{character.world_type}.json"
            template_path = Path(__file__).parent / "world_templates" / template_filename
            
            if template_path.exists():
                print(f"📍 Loading template from: {template_path}")
                with open(template_path, 'r') as f:
                    template_data = json.load(f)
                template = WorldTemplate(template_data=template_data)
                self.world = template.to_world_state()
                print(f"✅ Loaded template: {template_data.get('world_name', character.world_type)}")
            else:
                print(f"⚠️ Template file not found: {template_path}")
                # Fallback to empty world
                self.world = WorldState(f"custom_{self.session_id}")
        else:
            # Create empty world for custom
            self.world = WorldState(f"custom_{self.session_id}")
        
        # Create director
        self.director = StructuredDirector(
            world=self.world,
            config=DirectorConfig(
                content_rating="mature",
                enable_romance=True,
                narrative_style=["immersive", "character-driven", "atmospheric"]
            )
        )
        
        # Establish initial scene
        if character.world_type == "custom" and character.custom_world_prompt:
            prompt = f"""
            {character.custom_world_prompt}
            
            The player character is {self.player_name}.
            Description: {self.player_description}
            """
        else:
            # Template world - player enters
            location_obj = list(self.world.locations.values())[0] if self.world.locations else None
            location_name = location_obj.name if location_obj else "Unknown"
            location_desc = location_obj.description if location_obj else "A mysterious place"
            
            # Get existing NPCs from template
            npc_list = []
            for char_name, char in self.world.characters.items():
                npc_list.append(f"- {char_name}: {char.description}")
            
            # Build prompt based on actual location
            if npc_list:
                npc_text = f"""
            IMPORTANT: The following NPCs are already present:
            {chr(10).join(npc_list)}
            
            These NPCs notice the new arrival and react accordingly."""
            else:
                npc_text = "The location seems quiet, waiting for someone to arrive."
            
            prompt = f"""
            Setting: {location_name}
            {location_desc}
            
            {self.player_name} enters the scene.
            Character description: {self.player_description}
            {npc_text}
            
            Establish the scene with {self.player_name}'s entrance in this specific setting.
            """
        
        result = await self.director.establish_scene_from_prompt(
            initial_prompt=prompt,
            player_name=self.player_name
        )
        
        return result
    
    async def process_action(self, action: str) -> Dict[str, Any]:
        """Process player action"""
        if not self.director:
            raise ValueError("Session not initialized")
        
        result = await self.director.process_action_with_structure(
            player_name=self.player_name,
            action=action
        )
        
        # Save action to persistence
        try:
            await persistence_service.save_action(
                self.session_id,
                self.player_name,
                action,
                result
            )
            
            # Save checkpoint every 10 actions
            import uuid
            from datetime import datetime
            event_count = await self._get_event_count()
            if event_count % 10 == 0:
                await persistence_service.save_checkpoint(
                    self.session_id,
                    self.world,
                    f"Checkpoint after {event_count} actions"
                )
        except Exception as e:
            print(f"⚠️ Persistence error (non-fatal): {e}")
        
        return result
    
    async def _get_event_count(self) -> int:
        """Get number of events for this session"""
        try:
            stats = await persistence_service.manager.get_session_stats(self.session_id)
            return stats.get("total_events", 0)
        except:
            return 0
    
    def get_scene_data(self) -> Dict[str, Any]:
        """Get current scene data"""
        if not self.director or not self.director.current_scene:
            return {}
        
        scene = self.director.current_scene
        characters_data = []
        
        for char in scene.present_characters:
            char_data = {
                "name": char.name,
                "description": char.description,
                "position": f"{char.position.posture}",
                "is_player": char.name == self.player_name,
                "emotional_state": char.emotional_state.value if hasattr(char, 'emotional_state') else "calm",
                "clothing": char.clothing.value if hasattr(char, 'clothing') else "fully dressed",
                "relationship": "neutral"  # Default
            }
            
            # Add NPC-specific data if this is an NPC
            if char.name != self.player_name and hasattr(self.director, 'memory_service'):
                # Get relationship from character object
                if hasattr(char, 'relationships') and self.player_name in char.relationships:
                    rel = char.relationships[self.player_name]
                    if hasattr(rel, 'relationship_type'):
                        char_data["relationship"] = rel.relationship_type.value.lower()
                
                # Get memory summary for this NPC
                try:
                    npc_context = self.director.memory_service.get_npc_context(char.name, self.player_name)
                    if npc_context:
                        # Add memory count and details
                        memories = npc_context.get('memories_of_player', [])
                        char_data["memory_count"] = len(memories)
                        
                        # Add most recent memory with proper formatting
                        if memories and len(memories) > 0:
                            char_data["last_memory"] = memories[0].get('when', 'Recently')
                        
                        # Add actual gossip content (not just a flag)
                        recent_gossip = npc_context.get('recent_gossip', [])
                        if recent_gossip:
                            char_data["knows_gossip"] = True
                            char_data["recent_gossip"] = recent_gossip[:2]  # Last 2 gossip items
                        
                        # Add world knowledge
                        recent_events = npc_context.get('recent_events', [])
                        if recent_events:
                            char_data["world_knowledge"] = recent_events[:2]
                except Exception as e:
                    print(f"Warning: Failed to get memory context for {char.name}: {e}")
                    # Continue without memory data rather than crash
            
            characters_data.append(char_data)
        
        return {
            "location": {
                "name": scene.location.name,
                "description": scene.location.description,
                "mood": scene.location.mood
            },
            "characters": characters_data,
            "tension": scene.tension_level,
            "world_state": self.director.world_state if hasattr(self.director, 'world_state') else {}
        }


@router.post("/api/game/start")
async def start_game(character: CharacterCreation) -> Dict[str, Any]:
    """Start a new game session"""
    import uuid
    session_id = str(uuid.uuid4())
    
    # Create session
    session = GameSession(session_id)
    active_sessions[session_id] = session
    session_manager.add_session(session_id, session)
    
    # Initialize game
    result = await session.initialize(character)
    
    if result.get("success"):
        scene_data = result.get("scene_data", {})
        return {
            "success": True,
            "session_id": session_id,
            "narrative": scene_data.get("narrative", ""),
            "location": scene_data.get("location", {}),
            "characters": scene_data.get("npc_characters", []),
            "player": {
                "name": character.name,
                "description": character.description
            }
        }
    else:
        return {
            "success": False,
            "error": result.get("error", "Failed to initialize game")
        }


@router.post("/api/game/action")
async def process_action(action: PlayerAction) -> Dict[str, Any]:
    """Process a player action"""
    session = active_sessions.get(action.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    result = await session.process_action(action.action)
    
    # Debug logging
    dialogue_data = result.get("dialogue", [])
    print(f"📤 HTTP Response: narrative={len(result.get('narrative', ''))} chars, dialogue={len(dialogue_data)} exchanges")
    for d in dialogue_data:
        print(f"   - {d.get('speaker', 'Unknown')}: {d.get('dialogue', '')[:50]}...")
    
    return {
        "success": result.get("success", False),
        "narrative": result.get("narrative", ""),
        "dialogue": dialogue_data,
        "scene_updates": session.get_scene_data(),
        "was_retry": result.get("was_retry", False),
        "response_time": result.get("response_time", 0)
    }


@router.get("/api/game/scene/{session_id}")
async def get_scene(session_id: str) -> Dict[str, Any]:
    """Get current scene data"""
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.get_scene_data()


@router.post("/api/game/save/{session_id}")
async def save_game(session_id: str, save_name: str) -> Dict[str, Any]:
    """Save game state"""
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Save world state
    from .services.world_builder import WorldBuilder
    builder = WorldBuilder()
    Path("saved_worlds").mkdir(exist_ok=True)
    builder.save_world(session.world, save_name)
    
    return {"success": True, "message": f"Game saved as {save_name}"}


@router.websocket("/ws/game/{session_id}")
async def game_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for real-time game updates"""
    await websocket.accept()
    
    session = active_sessions.get(session_id)
    if not session:
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return
    
    session.websocket = websocket
    
    try:
        while True:
            # Receive action from client
            data = await websocket.receive_json()
            action = data.get("action", "")
            
            if action == "quit":
                break
            
            # Process action
            result = await session.process_action(action)
            
            # Send response
            response_data = {
                "type": "action_result",
                "narrative": result.get("narrative", ""),
                "dialogue": result.get("dialogue", []),
                "scene": session.get_scene_data()
            }
            
            # Debug: log what we're sending
            print(f"📤 WebSocket sending: narrative={len(response_data['narrative'])} chars, dialogue={len(response_data['dialogue'])} exchanges")
            for d in response_data['dialogue']:
                print(f"   - {d.get('speaker', 'Unknown')}: {d.get('dialogue', '')[:50]}...")
            
            await websocket.send_json(response_data)
            
    except WebSocketDisconnect:
        # Clean up session
        if session_id in active_sessions:
            del active_sessions[session_id]
    except Exception as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close()


@router.get("/api/templates")
async def get_templates() -> List[Dict[str, Any]]:
    """Get available world templates from the world_templates directory"""
    import os
    import json
    from pathlib import Path
    
    templates = []
    templates_dir = Path(__file__).parent / "world_templates"
    
    # Always include custom option
    templates.append({
        "id": "custom",
        "name": "Custom World",
        "description": "Create your own unique setting",
        "icon": "🌍"
    })
    
    if templates_dir.exists():
        for template_file in templates_dir.glob("*.json"):
            try:
                with open(template_file, 'r') as f:
                    template_data = json.load(f)
                    template_id = template_file.stem  # filename without extension
                    templates.append({
                        "id": template_id,
                        "name": template_data.get("world_name", template_id.replace("_", " ").title()),
                        "description": template_data.get("description", "A unique world setting"),
                        "icon": template_data.get("icon", "🏰"),
                        "mood": template_data.get("locations", [{}])[0].get("mood", "mysterious")
                    })
            except Exception as e:
                print(f"Error loading template {template_file}: {e}")
    
    return templates


@router.get("/api/sessions")
async def get_sessions() -> List[Dict]:
    """Get all active game sessions"""
    await persistence_service.initialize()
    sessions = await persistence_service.manager.get_active_sessions()
    return sessions


@router.get("/api/session/{session_id}/memories")
async def get_session_memories(session_id: str) -> Dict[str, Any]:
    """Get all NPC memories for a session"""
    # First check if session exists in active sessions
    if session_id not in active_sessions:
        # Try to load from database
        await persistence_service.initialize()
        session_data = await persistence_service.manager.load_snapshot(session_id)
        if not session_data:
            return {"error": "Session not found", "memories": {}}
        
        # For now, return a message that session needs to be resumed
        return {
            "error": "Session not active",
            "message": "This session exists but is not currently active. Resume the session to view memories.",
            "memories": {},
            "session_info": {
                "player_name": session_data.get("player_name", "Unknown"),
                "world_type": session_data.get("world_type", "Unknown"),
                "created_at": session_data.get("created_at", "Unknown")
            }
        }
    
    session = active_sessions[session_id]
    
    # Check if session has a director
    if not hasattr(session, 'director') or session.director is None:
        return {"error": "Session not fully initialized", "memories": {}, "message": "Director not initialized for this session"}
    
    if not hasattr(session.director, 'memory_service'):
        return {"error": "Memory service not initialized", "memories": {}, "message": "Memory tracking not enabled for this session"}
    
    memory_service = session.director.memory_service
    all_memories = {}
    
    # Get memories for each NPC
    for npc_name in memory_service.npcs.keys():
        npc_memories = memory_service.get_npc_memories(npc_name)
        
        # Convert memories to serializable format
        all_memories[npc_name] = [
            {
                "type": mem.memory_type.value,
                "content": mem.content,
                "importance": mem.importance.value,
                "about": list(mem.about_characters) if mem.about_characters else [],
                "tags": list(mem.tags),
                "timestamp": mem.timestamp.isoformat(),
                "decay": mem.calculate_decay(memory_service.current_time),
                "effective_importance": mem.get_effective_importance(memory_service.current_time)
            }
            for mem in npc_memories
        ]
    
    # Get gossip network
    gossip_network = {}
    for npc, connections in memory_service.gossip_network.items():
        gossip_network[npc] = list(connections)
    
    # Get world events
    world_events = [
        {
            "type": event.event_type,
            "description": event.description,
            "participants": list(event.participants),
            "timestamp": event.timestamp.isoformat(),
            "importance": event.importance.value
        }
        for event in memory_service.world_events[-10:]  # Last 10 events
    ]
    
    return {
        "success": True,
        "memories": all_memories,
        "gossip_network": gossip_network,
        "world_events": world_events,
        "world_time": memory_service.current_time.isoformat(),
        "total_memories": sum(len(mems) for mems in all_memories.values())
    }


@router.get("/api/session/{session_id}/npc/{npc_name}/memories")
async def get_npc_memories(session_id: str, npc_name: str) -> Dict[str, Any]:
    """Get memories for a specific NPC"""
    if session_id not in active_sessions:
        return {"error": "Session not found"}
    
    session = active_sessions[session_id]
    
    if not hasattr(session.director, 'memory_service'):
        return {"error": "Memory service not initialized"}
    
    memory_service = session.director.memory_service
    
    if npc_name not in memory_service.npcs:
        return {"error": f"NPC {npc_name} not found"}
    
    # Get NPC profile
    npc_profile = memory_service.npcs[npc_name]
    
    # Get memories
    memories = memory_service.get_npc_memories(npc_name)
    
    # Get memories about the player
    player_memories = memory_service.get_npc_memories(
        npc_name, 
        about_character=session.player_name
    )
    
    return {
        "success": True,
        "npc": {
            "name": npc_profile.name,
            "personality": npc_profile.personality_traits,
            "interests": npc_profile.interests,
            "memory_tendency": npc_profile.memory_tendency.value
        },
        "total_memories": len(memories),
        "memories_about_player": len(player_memories),
        "recent_memories": [
            {
                "content": mem.content,
                "type": mem.memory_type.value,
                "importance": mem.importance.value,
                "about": list(mem.about_characters) if mem.about_characters else [],
                "timestamp": mem.timestamp.isoformat()
            }
            for mem in memories[:10]  # Last 10 memories
        ],
        "player_memories": [
            {
                "content": mem.content,
                "type": mem.memory_type.value,
                "importance": mem.importance.value,
                "timestamp": mem.timestamp.isoformat()
            }
            for mem in player_memories[:5]  # Last 5 about player
        ]
    }


@router.get("/api/session/{session_id}/events")
async def get_session_events(session_id: str, limit: int = 100) -> List[Dict]:
    """Get events for a session (for debugging)"""
    await persistence_service.initialize()
    events = await persistence_service.manager.get_events(session_id, limit=limit)
    return [
        {
            "event_id": e.event_id,
            "timestamp": e.timestamp.isoformat(),
            "type": e.event_type,
            "actor": e.actor,
            "data": e.data
        }
        for e in events
    ]


@router.get("/api/session/{session_id}/stats")
async def get_session_stats(session_id: str) -> Dict:
    """Get statistics for a session"""
    await persistence_service.initialize()
    stats = await persistence_service.manager.get_session_stats(session_id)
    return stats


@router.get("/api/session/{session_id}/status")
async def get_session_status(session_id: str) -> Dict[str, Any]:
    """Check if a session is still active and can be resumed"""
    # Check if session is in active memory
    if session_id in active_sessions:
        session = active_sessions[session_id]
        return {
            "active": True,
            "in_memory": True,
            "player_name": session.player_name,
            "location": session.world.active_scene.location.name if session.world and session.world.active_scene and session.world.active_scene.location else "Unknown"
        }
    
    # Check persistence layer
    await persistence_service.initialize()
    events = await persistence_service.manager.get_events(session_id, limit=1)
    
    if events:
        # Session exists in storage
        stats = await persistence_service.manager.get_session_stats(session_id)
        return {
            "active": True,
            "in_memory": False,
            "player_name": stats.get('player_name', 'Unknown'),
            "event_count": stats.get('event_count', 0),
            "can_resume": True
        }
    
    return {
        "active": False,
        "in_memory": False,
        "can_resume": False
    }


@router.get("/api/session/{session_id}/resume")
async def resume_session(session_id: str) -> Dict[str, Any]:
    """Resume an existing session from persistence or memory"""
    
    # Check if already in active memory
    if session_id in active_sessions:
        session = active_sessions[session_id]
        
        # Build response from active session
        world = session.world
        director = session.director
        
        # Get conversation history
        history = []
        if director and hasattr(director, 'conversation_history'):
            for entry in director.conversation_history[-20:]:  # Last 20 exchanges
                history.append({
                    "action": entry.get('action', ''),
                    "narrative": entry.get('narrative', ''),
                    "dialogue": entry.get('dialogue', [])
                })
        
        characters = []
        if world:
            for char_name, char in world.characters.items():
                characters.append({
                    "name": char.name,
                    "is_player": char.name == session.player_name,
                    "position": char.position.posture if hasattr(char, 'position') and hasattr(char.position, 'posture') else "present",
                    "description": char.description[:100] if char.description else ""
                })
        
        return {
            "success": True,
            "from_memory": True,
            "player_name": session.player_name,
            "location": {
                "name": world.active_scene.location.name if world and world.active_scene and world.active_scene.location else "Unknown",
                "mood": world.active_scene.location.mood if world and world.active_scene and world.active_scene.location else "",
                "description": world.active_scene.location.description if world and world.active_scene and world.active_scene.location else ""
            },
            "characters": characters,
            "tension": world.active_scene.tension_level if world and world.active_scene and hasattr(world.active_scene, 'tension_level') else 0,
            "history": history,
            "last_narrative": history[-1]['narrative'] if history else "Your adventure continues..."
        }
    
    # Try to restore from persistence
    await persistence_service.initialize()
    
    try:
        # Get recent events for history
        events = await persistence_service.manager.get_events(session_id, limit=50)
        if not events:
            return {
                "success": False,
                "error": "Session not found"
            }
        
        # Get session info from database
        stats = await persistence_service.manager.get_session_stats(session_id)
        player_name = stats.get('player_name', 'Unknown')
        
        # Build history from events
        history = []
        
        # Try to get location details from events
        # Set defaults based on world type
        if stats.get('world_type') == 'fantasy_tavern':
            location_name = "The Dragon's Rest"
            location_mood = "welcoming"
            location_description = "A cozy tavern with wooden beams and a roaring fireplace"
            tension_level = 40
        else:
            location_name = "Unknown Location"
            location_mood = ""
            location_description = ""
            tension_level = 0
        
        # Look for scene/location events in the event history
        for event in events:
            if event.event_type == 'scene_established':
                scene_data = event.data
                if 'location' in scene_data:
                    location_name = scene_data['location'].get('name', location_name)
                    location_mood = scene_data['location'].get('mood', '')
                    location_description = scene_data['location'].get('description', '')
                # Check for tension in scene data
                if 'tension' in scene_data:
                    tension_level = scene_data.get('tension', 0)
                elif 'tension_level' in scene_data:
                    tension_level = scene_data.get('tension_level', 0)
            elif event.event_type == 'location_change':
                location_name = event.data.get('location_name', location_name)
                location_mood = event.data.get('mood', location_mood)
                location_description = event.data.get('description', location_description)
            elif event.event_type == 'tension_change':
                tension_level = event.data.get('new_tension', tension_level)
        
        # Only include real player name if we have it
        characters = []
        if player_name and player_name != 'Unknown':
            characters.append({"name": player_name, "is_player": True, "position": "present", "description": ""})
        
        # Look for character events to build character list
        npc_chars = {}
        for event in events:
            if event.event_type == 'character_introduced':
                char_name = event.data.get('name')
                if char_name and char_name != player_name:
                    npc_chars[char_name] = {
                        "name": char_name,
                        "is_player": False,
                        "position": event.data.get('position', 'present'),
                        "description": event.data.get('description', '')
                    }
        
        # If no NPCs found in events, use defaults for fantasy tavern
        if not npc_chars and stats.get('world_type') == 'fantasy_tavern':
            npc_chars = {
                "Martha": {"name": "Martha", "is_player": False, "position": "behind the bar", "description": "The innkeeper"},
                "Hooded Stranger": {"name": "Hooded Stranger", "is_player": False, "position": "in the corner", "description": "A mysterious figure"}
            }
        
        characters.extend(npc_chars.values())
        
        # Process events to build history
        for event in events:
            if event.event_type == 'action':
                # This is a player action
                action_text = event.data.get('action', '')
                # Look for the next narrative/dialogue events for the response
                continue  # We'll match them up below
                
        # Actually, let's rebuild history more carefully
        history = []
        i = 0
        while i < len(events):
            event = events[i]
            if event.event_type == 'action':
                # Found an action, collect the response
                action_text = event.data.get('action', '')
                narrative_text = ''
                dialogue_list = []
                
                # Look at next events for narrative and dialogue
                j = i + 1
                while j < len(events) and events[j].event_type in ['narrative', 'dialogue']:
                    if events[j].event_type == 'narrative':
                        narrative_text = events[j].data.get('text', '')
                    elif events[j].event_type == 'dialogue':
                        # Check different field names for dialogue text
                        dialogue_text = events[j].data.get('dialogue') or events[j].data.get('text', '')
                        dialogue_list.append({
                            "speaker": events[j].data.get('speaker', 'Unknown'),
                            "dialogue": dialogue_text,
                            "tone": events[j].data.get('tone', '')
                        })
                    j += 1
                
                history.append({
                    "id": f"event-{i}-{int(event.timestamp.timestamp() * 1000) if hasattr(event, 'timestamp') else i}",
                    "action": action_text,
                    "narrative": narrative_text,
                    "dialogue": dialogue_list
                })
                i = j
            else:
                i += 1
        
        # Create new session and add to active sessions
        new_session = GameSession(session_id)
        new_session.player_name = player_name
        new_session.player_description = stats.get('metadata', {}).get('description', '')
        
        # Initialize world based on world type
        from .services.world_builder import WorldTemplate, PRESET_WORLDS
        from backend.models import Character, Position
        
        world_type = stats.get('world_type', 'fantasy_tavern')
        if world_type in PRESET_WORLDS:
            template = WorldTemplate(template_data=PRESET_WORLDS[world_type])
            new_session.world = template.to_world_state()
        else:
            new_session.world = WorldState(f"resumed_{session_id}")
        
        # Add the player character to the world
        player_char = Character(
            name=player_name,
            description=new_session.player_description
        )
        player_char.position = Position(x=0, y=0, z=0, location="The Dragon's Rest", posture="standing")
        # Mark as player using an attribute
        player_char.is_player = True
        new_session.world.add_character(player_char)
        new_session.world.player_character = player_char
        
        # Initialize director with proper config
        from .services.director import DirectorConfig
        new_session.director = StructuredDirector(
            world=new_session.world,
            config=DirectorConfig(
                content_rating="mature",
                enable_romance=True,
                narrative_style=["immersive", "character-driven", "atmospheric"]
            )
        )
        
        # IMPORTANT: Restore conversation history to the director
        if hasattr(new_session.director, 'conversation_history'):
            new_session.director.conversation_history = history
            print(f"✅ Restored {len(history)} history entries to director")
            print(f"   Characters in world: {list(new_session.world.characters.keys())}")
            if history:
                print(f"   First history has dialogue from: {[d['speaker'] for d in history[0].get('dialogue', [])]}")
        
        # CRITICAL: Set up the current scene with all characters!
        from backend.models import Scene
        
        # Get the first location or create a default one
        if new_session.world.locations:
            location = list(new_session.world.locations.values())[0]
        else:
            # Create default location based on world type
            from backend.models import Location
            if world_type == 'fantasy_tavern':
                location = Location(
                    name="The Dragon's Rest",
                    description="A cozy tavern with wooden beams and a roaring fireplace",
                    mood="welcoming"
                )
                new_session.world.add_location(location)
            else:
                location = Location(name="Unknown Location", description="")
                new_session.world.add_location(location)
        
        # Create scene with all characters present
        scene = Scene(
            scene_type="tavern" if world_type == 'fantasy_tavern' else "indoor",
            location=location
        )
        scene.tension_level = tension_level
        
        # Add ALL characters to the scene
        for char_name, char in new_session.world.characters.items():
            scene.add_character(char)
            print(f"   ✅ Added {char_name} to current scene")
        
        # Set the current scene on the director
        new_session.director.current_scene = scene
        print(f"   ✅ Set current scene with {len(scene.present_characters)} characters")
        
        # Store in active sessions for future use
        active_sessions[session_id] = new_session
        
        return {
            "success": True,
            "from_memory": False,
            "from_persistence": True,
            "player_name": player_name,
            "location": {
                "name": location_name,
                "mood": location_mood,
                "description": location_description
            },
            "characters": characters,
            "tension": tension_level,
            "history": history[-20:],  # Last 20 entries
            "last_narrative": history[-1]['narrative'] if history else "Your adventure continues...",
            "needs_reconstruction": True
        }
        
    except Exception as e:
        print(f"Error resuming session {session_id}: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/api/session/{session_id}/export")
async def export_session(session_id: str, format: str = "markdown") -> Dict:
    """Export session to file"""
    await persistence_service.initialize()
    
    if format == "markdown":
        path = await persistence_service.export_story(session_id)
        return {"success": True, "path": path}
    elif format == "jsonl":
        path = f"exports/{session_id}/debug.jsonl"
        await persistence_service.manager.export_debug_log(session_id, path)
        return {"success": True, "path": path}
    else:
        return {"success": False, "error": "Invalid format"}


@router.post("/api/session/{session_id}/load")
async def load_session(session_id: str) -> Dict:
    """Load a saved session"""
    await persistence_service.initialize()
    
    # Load from checkpoint
    world_state = await persistence_service.load_session(session_id)
    if world_state:
        # Recreate session
        session = GameSession(session_id)
        session.world = world_state
        active_sessions[session_id] = session
        return {"success": True, "message": "Session loaded from checkpoint"}
    else:
        return {"success": False, "error": "No checkpoint found"}


# Character management endpoints
@router.get("/api/characters")
async def get_characters() -> List[Dict]:
    """Get all saved player characters"""
    return character_manager.list_characters()


@router.get("/api/characters/recent")
async def get_recent_characters(limit: int = 5) -> List[Dict]:
    """Get recently used characters"""
    return character_manager.get_recent_characters(limit)


@router.post("/api/characters/save")
async def save_character(character: CharacterCreation) -> Dict:
    """Save a player character for reuse"""
    filepath = character_manager.save_character(
        name=character.name,
        description=character.description
    )
    return {"success": True, "filepath": filepath, "message": f"Character {character.name} saved"}


@router.get("/api/characters/{name}")
async def get_character(name: str) -> Dict:
    """Get a specific saved character"""
    char_data = character_manager.load_character(name)
    if char_data:
        return {"success": True, "character": char_data}
    else:
        raise HTTPException(status_code=404, detail="Character not found")


@router.delete("/api/characters/{name}")
async def delete_character(name: str) -> Dict:
    """Delete a saved character"""
    if character_manager.delete_character(name):
        return {"success": True, "message": f"Character {name} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Character not found")


@router.delete("/api/session/{session_id}")
async def delete_session(session_id: str) -> Dict:
    """Delete a saved session"""
    await persistence_service.initialize()
    
    # Remove from active sessions if present
    if session_id in active_sessions:
        del active_sessions[session_id]
    
    # Delete from database
    try:
        async with persistence_service.db_conn.cursor() as cursor:
            # Delete all events for this session
            await cursor.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
            # Delete the session record
            await cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            await persistence_service.db_conn.commit()
            
        return {"success": True, "message": f"Session {session_id} deleted"}
    except Exception as e:
        print(f"Error deleting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))