"""
Main FastAPI application for Narrative Engine
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from contextlib import asynccontextmanager
import json
import asyncio
from typing import Dict, List, Optional
import logging

from backend.models import WorldState, Character, Scene, Location, Position
from backend.models.character import ClothingState, EmotionalState
from backend.services.director import Director, DirectorConfig
from backend.services.llm_service import LLMFactory, LLMConfig, LLMProvider

# Import API routes
from backend.api_routes import router as game_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global world state (in production, use Redis)
world_states: Dict[str, WorldState] = {}
active_connections: Dict[str, WebSocket] = {}
directors: Dict[str, Director] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("Starting Narrative Engine...")
    # Initialize default world
    world = WorldState(world_id="default")
    world_states["default"] = world
    
    # Add default locations
    world.add_location(Location(
        name="Tavern",
        description="A cozy tavern with wooden tables and a crackling fireplace",
        props=["bar", "tables", "fireplace", "stairs"],
        mood="warm"
    ))
    
    world.add_location(Location(
        name="Hot Springs", 
        description="Natural hot springs surrounded by smooth rocks and rising steam",
        props=["pools", "rocks", "towels", "changing area"],
        mood="intimate"
    ))
    
    yield
    
    logger.info("Shutting down Narrative Engine...")


# Create FastAPI app
app = FastAPI(
    title="Narrative Engine",
    description="Next-generation platform for character-driven interactive storytelling",
    version="0.1.0",
    lifespan=lifespan
)

# Include game routes
app.include_router(game_router)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REST API ENDPOINTS
# ============================================================================

@app.get("/api")
async def api_info():
    """API info endpoint"""
    return {"message": "Narrative Engine API", "version": "0.1.0"}


@app.get("/api/world/{world_id}")
async def get_world_state(world_id: str = "default"):
    """Get current world state"""
    if world_id not in world_states:
        raise HTTPException(status_code=404, detail="World not found")
    
    world = world_states[world_id]
    return world.to_dict()


@app.post("/api/world/{world_id}/character")
async def create_character(world_id: str, character_data: dict):
    """Create new character"""
    if world_id not in world_states:
        raise HTTPException(status_code=404, detail="World not found")
    
    world = world_states[world_id]
    
    # Create character from data
    character = Character(
        name=character_data["name"],
        description=character_data.get("description", ""),
        age=character_data.get("age"),
        personality_traits=character_data.get("personality_traits", [])
    )
    
    # Add to world
    world.add_character(character)
    
    return {"message": f"Character {character.name} created", "character": character.to_dict()}


@app.put("/api/world/{world_id}/character/{char_name}/position")
async def update_character_position(world_id: str, char_name: str, position_data: dict):
    """Update character position"""
    if world_id not in world_states:
        raise HTTPException(status_code=404, detail="World not found")
    
    world = world_states[world_id]
    
    if char_name not in world.characters:
        raise HTTPException(status_code=404, detail="Character not found")
    
    # Update position
    new_position = Position(
        x=position_data.get("x", 0),
        y=position_data.get("y", 0),
        z=position_data.get("z", 0),
        location=position_data.get("location", "unknown"),
        posture=position_data.get("posture", "standing")
    )
    
    result = world.move_character(char_name, to_position=new_position)
    
    # Broadcast update to connected clients
    await broadcast_update(world_id, {
        "type": "position_update",
        "character": char_name,
        "position": new_position.to_dict()
    })
    
    return {"message": result, "position": new_position.to_dict()}


@app.put("/api/world/{world_id}/character/{char_name}/clothing")
async def update_character_clothing(world_id: str, char_name: str, clothing_data: dict):
    """Update character clothing"""
    if world_id not in world_states:
        raise HTTPException(status_code=404, detail="World not found")
    
    world = world_states[world_id]
    
    if char_name not in world.characters:
        raise HTTPException(status_code=404, detail="Character not found")
    
    character = world.characters[char_name]
    
    # Update clothing
    new_state = ClothingState(clothing_data.get("state", "fully dressed"))
    description = clothing_data.get("description", "")
    
    result = character.change_clothing(new_state, description)
    
    # Broadcast update
    await broadcast_update(world_id, {
        "type": "clothing_update",
        "character": char_name,
        "clothing": new_state.value,
        "description": description
    })
    
    return {"message": result, "clothing": new_state.value}


@app.post("/api/world/{world_id}/scene")
async def set_scene(world_id: str, scene_data: dict):
    """Set active scene"""
    if world_id not in world_states:
        raise HTTPException(status_code=404, detail="World not found")
    
    world = world_states[world_id]
    location_name = scene_data.get("location")
    
    if not location_name or location_name not in world.locations:
        raise HTTPException(status_code=404, detail="Location not found")
    
    # Set scene
    scene = world.set_scene(location_name)
    
    # Add characters if specified
    for char_name in scene_data.get("characters", []):
        if char_name in world.characters:
            scene.add_character(world.characters[char_name])
    
    # Apply scene template if specified
    template = scene_data.get("template")
    if template:
        scene_manager = SceneManager()
        scene_manager.apply_template(template, scene, world)
    
    return {"message": f"Scene set to {location_name}", "scene": scene.to_dict()}


@app.post("/api/world/{world_id}/checkpoint")
async def create_checkpoint(world_id: str, checkpoint_data: dict):
    """Create world checkpoint"""
    if world_id not in world_states:
        raise HTTPException(status_code=404, detail="World not found")
    
    world = world_states[world_id]
    name = checkpoint_data.get("name", f"checkpoint_{len(world.checkpoints)}")
    
    result = world.create_checkpoint(name)
    
    return {"message": result, "checkpoints": list(world.checkpoints.keys())}


@app.post("/api/world/{world_id}/checkpoint/{name}/restore")
async def restore_checkpoint(world_id: str, name: str):
    """Restore world checkpoint"""
    if world_id not in world_states:
        raise HTTPException(status_code=404, detail="World not found")
    
    world = world_states[world_id]
    
    if name not in world.checkpoints:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    
    result = world.restore_checkpoint(name)
    
    # Broadcast major update
    await broadcast_update(world_id, {
        "type": "checkpoint_restored",
        "checkpoint": name,
        "world_state": world.to_dict()
    })
    
    return {"message": result}


@app.post("/api/world/{world_id}/interact")
async def character_interaction(world_id: str, interaction_data: dict):
    """Handle character interaction"""
    if world_id not in world_states:
        raise HTTPException(status_code=404, detail="World not found")
    
    world = world_states[world_id]
    
    char1_name = interaction_data.get("character1")
    char2_name = interaction_data.get("character2")
    interaction_type = interaction_data.get("type", "talk")
    description = interaction_data.get("description", "")
    
    if char1_name not in world.characters or char2_name not in world.characters:
        raise HTTPException(status_code=404, detail="Character not found")
    
    char1 = world.characters[char1_name]
    char2 = world.characters[char2_name]
    
    # Process interaction
    result = char1.interact_with(char2, interaction_type, description)
    
    # Generate narrative response (simplified for now)
    narrative = f"{char1.name} and {char2.name} interact ({interaction_type})"
    
    # Broadcast narrative
    await broadcast_update(world_id, {
        "type": "interaction",
        "narrative": narrative,
        "participants": [char1_name, char2_name]
    })
    
    return {"message": result, "narrative": narrative}


# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================

@app.websocket("/ws/{world_id}/{client_id}")
async def websocket_endpoint(websocket: WebSocket, world_id: str, client_id: str):
    """WebSocket connection for real-time updates"""
    await websocket.accept()
    
    # Store connection
    connection_key = f"{world_id}:{client_id}"
    active_connections[connection_key] = websocket
    
    try:
        # Send initial world state
        if world_id in world_states:
            await websocket.send_json({
                "type": "world_state",
                "data": world_states[world_id].to_dict()
            })
        
        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_json()
            
            # Process player action
            if data.get("type") == "action":
                await process_player_action(world_id, client_id, data.get("action", ""))
            
    except WebSocketDisconnect:
        del active_connections[connection_key]
        logger.info(f"Client {client_id} disconnected from world {world_id}")


async def broadcast_update(world_id: str, update: dict):
    """Broadcast update to all connected clients in a world"""
    for key, websocket in active_connections.items():
        if key.startswith(f"{world_id}:"):
            try:
                await websocket.send_json(update)
            except:
                # Connection might be closed
                pass


async def process_player_action(world_id: str, client_id: str, action: str):
    """Process player action and generate narrative response"""
    if world_id not in world_states:
        return
    
    world = world_states[world_id]
    
    # Get or create director for this world
    if world_id not in directors:
        try:
            # Try to create LLM service from environment
            llm_service = LLMFactory.create_from_env()
        except Exception as e:
            logger.warning(f"Could not create LLM service: {e}")
            # Create mock service for testing
            llm_service = None
        
        directors[world_id] = Director(
            world=world,
            llm_service=llm_service,
            config=DirectorConfig(
                content_rating="mature",
                enable_romance=True,
                narrative_style=["immersive", "sensory", "character-driven"]
            )
        )
    
    director = directors[world_id]
    
    try:
        # Process the player action
        result = await director.process_player_action(client_id, action)
        
        # Broadcast narrative to all clients
        await broadcast_update(world_id, {
            "type": "narrative",
            "action": action,
            "response": result.get("narrative", "The world responds to your action."),
            "player": client_id,
            "npc_actions": result.get("npc_actions", []),
            "events": result.get("events", [])
        })
        
        # Update world state if needed
        if result.get("world_state"):
            await broadcast_update(world_id, {
                "type": "world_state",
                "data": result["world_state"]
            })
            
    except Exception as e:
        logger.error(f"Error processing action: {e}")
        # Fallback response
        await broadcast_update(world_id, {
            "type": "narrative",
            "action": action,
            "response": f"You {action}.",
            "player": client_id
        })


# ============================================================================
# SERVE FRONTEND
# ============================================================================

@app.get("/")
async def home():
    """Serve the new game interface"""
    import os
    game_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "game.html")
    if os.path.exists(game_path):
        return FileResponse(game_path)
    else:
        return FileResponse(os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html"))


@app.get("/sessions")
async def sessions_page():
    """Serve the session manager interface"""
    import os
    sessions_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "sessions.html")
    if os.path.exists(sessions_path):
        return FileResponse(sessions_path)
    else:
        return HTMLResponse(content="Sessions page not found", status_code=404)


@app.get("/memories")
async def memories_page():
    """Serve the NPC memory viewer interface"""
    import os
    memories_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "memories.html")
    if os.path.exists(memories_path):
        return FileResponse(memories_path)
    else:
        return HTMLResponse(content="Memory viewer page not found", status_code=404)

@app.get("/play", response_class=HTMLResponse)
async def play_interface():
    """Serve the main play interface"""
    import os
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "game.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    else:
        # Fallback to simple interface
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Narrative Engine - Fallback</title>
        </head>
        <body>
            <h1>Narrative Engine</h1>
            <p>Frontend file not found. Please check installation.</p>
        </body>
        </html>
        """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)