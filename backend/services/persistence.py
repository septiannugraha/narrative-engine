"""
Advanced Persistence System for Narrative Engine
Multi-layer storage with event sourcing and state snapshots
"""

import json
import sqlite3
import asyncio
import aiosqlite
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pickle
import lz4.frame
from enum import Enum

try:
    import redis.asyncio as redis
except ImportError:
    redis = None


class StorageLayer(Enum):
    """Different storage layers for different purposes"""
    HOT_CACHE = "redis"        # Active sessions, fast access
    EVENT_LOG = "sqlite"        # Event sourcing, append-only
    SNAPSHOTS = "sqlite"        # Periodic world state snapshots
    NARRATIVE = "markdown"      # Human-readable story export
    DEBUG_LOG = "jsonl"        # Debug/replay format


@dataclass
class GameEvent:
    """Immutable event in the game"""
    event_id: str
    session_id: str
    timestamp: datetime
    event_type: str  # "action", "dialogue", "state_change", "scene_change"
    actor: str  # Player name or system
    data: Dict[str, Any]
    metadata: Dict[str, Any]  # LLM response, tokens used, etc.


@dataclass
class StateSnapshot:
    """Complete world state at a point in time"""
    snapshot_id: str
    session_id: str
    timestamp: datetime
    event_id: str  # Last event before snapshot
    world_state: bytes  # Compressed pickle
    narrative_summary: str  # Human-readable summary
    stats: Dict[str, Any]  # Token count, turn number, etc.


class PersistenceManager:
    """
    Multi-layer persistence with:
    1. Redis for hot cache (optional)
    2. SQLite for events and snapshots
    3. File exports for debugging
    """
    
    def __init__(self, db_path: str = "narrative_engine.db", 
                 redis_url: Optional[str] = None):
        self.db_path = db_path
        self.redis_url = redis_url
        self.redis_client = None
        self.db_conn = None
        
    async def initialize(self):
        """Initialize storage layers"""
        # SQLite for events and snapshots
        self.db_conn = await aiosqlite.connect(self.db_path)
        await self._create_tables()
        
        # Redis for hot cache (optional)
        if redis and self.redis_url:
            try:
                self.redis_client = redis.from_url(self.redis_url)
                await self.redis_client.ping()
                print("✅ Redis cache connected")
            except Exception as e:
                print(f"⚠️ Redis not available, using SQLite only: {e}")
                self.redis_client = None
    
    async def _create_tables(self):
        """Create SQLite tables for persistence"""
        async with self.db_conn.cursor() as cursor:
            # Event sourcing table
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS game_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    data TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Create index separately
            await cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_time 
                ON game_events (session_id, timestamp)
            """)
            
            # Snapshot table for fast recovery
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    event_id TEXT NOT NULL,
                    world_state BLOB NOT NULL,
                    narrative_summary TEXT,
                    stats TEXT
                )
            """)
            
            # Create index for snapshots
            await cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_snap 
                ON state_snapshots (session_id, timestamp)
            """)
            
            # Session metadata
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    player_name TEXT,
                    world_type TEXT,
                    metadata TEXT,  -- JSON
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            await self.db_conn.commit()
    
    # ========================================================================
    # EVENT SOURCING
    # ========================================================================
    
    async def log_event(self, event: GameEvent) -> str:
        """Log an immutable game event"""
        async with self.db_conn.cursor() as cursor:
            await cursor.execute("""
                INSERT INTO game_events 
                (event_id, session_id, timestamp, event_type, actor, data, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.session_id,
                event.timestamp.timestamp(),
                event.event_type,
                event.actor,
                json.dumps(event.data, default=str),
                json.dumps(event.metadata, default=str) if event.metadata else None
            ))
            await self.db_conn.commit()
        
        # Also cache in Redis if available
        if self.redis_client:
            key = f"event:{event.session_id}:{event.event_id}"
            await self.redis_client.setex(
                key, 
                3600,  # 1 hour TTL
                json.dumps(asdict(event), default=str)
            )
        
        return event.event_id
    
    async def get_events(self, session_id: str, 
                        since: Optional[datetime] = None,
                        limit: int = 100) -> List[GameEvent]:
        """Retrieve events for replay or debugging"""
        query = """
            SELECT * FROM game_events 
            WHERE session_id = ?
        """
        params = [session_id]
        
        if since:
            query += " AND timestamp > ?"
            params.append(since.timestamp())
        
        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)
        
        async with self.db_conn.cursor() as cursor:
            await cursor.execute(query, params)
            rows = await cursor.fetchall()
        
        events = []
        for row in rows:
            events.append(GameEvent(
                event_id=row[0],
                session_id=row[1],
                timestamp=datetime.fromtimestamp(row[2]),
                event_type=row[3],
                actor=row[4],
                data=json.loads(row[5]),
                metadata=json.loads(row[6]) if row[6] else {}
            ))
        
        return events
    
    # ========================================================================
    # STATE SNAPSHOTS
    # ========================================================================
    
    async def save_snapshot(self, session_id: str, world_state: Any,
                           narrative_summary: str = "",
                           stats: Dict[str, Any] = None) -> str:
        """Save compressed world state snapshot"""
        import uuid
        
        # Get last event ID
        async with self.db_conn.cursor() as cursor:
            await cursor.execute("""
                SELECT event_id FROM game_events 
                WHERE session_id = ? 
                ORDER BY timestamp DESC LIMIT 1
            """, (session_id,))
            row = await cursor.fetchone()
            last_event_id = row[0] if row else "initial"
        
        # Compress world state
        pickled = pickle.dumps(world_state)
        compressed = lz4.frame.compress(pickled)
        
        snapshot = StateSnapshot(
            snapshot_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=datetime.now(),
            event_id=last_event_id,
            world_state=compressed,
            narrative_summary=narrative_summary,
            stats=stats or {}
        )
        
        async with self.db_conn.cursor() as cursor:
            await cursor.execute("""
                INSERT INTO state_snapshots
                (snapshot_id, session_id, timestamp, event_id, 
                 world_state, narrative_summary, stats)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.snapshot_id,
                snapshot.session_id,
                snapshot.timestamp.timestamp(),
                snapshot.event_id,
                snapshot.world_state,
                snapshot.narrative_summary,
                json.dumps(snapshot.stats)
            ))
            await self.db_conn.commit()
        
        # Cache in Redis for fast access
        if self.redis_client:
            key = f"snapshot:{session_id}:latest"
            await self.redis_client.setex(
                key,
                3600,
                compressed
            )
        
        return snapshot.snapshot_id
    
    async def load_snapshot(self, session_id: str, 
                           snapshot_id: Optional[str] = None) -> Any:
        """Load world state from snapshot"""
        # Try Redis first
        if self.redis_client and not snapshot_id:
            key = f"snapshot:{session_id}:latest"
            cached = await self.redis_client.get(key)
            if cached:
                decompressed = lz4.frame.decompress(cached)
                return pickle.loads(decompressed)
        
        # Load from SQLite
        query = """
            SELECT world_state FROM state_snapshots
            WHERE session_id = ?
        """
        params = [session_id]
        
        if snapshot_id:
            query += " AND snapshot_id = ?"
            params.append(snapshot_id)
        else:
            query += " ORDER BY timestamp DESC LIMIT 1"
        
        async with self.db_conn.cursor() as cursor:
            await cursor.execute(query, params)
            row = await cursor.fetchone()
        
        if row:
            decompressed = lz4.frame.decompress(row[0])
            return pickle.loads(decompressed)
        
        return None
    
    # ========================================================================
    # SESSION MANAGEMENT
    # ========================================================================
    
    async def create_session(self, session_id: str, player_name: str,
                           world_type: str, metadata: Dict = None) -> None:
        """Create new game session"""
        async with self.db_conn.cursor() as cursor:
            await cursor.execute("""
                INSERT INTO sessions 
                (session_id, created_at, updated_at, player_name, 
                 world_type, metadata, is_active)
                VALUES (?, ?, ?, ?, ?, ?, 1)
            """, (
                session_id,
                datetime.now().timestamp(),
                datetime.now().timestamp(),
                player_name,
                world_type,
                json.dumps(metadata or {})
            ))
            await self.db_conn.commit()
    
    async def get_active_sessions(self) -> List[Dict]:
        """Get all active sessions for dashboard"""
        async with self.db_conn.cursor() as cursor:
            await cursor.execute("""
                SELECT session_id, player_name, world_type, 
                       created_at, updated_at, metadata
                FROM sessions 
                WHERE is_active = 1
                ORDER BY updated_at DESC
            """)
            rows = await cursor.fetchall()
        
        sessions = []
        for row in rows:
            sessions.append({
                "session_id": row[0],
                "player_name": row[1],
                "world_type": row[2],
                "created_at": datetime.fromtimestamp(row[3]),
                "updated_at": datetime.fromtimestamp(row[4]),
                "metadata": json.loads(row[5])
            })
        
        return sessions
    
    # ========================================================================
    # EXPORT/DEBUG FEATURES
    # ========================================================================
    
    async def export_narrative(self, session_id: str, 
                              output_path: str) -> None:
        """Export human-readable narrative as Markdown"""
        events = await self.get_events(session_id, limit=10000)
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, "w") as f:
            f.write(f"# Game Session: {session_id}\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write("---\n\n")
            
            for event in events:
                if event.event_type == "action":
                    f.write(f"## {event.actor}\n")
                    f.write(f"*{event.timestamp.strftime('%H:%M:%S')}*\n\n")
                    f.write(f"> {event.data.get('action', '')}\n\n")
                    
                elif event.event_type == "narrative":
                    f.write(event.data.get('text', '') + "\n\n")
                    
                elif event.event_type == "dialogue":
                    speaker = event.data.get('speaker', 'Unknown')
                    dialogue = event.data.get('dialogue', '')
                    f.write(f"**{speaker}**: \"{dialogue}\"\n\n")
    
    async def export_debug_log(self, session_id: str, 
                              output_path: str) -> None:
        """Export JSONL format for debugging/replay"""
        events = await self.get_events(session_id, limit=10000)
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, "w") as f:
            for event in events:
                line = json.dumps(asdict(event), default=str)
                f.write(line + "\n")
    
    async def replay_session(self, session_id: str, 
                            until_event: Optional[str] = None):
        """Replay a session for debugging"""
        # Load initial snapshot or start fresh
        world_state = await self.load_snapshot(session_id)
        if not world_state:
            print("No snapshot found, replaying from beginning")
        
        # Get all events
        events = await self.get_events(session_id)
        
        replay_state = {
            "world": world_state,
            "events": []
        }
        
        for event in events:
            replay_state["events"].append(event)
            
            # Apply event to world state
            # (This would integrate with your Director)
            
            if until_event and event.event_id == until_event:
                break
        
        return replay_state
    
    async def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a session"""
        async with self.db_conn.cursor() as cursor:
            # Get session info including player name
            await cursor.execute("""
                SELECT player_name, world_type, created_at, updated_at, metadata
                FROM sessions
                WHERE session_id = ?
            """, (session_id,))
            session_row = await cursor.fetchone()
            
            if not session_row:
                return {}
            
            # Event counts by type
            await cursor.execute("""
                SELECT event_type, COUNT(*) 
                FROM game_events 
                WHERE session_id = ?
                GROUP BY event_type
            """, (session_id,))
            event_counts = dict(await cursor.fetchall())
            
            # Token usage over time
            await cursor.execute("""
                SELECT 
                    SUM(json_extract(metadata, '$.tokens_used')) as total_tokens,
                    COUNT(*) as total_events,
                    MIN(timestamp) as start_time,
                    MAX(timestamp) as end_time
                FROM game_events
                WHERE session_id = ?
            """, (session_id,))
            row = await cursor.fetchone()
            
            return {
                "player_name": session_row[0],
                "world_type": session_row[1],
                "created_at": session_row[2],
                "updated_at": session_row[3],
                "metadata": json.loads(session_row[4]) if session_row[4] else {},
                "event_counts": event_counts,
                "total_tokens": row[0] or 0,
                "total_events": row[1],
                "duration": (row[3] - row[2]) if row[2] and row[3] else 0,
                "start_time": datetime.fromtimestamp(row[2]) if row[2] else None,
                "end_time": datetime.fromtimestamp(row[3]) if row[3] else None
            }
    
    async def get_active_sessions(self) -> List[Dict]:
        """Get all active sessions"""
        async with self.db_conn.cursor() as cursor:
            await cursor.execute("""
                SELECT session_id, player_name, world_type, 
                       created_at, updated_at, metadata
                FROM sessions
                WHERE is_active = 1
                ORDER BY updated_at DESC
            """)
            rows = await cursor.fetchall()
        
        return [
            {
                "session_id": row[0],
                "player_name": row[1],
                "world_type": row[2],
                "created_at": datetime.fromisoformat(row[3]) if isinstance(row[3], str) else datetime.fromtimestamp(row[3]),
                "updated_at": datetime.fromisoformat(row[4]) if isinstance(row[4], str) else datetime.fromtimestamp(row[4]),
                "metadata": json.loads(row[5]) if row[5] else {}
            }
            for row in rows
        ]
    
    async def cleanup(self):
        """Clean up connections"""
        if self.db_conn:
            await self.db_conn.close()
        if self.redis_client:
            await self.redis_client.close()


# ============================================================================
# Persistence Service Integration
# ============================================================================

class PersistenceService:
    """High-level service for game persistence"""
    
    def __init__(self):
        self.manager = PersistenceManager()
        self._initialized = False
    
    async def initialize(self):
        """Initialize persistence layers"""
        if not self._initialized:
            await self.manager.initialize()
            self._initialized = True
    
    async def save_action(self, session_id: str, player_name: str,
                         action: str, response: Dict) -> None:
        """Save a player action and its response"""
        import uuid
        
        # Log the action event
        action_event = GameEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=datetime.now(),
            event_type="action",
            actor=player_name,
            data={"action": action},
            metadata={
                "tokens_used": response.get("tokens_used", 0),
                "model": response.get("model", "unknown")
            }
        )
        await self.manager.log_event(action_event)
        
        # Log narrative response
        if response.get("narrative"):
            narrative_event = GameEvent(
                event_id=str(uuid.uuid4()),
                session_id=session_id,
                timestamp=datetime.now(),
                event_type="narrative",
                actor="system",
                data={"text": response["narrative"]},
                metadata={}
            )
            await self.manager.log_event(narrative_event)
        
        # Log dialogue
        for dialogue in response.get("dialogue", []):
            dialogue_event = GameEvent(
                event_id=str(uuid.uuid4()),
                session_id=session_id,
                timestamp=datetime.now(),
                event_type="dialogue",
                actor=dialogue.get("speaker", "unknown"),
                data=dialogue,
                metadata={}
            )
            await self.manager.log_event(dialogue_event)
    
    async def save_checkpoint(self, session_id: str, 
                            world_state: Any,
                            summary: str = "") -> str:
        """Save a checkpoint of the world state"""
        # Get stats
        stats = await self.manager.get_session_stats(session_id)
        
        # Save snapshot
        snapshot_id = await self.manager.save_snapshot(
            session_id,
            world_state,
            narrative_summary=summary,
            stats=stats
        )
        
        return snapshot_id
    
    async def load_session(self, session_id: str) -> Any:
        """Load a session from the latest checkpoint"""
        return await self.manager.load_snapshot(session_id)
    
    async def export_story(self, session_id: str) -> str:
        """Export the story as Markdown"""
        output_path = f"exports/{session_id}/story.md"
        await self.manager.export_narrative(session_id, output_path)
        return output_path


# Singleton instance
persistence_service = PersistenceService()