"""
Simple Session Manager - Lightweight memory management
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages game sessions with automatic cleanup"""
    
    def __init__(self, max_sessions: int = 100, session_ttl_hours: int = 24):
        self.sessions: Dict[str, any] = {}
        self.session_metadata: Dict[str, dict] = {}
        self.max_sessions = max_sessions
        self.session_ttl = timedelta(hours=session_ttl_hours)
        self.cleanup_task = None
        
    async def start_cleanup_task(self):
        """Start the background cleanup task"""
        if not self.cleanup_task:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info(f"Session cleanup task started (TTL: {self.session_ttl})")
    
    async def stop_cleanup_task(self):
        """Stop the background cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            self.cleanup_task = None
    
    async def _cleanup_loop(self):
        """Background task that cleans up old sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self.cleanup_old_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def cleanup_old_sessions(self):
        """Remove sessions older than TTL"""
        now = datetime.now()
        sessions_to_remove = []
        
        for session_id, metadata in self.session_metadata.items():
            last_activity = metadata.get('last_activity', metadata.get('created_at'))
            if now - last_activity > self.session_ttl:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            await self.remove_session(session_id)
            logger.info(f"Cleaned up old session: {session_id}")
        
        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
    
    def add_session(self, session_id: str, session: any) -> bool:
        """Add a new session"""
        # Clean up if we're at max capacity
        if len(self.sessions) >= self.max_sessions:
            # Remove oldest session
            oldest_id = min(
                self.session_metadata.keys(),
                key=lambda k: self.session_metadata[k].get('last_activity', datetime.min)
            )
            asyncio.create_task(self.remove_session(oldest_id))
            logger.warning(f"Max sessions reached, removing oldest: {oldest_id}")
        
        self.sessions[session_id] = session
        self.session_metadata[session_id] = {
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'player_name': getattr(session, 'player_name', 'Unknown')
        }
        logger.info(f"Added session: {session_id}")
        return True
    
    def get_session(self, session_id: str) -> Optional[any]:
        """Get a session and update last activity"""
        if session_id in self.sessions:
            self.session_metadata[session_id]['last_activity'] = datetime.now()
            return self.sessions[session_id]
        return None
    
    async def remove_session(self, session_id: str) -> bool:
        """Remove a session and clean up its resources"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Clean up session resources
            if hasattr(session, 'cleanup'):
                try:
                    await session.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up session {session_id}: {e}")
            
            # Save memory state if applicable
            if hasattr(session, 'director') and hasattr(session.director, 'memory_service'):
                try:
                    session.director.memory_service.save_state()
                except Exception as e:
                    logger.error(f"Error saving memory state for {session_id}: {e}")
            
            del self.sessions[session_id]
            del self.session_metadata[session_id]
            logger.info(f"Removed session: {session_id}")
            return True
        return False
    
    def get_active_sessions(self) -> list:
        """Get list of active sessions with metadata"""
        result = []
        for session_id, metadata in self.session_metadata.items():
            result.append({
                'session_id': session_id,
                'player_name': metadata.get('player_name', 'Unknown'),
                'created_at': metadata['created_at'].isoformat(),
                'last_activity': metadata['last_activity'].isoformat(),
                'age_minutes': int((datetime.now() - metadata['created_at']).total_seconds() / 60)
            })
        return sorted(result, key=lambda x: x['last_activity'], reverse=True)
    
    def get_stats(self) -> dict:
        """Get session manager statistics"""
        now = datetime.now()
        active_count = len(self.sessions)
        
        if active_count > 0:
            oldest = min(
                self.session_metadata.values(),
                key=lambda m: m['created_at']
            )
            newest = max(
                self.session_metadata.values(),
                key=lambda m: m['created_at']
            )
            
            return {
                'active_sessions': active_count,
                'max_sessions': self.max_sessions,
                'oldest_session_age_hours': (now - oldest['created_at']).total_seconds() / 3600,
                'newest_session_age_minutes': (now - newest['created_at']).total_seconds() / 60,
                'ttl_hours': self.session_ttl.total_seconds() / 3600
            }
        
        return {
            'active_sessions': 0,
            'max_sessions': self.max_sessions,
            'ttl_hours': self.session_ttl.total_seconds() / 3600
        }

# Global session manager instance
session_manager = SessionManager(max_sessions=50, session_ttl_hours=12)