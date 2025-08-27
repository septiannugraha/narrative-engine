"""
NPC Memory and World Simulation Service

This service manages:
1. NPC memories of past interactions
2. Gossip and information spreading
3. World state progression (time, events, news)
4. Background simulation of the wider world
"""

import json
import random
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path


class MemoryType(Enum):
    """Types of memories NPCs can have"""
    DIRECT_INTERACTION = "direct"  # Direct conversation/action with player
    WITNESSED = "witnessed"  # Saw something happen
    HEARD_GOSSIP = "gossip"  # Heard from another NPC
    WORLD_EVENT = "event"  # General world event (news, weather, etc)


class MemoryImportance(Enum):
    """How important/memorable an event is"""
    TRIVIAL = 1  # Forgotten quickly
    MINOR = 2  # Remember for a few days
    NOTABLE = 3  # Remember for weeks
    SIGNIFICANT = 4  # Remember for months
    LIFE_CHANGING = 5  # Never forget


@dataclass
class Memory:
    """A single memory held by an NPC"""
    memory_id: str
    npc_name: str
    memory_type: MemoryType
    content: str
    importance: MemoryImportance
    timestamp: datetime
    location: str
    involved_characters: List[str] = field(default_factory=list)
    emotional_impact: str = "neutral"  # happy, sad, angry, fearful, etc
    tags: List[str] = field(default_factory=list)  # searchable tags
    decay_rate: float = 1.0  # How fast this memory fades (1.0 = normal)
    source_npc: Optional[str] = None  # Who told them (for gossip)
    
    def get_age_days(self) -> float:
        """Get how old this memory is in days"""
        return (datetime.now() - self.timestamp).total_seconds() / 86400
    
    def get_recall_strength(self) -> float:
        """Calculate how well this memory can be recalled (0-1)"""
        age_days = self.get_age_days()
        base_decay = {
            MemoryImportance.TRIVIAL: 1.0,  # Forget in 1 day
            MemoryImportance.MINOR: 7.0,  # Forget in a week
            MemoryImportance.NOTABLE: 30.0,  # Forget in a month
            MemoryImportance.SIGNIFICANT: 180.0,  # Forget in 6 months
            MemoryImportance.LIFE_CHANGING: 99999.0  # Never forget
        }
        decay_days = base_decay[self.importance] / self.decay_rate
        strength = max(0, 1 - (age_days / decay_days))
        return strength


@dataclass
class NPCProfile:
    """Profile for an NPC including personality traits that affect memory"""
    name: str
    memory_capacity: int = 50  # Max memories before forgetting
    gossip_tendency: float = 0.5  # 0-1, how likely to share gossip
    memory_retention: float = 1.0  # Multiplier for memory decay
    interests: List[str] = field(default_factory=list)  # Topics they remember better
    relationships: Dict[str, float] = field(default_factory=dict)  # name -> trust level
    current_location: str = ""
    daily_routine: Dict[str, str] = field(default_factory=dict)  # time -> location
    personality_traits: List[str] = field(default_factory=list)  # nosy, discrete, forgetful, etc


@dataclass
class WorldEvent:
    """An event happening in the wider world"""
    event_id: str
    event_type: str  # festival, news, weather, crime, arrival, departure, etc
    description: str
    location: str
    timestamp: datetime
    duration_hours: float = 0  # 0 for instant events
    affected_areas: List[str] = field(default_factory=list)
    importance: MemoryImportance = MemoryImportance.MINOR
    spread_rate: float = 1.0  # How fast news spreads (1.0 = normal)
    details: Dict[str, Any] = field(default_factory=dict)


class NPCMemoryService:
    """Service for managing NPC memories and world simulation"""
    
    def __init__(self, world_id: str):
        self.world_id = world_id
        self.npcs: Dict[str, NPCProfile] = {}
        self.memories: Dict[str, List[Memory]] = {}  # npc_name -> memories
        self.world_events: List[WorldEvent] = []
        self.current_time = datetime.now()
        self.world_clock_speed = 1.0  # Time multiplier (2.0 = 2x speed)
        self.gossip_network: Dict[str, Set[str]] = {}  # who talks to whom
        
        # World state tracking
        self.world_state = {
            "time_of_day": "evening",
            "day_of_week": "Seventhday",
            "season": "autumn",
            "weather": "rainy",
            "town_mood": "quiet",
            "ongoing_events": [],
            "recent_news": [],
            "ambient_activity": []  # Background things happening
        }
        
        # Initialize save directory
        self.save_dir = Path(f"world_state/{world_id}")
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def add_npc(self, profile: NPCProfile):
        """Register an NPC in the world"""
        self.npcs[profile.name] = profile
        if profile.name not in self.memories:
            self.memories[profile.name] = []
    
    def record_interaction(self, 
                          npc_name: str,
                          player_name: str,
                          action: str,
                          dialogue: List[Dict[str, str]],
                          location: str,
                          emotional_impact: str = "neutral",
                          importance: MemoryImportance = MemoryImportance.MINOR):
        """Record a direct interaction between NPC and player"""
        
        if npc_name not in self.npcs:
            # Auto-create NPC profile if doesn't exist
            self.add_npc(NPCProfile(name=npc_name))
        
        # Create memory content from dialogue
        content_parts = [f"{player_name} {action}"]
        for exchange in dialogue:
            speaker = exchange.get('speaker', 'Someone')
            text = exchange.get('dialogue', exchange.get('text', ''))
            content_parts.append(f"{speaker} said: \"{text}\"")
        
        memory = Memory(
            memory_id=f"{npc_name}_{datetime.now().timestamp()}",
            npc_name=npc_name,
            memory_type=MemoryType.DIRECT_INTERACTION,
            content=" ".join(content_parts),
            importance=importance,
            timestamp=self.current_time,
            location=location,
            involved_characters=[player_name],
            emotional_impact=emotional_impact,
            tags=self._extract_tags(action, dialogue)
        )
        
        self._add_memory(npc_name, memory)
        
        # Check if other NPCs witnessed this
        self._propagate_witnessed_event(memory, exclude=[npc_name, player_name])
    
    def record_witnessed_event(self,
                              observer_npc: str,
                              event_description: str,
                              involved_characters: List[str],
                              location: str,
                              importance: MemoryImportance = MemoryImportance.MINOR):
        """Record something an NPC witnessed"""
        
        if observer_npc not in self.npcs:
            self.add_npc(NPCProfile(name=observer_npc))
        
        memory = Memory(
            memory_id=f"{observer_npc}_witness_{datetime.now().timestamp()}",
            npc_name=observer_npc,
            memory_type=MemoryType.WITNESSED,
            content=event_description,
            importance=importance,
            timestamp=self.current_time,
            location=location,
            involved_characters=involved_characters,
            tags=self._extract_tags(event_description, [])
        )
        
        self._add_memory(observer_npc, memory)
    
    def spread_gossip(self, source_npc: str, target_npc: str, memory: Memory) -> bool:
        """Attempt to spread gossip from one NPC to another"""
        
        if source_npc not in self.npcs or target_npc not in self.npcs:
            return False
        
        source = self.npcs[source_npc]
        target = self.npcs[target_npc]
        
        # Check if they would share (based on relationship and personality)
        trust_level = source.relationships.get(target_npc, 0.3)
        share_chance = source.gossip_tendency * trust_level
        
        if random.random() > share_chance:
            return False
        
        # Create gossip memory for target
        gossip_memory = Memory(
            memory_id=f"{target_npc}_gossip_{datetime.now().timestamp()}",
            npc_name=target_npc,
            memory_type=MemoryType.HEARD_GOSSIP,
            content=f"{source_npc} told me: {memory.content}",
            importance=MemoryImportance(max(1, memory.importance.value - 1)),  # Gossip loses importance
            timestamp=self.current_time,
            location=target.current_location,
            involved_characters=memory.involved_characters,
            emotional_impact=memory.emotional_impact,
            tags=memory.tags,
            source_npc=source_npc
        )
        
        self._add_memory(target_npc, gossip_memory)
        return True
    
    def simulate_time_passage(self, hours: float):
        """Simulate the passage of time in the world"""
        
        self.current_time += timedelta(hours=hours * self.world_clock_speed)
        
        # Update time of day
        hour = self.current_time.hour
        if 5 <= hour < 12:
            self.world_state["time_of_day"] = "morning"
        elif 12 <= hour < 17:
            self.world_state["time_of_day"] = "afternoon"
        elif 17 <= hour < 21:
            self.world_state["time_of_day"] = "evening"
        else:
            self.world_state["time_of_day"] = "night"
        
        # Move NPCs according to routines
        self._update_npc_locations()
        
        # Random chance of world events
        if random.random() < 0.1 * hours:  # 10% chance per hour
            self._generate_random_event()
        
        # Spread existing gossip
        self._propagate_gossip()
        
        # Clean up old memories
        self._decay_memories()
    
    def _update_npc_locations(self):
        """Update NPC locations based on time and routines"""
        time_key = self.world_state["time_of_day"]
        
        for npc_name, profile in self.npcs.items():
            if time_key in profile.daily_routine:
                profile.current_location = profile.daily_routine[time_key]
    
    def _generate_random_event(self):
        """Generate a random world event"""
        
        event_types = [
            ("merchant_arrival", "A merchant caravan arrived in town", MemoryImportance.MINOR),
            ("weather_change", "The weather suddenly changed", MemoryImportance.TRIVIAL),
            ("minor_incident", "There was a commotion in the market", MemoryImportance.MINOR),
            ("festival_announcement", "A festival was announced for next week", MemoryImportance.NOTABLE),
            ("mysterious_stranger", "A mysterious stranger was seen", MemoryImportance.NOTABLE),
        ]
        
        event_type, description, importance = random.choice(event_types)
        
        event = WorldEvent(
            event_id=f"event_{datetime.now().timestamp()}",
            event_type=event_type,
            description=description,
            location="town_square",
            timestamp=self.current_time,
            importance=importance,
            affected_areas=["town_square", "market", "tavern"]
        )
        
        self.world_events.append(event)
        self.world_state["recent_news"].append(description)
        
        # NPCs in affected areas learn about it
        for npc_name, profile in self.npcs.items():
            if profile.current_location in event.affected_areas:
                memory = Memory(
                    memory_id=f"{npc_name}_event_{event.event_id}",
                    npc_name=npc_name,
                    memory_type=MemoryType.WORLD_EVENT,
                    content=description,
                    importance=importance,
                    timestamp=self.current_time,
                    location=profile.current_location,
                    tags=[event_type]
                )
                self._add_memory(npc_name, memory)
    
    def _propagate_gossip(self):
        """Simulate gossip spreading between NPCs"""
        
        # Group NPCs by location
        location_groups = {}
        for npc_name, profile in self.npcs.items():
            loc = profile.current_location
            if loc not in location_groups:
                location_groups[loc] = []
            location_groups[loc].append(npc_name)
        
        # NPCs in same location might share gossip
        for location, npcs in location_groups.items():
            if len(npcs) < 2:
                continue
            
            # Random pairs might gossip
            for _ in range(min(3, len(npcs) // 2)):
                if len(npcs) >= 2:
                    source, target = random.sample(npcs, 2)
                    
                    # Find a recent memory to share
                    if source in self.memories:
                        recent_memories = [m for m in self.memories[source] 
                                         if m.get_recall_strength() > 0.5 
                                         and m.memory_type != MemoryType.HEARD_GOSSIP]
                        if recent_memories:
                            memory = random.choice(recent_memories)
                            self.spread_gossip(source, target, memory)
    
    def _propagate_witnessed_event(self, event_memory: Memory, exclude: List[str]):
        """Check if other NPCs in the location witnessed an event"""
        
        witnesses = []
        for npc_name, profile in self.npcs.items():
            if npc_name not in exclude and profile.current_location == event_memory.location:
                # They were there, might have seen it
                if random.random() < 0.7:  # 70% chance to notice
                    witnesses.append(npc_name)
        
        for witness in witnesses:
            self.record_witnessed_event(
                witness,
                f"Witnessed: {event_memory.content}",
                event_memory.involved_characters,
                event_memory.location,
                MemoryImportance(max(1, event_memory.importance.value - 1))
            )
    
    def _add_memory(self, npc_name: str, memory: Memory):
        """Add a memory to an NPC, managing capacity"""
        
        if npc_name not in self.memories:
            self.memories[npc_name] = []
        
        memories = self.memories[npc_name]
        memories.append(memory)
        
        # Manage capacity - forget least important/oldest
        profile = self.npcs.get(npc_name)
        if profile and len(memories) > profile.memory_capacity:
            # Sort by importance and recall strength
            memories.sort(key=lambda m: (m.importance.value * m.get_recall_strength()), reverse=True)
            self.memories[npc_name] = memories[:profile.memory_capacity]
    
    def _decay_memories(self):
        """Remove memories that have decayed too much"""
        
        for npc_name in list(self.memories.keys()):
            self.memories[npc_name] = [
                m for m in self.memories[npc_name]
                if m.get_recall_strength() > 0.05  # Keep if > 5% strength
            ]
    
    def _extract_tags(self, text: str, dialogue: List[Dict]) -> List[str]:
        """Extract searchable tags from text and dialogue"""
        
        tags = []
        keywords = ["fight", "kiss", "gold", "quest", "danger", "love", "death", "magic", "steal", "help"]
        
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                tags.append(keyword)
        
        for exchange in dialogue:
            dialogue_text = exchange.get('dialogue', '').lower()
            for keyword in keywords:
                if keyword in dialogue_text and keyword not in tags:
                    tags.append(keyword)
        
        return tags
    
    def get_npc_memories(self, npc_name: str, 
                         min_importance: Optional[MemoryImportance] = None,
                         about_character: Optional[str] = None,
                         memory_type: Optional[MemoryType] = None,
                         max_age_days: Optional[float] = None) -> List[Memory]:
        """Get filtered memories for an NPC"""
        
        if npc_name not in self.memories:
            return []
        
        memories = self.memories[npc_name]
        
        # Apply filters
        if min_importance:
            memories = [m for m in memories if m.importance.value >= min_importance.value]
        
        if about_character:
            memories = [m for m in memories if about_character in m.involved_characters]
        
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]
        
        if max_age_days:
            memories = [m for m in memories if m.get_age_days() <= max_age_days]
        
        # Sort by recall strength
        memories.sort(key=lambda m: m.get_recall_strength(), reverse=True)
        
        return memories
    
    def get_npc_context(self, npc_name: str, player_name: str) -> Dict[str, Any]:
        """Get full context for an NPC including memories and current state"""
        
        profile = self.npcs.get(npc_name)
        if not profile:
            return {}
        
        # Get relevant memories
        player_memories = self.get_npc_memories(
            npc_name,
            about_character=player_name,
            max_age_days=30  # Last month
        )
        
        recent_events = self.get_npc_memories(
            npc_name,
            memory_type=MemoryType.WORLD_EVENT,
            max_age_days=7  # Last week
        )
        
        recent_gossip = self.get_npc_memories(
            npc_name,
            memory_type=MemoryType.HEARD_GOSSIP,
            max_age_days=3  # Last 3 days
        )
        
        return {
            "profile": asdict(profile),
            "memories_of_player": [
                {
                    "content": m.content,
                    "when": f"{m.get_age_days():.1f} days ago",
                    "importance": m.importance.name,
                    "emotion": m.emotional_impact
                }
                for m in player_memories[:5]  # Top 5 memories
            ],
            "recent_events": [m.content for m in recent_events[:3]],
            "recent_gossip": [m.content for m in recent_gossip[:3]],
            "current_location": profile.current_location,
            "relationship_status": profile.relationships.get(player_name, 0.0),
            "world_state": self.world_state
        }
    
    def save_state(self):
        """Save the entire memory state to disk"""
        
        state = {
            "world_id": self.world_id,
            "current_time": self.current_time.isoformat(),
            "world_state": self.world_state,
            "npcs": {name: asdict(prof) for name, prof in self.npcs.items()},
            "memories": {
                npc: [asdict(m) for m in mems]
                for npc, mems in self.memories.items()
            },
            "world_events": [asdict(e) for e in self.world_events[-100:]],  # Keep last 100 events
            "gossip_network": {k: list(v) for k, v in self.gossip_network.items()}
        }
        
        save_path = self.save_dir / "npc_memory_state.json"
        with open(save_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self):
        """Load memory state from disk"""
        
        save_path = self.save_dir / "npc_memory_state.json"
        if not save_path.exists():
            return False
        
        with open(save_path, 'r') as f:
            state = json.load(f)
        
        self.world_id = state["world_id"]
        self.current_time = datetime.fromisoformat(state["current_time"])
        self.world_state = state["world_state"]
        
        # Reconstruct NPCs
        self.npcs = {}
        for name, prof_dict in state["npcs"].items():
            self.npcs[name] = NPCProfile(**prof_dict)
        
        # Reconstruct memories
        self.memories = {}
        for npc, mem_list in state["memories"].items():
            self.memories[npc] = []
            for mem_dict in mem_list:
                # Convert string enums back
                mem_dict["memory_type"] = MemoryType[mem_dict["memory_type"]]
                mem_dict["importance"] = MemoryImportance[mem_dict["importance"]]
                mem_dict["timestamp"] = datetime.fromisoformat(mem_dict["timestamp"])
                self.memories[npc].append(Memory(**mem_dict))
        
        # Reconstruct world events
        self.world_events = []
        for event_dict in state.get("world_events", []):
            event_dict["importance"] = MemoryImportance[event_dict["importance"]]
            event_dict["timestamp"] = datetime.fromisoformat(event_dict["timestamp"])
            self.world_events.append(WorldEvent(**event_dict))
        
        # Reconstruct gossip network
        self.gossip_network = {k: set(v) for k, v in state.get("gossip_network", {}).items()}
        
        return True


# Singleton instance management
_memory_services: Dict[str, NPCMemoryService] = {}


def get_memory_service(world_id: str) -> NPCMemoryService:
    """Get or create a memory service for a world"""
    if world_id not in _memory_services:
        _memory_services[world_id] = NPCMemoryService(world_id)
        # Try to load existing state
        _memory_services[world_id].load_state()
    return _memory_services[world_id]