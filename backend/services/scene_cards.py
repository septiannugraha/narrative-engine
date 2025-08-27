"""
Scene Cards System - Load complete scenarios like SillyTavern
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SceneCard:
    """A complete scenario card with world, characters, and opening"""
    
    def __init__(self, card_id: str, data: Dict):
        self.card_id = card_id
        self.name = data.get("name", "Unnamed Scene")
        self.description = data.get("description", "")
        self.tags = data.get("tags", [])
        self.author = data.get("author", "Anonymous")
        self.version = data.get("version", "1.0")
        
        # Core content
        self.opening_message = data.get("opening_message", "")
        self.scenario = data.get("scenario", "")
        self.world_info = data.get("world_info", {})
        self.characters = data.get("characters", [])
        
        # Settings
        self.content_rating = data.get("content_rating", "mature")
        self.recommended_personas = data.get("recommended_personas", [])
        self.romance_available = data.get("romance_available", True)
        
        # Lorebook entries
        self.lorebook = data.get("lorebook", {})
        
        # Alternate greetings/openings
        self.alternate_openings = data.get("alternate_openings", [])
        
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "card_id": self.card_id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "author": self.author,
            "version": self.version,
            "opening_message": self.opening_message,
            "scenario": self.scenario,
            "world_info": self.world_info,
            "characters": self.characters,
            "content_rating": self.content_rating,
            "recommended_personas": self.recommended_personas,
            "romance_available": self.romance_available,
            "lorebook": self.lorebook,
            "alternate_openings": self.alternate_openings
        }


class SceneCardManager:
    """Manages scene cards for quick scenario loading"""
    
    def __init__(self, cards_dir: str = "scene_cards"):
        self.cards_dir = Path(cards_dir)
        self.cards_dir.mkdir(exist_ok=True)
        self.cards: Dict[str, SceneCard] = {}
        self.load_all_cards()
        
        # Create default cards if none exist
        if len(self.cards) == 0:
            self._create_default_cards()
    
    def _create_default_cards(self):
        """Create default scene cards"""
        
        # Romantic Tavern Evening
        self.create_card({
            "card_id": "romantic_tavern",
            "name": "A Warm Evening at the Hearth",
            "description": "A cozy tavern encounter with flirtatious innkeeper Martha and other interesting characters. No mysterious cockblocks!",
            "tags": ["medieval", "tavern", "romance", "social"],
            "author": "System",
            "version": "1.0",
            "opening_message": "*The warm glow of the hearth greets you as you push open the heavy oak door of the Hearth & Hare Inn. The innkeeper, Martha, looks up from polishing a mug, her green eyes sparkling as they meet yours. Her auburn hair cascades over her shoulders, and her tavern dress shows just enough to make your pulse quicken.*\n\n\"Well now,\" *she purrs, leaning forward on the bar in a way that's definitely intentional,* \"aren't you a sight for sore eyes on this chilly evening. First drink's on the house for handsome strangers. What's your pleasure?\"\n\n*Her fingertips brush yours as she slides a menu across the bar, and you notice her wedding ring is conspicuously absent.*",
            "scenario": "You've arrived at the Hearth & Hare Inn seeking lodging for the night. The innkeeper Martha, recently widowed and clearly interested, manages the establishment with her shy cousin Lily. The atmosphere is warm and inviting, with no mysterious figures or supernatural nonsense - just good ale, warm food, and warmer company.",
            "world_info": {
                "location": "The Hearth & Hare Inn",
                "time": "Evening",
                "atmosphere": "Warm and inviting",
                "opportunities": ["Romance with Martha", "Flirt with shy Lily", "Befriend locals", "Rent a private room"]
            },
            "characters": [
                {
                    "name": "Martha",
                    "role": "Innkeeper",
                    "personality": "Flirtatious, warm, lonely",
                    "appearance": "Auburn hair, green eyes, curvy figure",
                    "interest_level": "Very interested"
                },
                {
                    "name": "Lily",
                    "role": "Barmaid",
                    "personality": "Shy, curious, innocent",
                    "appearance": "Petite blonde, blue eyes",
                    "interest_level": "Curious but nervous"
                }
            ],
            "content_rating": "mature",
            "romance_available": True,
            "recommended_personas": ["charming", "rogue", "adventurer"],
            "lorebook": {
                "Martha's room": "A cozy chamber above the inn with a large bed and fireplace",
                "The storage room": "A private space behind the bar, filled with wine barrels",
                "Lily's shyness": "She drops things when nervous but becomes bold with wine"
            },
            "alternate_openings": [
                "Late night arrival with Martha alone at the bar",
                "Morning encounter with Lily serving breakfast",
                "Festival night with both women dressed up"
            ]
        })
        
        # Beach Resort Testing
        self.create_card({
            "card_id": "beach_resort",
            "name": "Sunset at Paradise Cove",
            "description": "A tropical beach resort perfect for testing physical descriptions and various character interactions",
            "tags": ["modern", "beach", "romance", "casual"],
            "author": "System",
            "version": "1.0",
            "opening_message": "*The golden sunset paints Paradise Cove Beach in warm hues as you arrive at the resort. The sound of waves mingles with distant laughter and tropical music from the tiki bar. Several beautiful people in various states of beachwear catch your eye - from the surfing instructor waxing his board to the curvy bartender mixing colorful drinks.*\n\n*Sofia, the bartender, notices you approaching and flashes a brilliant smile, her golden-brown skin glistening in the sunset.* \"Welcome to paradise, gorgeous! First timer? Let me make you something special.\"",
            "scenario": "You've just arrived at Paradise Cove Beach Resort for a vacation. The beach is populated with attractive staff and guests, all in beach attire that leaves little to imagination. Perfect for testing physical descriptions and romantic encounters.",
            "world_info": {
                "location": "Paradise Cove Beach Resort",
                "time": "Golden hour sunset",
                "atmosphere": "Relaxed and playful",
                "opportunities": ["Beach volleyball", "Sunset yoga", "Tiki bar socializing", "Private beach walks"]
            },
            "characters": [
                {
                    "name": "Kai",
                    "role": "Surf Instructor",
                    "personality": "Laid-back, confident",
                    "appearance": "Athletic, sun-bleached hair, tanned"
                },
                {
                    "name": "Sofia",
                    "role": "Bartender",
                    "personality": "Bubbly, flirtatious",
                    "appearance": "Curvy, golden-brown skin, bikini top"
                },
                {
                    "name": "Jessica",
                    "role": "Yoga Instructor",
                    "personality": "Serene, flexible",
                    "appearance": "Athletic redhead with freckles"
                }
            ],
            "content_rating": "mature",
            "romance_available": True,
            "recommended_personas": ["athletic", "charming", "adventurous"],
            "lorebook": {
                "The tiki bar": "Serves strong drinks that lower inhibitions",
                "Beach cabanas": "Private changing areas that can be locked",
                "Bonfire parties": "Happen nightly with dancing and drinks"
            }
        })
        
        # Cyberpunk Bar
        self.create_card({
            "card_id": "cyberpunk_bar",
            "name": "Neon Nights at Chrome Coffin",
            "description": "A gritty cyberpunk dive bar where information and pleasure have their price",
            "tags": ["cyberpunk", "sci-fi", "noir", "adult"],
            "author": "System",
            "version": "1.0",
            "opening_message": "*The Chrome Coffin's neon sign flickers as you descend into the underground bar. Synthwave pulses through smoke-filled air, and chrome fixtures reflect the pink and blue lighting. The bartender's cybernetic arms move with precision as he pours glowing drinks.*\n\n*In the corner booth, a woman with neural interface ports behind her ears watches you with calculating eyes. Her form-fitting bodysuit leaves little to imagination.* \"New face,\" *she observes.* \"Looking for work, pleasure, or both?\"",
            "scenario": "Night City's Chrome Coffin - where runners, fixers, and those seeking escape congregate. Everyone has augmentations, everyone has secrets, and everything has a price.",
            "world_info": {
                "location": "The Chrome Coffin, Underground Night City",
                "time": "2:00 AM",
                "atmosphere": "Dangerous and electric",
                "opportunities": ["Information trading", "Cybernetic pleasures", "Underground connections"]
            },
            "characters": [
                {
                    "name": "Vex",
                    "role": "Fixer",
                    "personality": "Calculating, sensual",
                    "appearance": "Neural ports, bodysuit, dangerous curves"
                },
                {
                    "name": "Chrome Johnny",
                    "role": "Bartender",
                    "personality": "Gruff, observant",
                    "appearance": "Cybernetic arms, scarred face"
                }
            ],
            "content_rating": "explicit",
            "romance_available": True,
            "recommended_personas": ["hacker", "mercenary", "corpo"],
            "lorebook": {
                "Braindance booth": "Experience someone else's memories... intimately",
                "Synth drugs": "Enhance sensations and lower inhibitions",
                "Back rooms": "Private areas for various transactions"
            }
        })
    
    def create_card(self, data: Dict) -> SceneCard:
        """Create a new scene card"""
        card_id = data.get("card_id", f"card_{datetime.now().timestamp()}")
        card = SceneCard(card_id, data)
        self.cards[card_id] = card
        self.save_card(card)
        logger.info(f"Created scene card: {card.name}")
        return card
    
    def save_card(self, card: SceneCard):
        """Save a scene card to disk"""
        file_path = self.cards_dir / f"{card.card_id}.json"
        with open(file_path, 'w') as f:
            json.dump(card.to_dict(), f, indent=2)
    
    def load_card(self, card_id: str) -> Optional[SceneCard]:
        """Load a specific scene card"""
        file_path = self.cards_dir / f"{card_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                card = SceneCard(card_id, data)
                self.cards[card_id] = card
                return card
        return None
    
    def load_all_cards(self):
        """Load all scene cards"""
        for file_path in self.cards_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    card_id = data.get("card_id", file_path.stem)
                    card = SceneCard(card_id, data)
                    self.cards[card_id] = card
            except Exception as e:
                logger.error(f"Failed to load scene card from {file_path}: {e}")
    
    def list_cards(self, tags: List[str] = None) -> List[SceneCard]:
        """List all cards, optionally filtered by tags"""
        cards = list(self.cards.values())
        
        if tags:
            cards = [c for c in cards 
                    if any(tag in c.tags for tag in tags)]
        
        return cards
    
    def get_card(self, card_id: str) -> Optional[SceneCard]:
        """Get a specific scene card"""
        return self.cards.get(card_id)
    
    def import_sillytavern_card(self, file_path: str) -> Optional[SceneCard]:
        """Import a SillyTavern format card"""
        # This would parse SillyTavern's format and convert
        # Implementation depends on their exact format
        pass

# Global scene card manager
scene_card_manager = SceneCardManager()