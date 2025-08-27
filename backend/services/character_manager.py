"""
Character Manager - Save and load player characters
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class CharacterManager:
    """Manages saved player characters for quick loading"""
    
    def __init__(self):
        self.save_dir = Path("saved_characters")
        self.save_dir.mkdir(exist_ok=True)
    
    def save_character(self, name: str, description: str, tags: List[str] = None) -> str:
        """Save a player character for reuse"""
        character_data = {
            "name": name,
            "description": description,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat()
        }
        
        # Use name as filename (sanitized)
        safe_name = "".join(c for c in name if c.isalnum() or c in "._- ")
        filepath = self.save_dir / f"{safe_name}.json"
        
        with open(filepath, 'w') as f:
            json.dump(character_data, f, indent=2)
        
        return str(filepath)
    
    def load_character(self, name: str) -> Optional[Dict]:
        """Load a saved character by name"""
        safe_name = "".join(c for c in name if c.isalnum() or c in "._- ")
        filepath = self.save_dir / f"{safe_name}.json"
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r') as f:
            character_data = json.load(f)
        
        # Update last used
        character_data["last_used"] = datetime.now().isoformat()
        with open(filepath, 'w') as f:
            json.dump(character_data, f, indent=2)
        
        return character_data
    
    def list_characters(self) -> List[Dict]:
        """List all saved characters"""
        characters = []
        
        for filepath in self.save_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    char_data = json.load(f)
                    char_data["filename"] = filepath.name
                    characters.append(char_data)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        # Sort by last used, most recent first
        characters.sort(key=lambda x: x.get("last_used", ""), reverse=True)
        
        return characters
    
    def delete_character(self, name: str) -> bool:
        """Delete a saved character"""
        safe_name = "".join(c for c in name if c.isalnum() or c in "._- ")
        filepath = self.save_dir / f"{safe_name}.json"
        
        if filepath.exists():
            filepath.unlink()
            return True
        return False
    
    def get_recent_characters(self, limit: int = 5) -> List[Dict]:
        """Get recently used characters"""
        characters = self.list_characters()
        return characters[:limit]


# Global instance
character_manager = CharacterManager()