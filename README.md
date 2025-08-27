# Narrative Engine

A next-generation narrative engine for immersive, AI-driven storytelling with persistent world state and multi-character agency.

## Features

### Core Capabilities
- **Persistent World State**: Characters, objects, and relationships maintain continuity across sessions
- **Multi-Character Agency**: Each NPC acts independently with their own goals and memories
- **Visual Debugging**: See character positions and relationships in real-time
- **Checkpoint System**: Save and rollback narrative states
- **Scene Templates**: Reusable configurations for common scenarios

### Technical Features
- FastAPI backend with WebSocket support for real-time updates
- SQLite for persistent storage with automatic migrations
- Redis caching for performance optimization
- Multi-LLM support (OpenAI, Anthropic, Gemini, local models)
- HTMX + Alpine.js minimal frontend (no heavy frameworks)
- Docker containerization for easy deployment

## Quick Start

### Prerequisites
- Python 3.9+
- Redis (optional, for caching)
- API key for your chosen LLM provider

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/narrative-engine.git
cd narrative-engine
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys and preferences
```

### Running the Engine

#### Development Mode
```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

#### Production with Docker
```bash
docker-compose up -d
```

### Basic Usage

Access the web interface at `http://localhost:8000`

#### Creating a Session
```python
from backend.services.session_manager import SessionManager

# Initialize session
session = SessionManager()
session_id = await session.create_session(
    world_template="cozy_tavern",
    player_name="Alice"
)
```

#### Adding Characters
```python
from backend.services.character_manager import CharacterManager

char_manager = CharacterManager()
await char_manager.add_character(
    session_id,
    name="Bartender",
    description="A gruff but kind tavern keeper",
    personality="Protective of regulars, suspicious of strangers"
)
```

## Architecture

### Backend Structure
```
backend/
├── models/          # Pydantic models for data validation
├── services/        # Core business logic
│   ├── director.py  # Main narrative orchestrator
│   ├── llm_service.py  # LLM provider abstraction
│   └── persistence.py  # Database operations
├── api_routes.py    # FastAPI route definitions
└── main.py          # Application entry point
```

### Key Components

- **Director Service**: Orchestrates narrative flow and character interactions
- **Character Manager**: Handles character state and memory
- **World Builder**: Manages spatial relationships and environment
- **Session Manager**: Tracks narrative sessions and checkpoints
- **LLM Service**: Abstraction layer for multiple AI providers

## Configuration

### LLM Providers

Configure your preferred provider in `.env`:

```env
# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4-turbo-preview

# Anthropic
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-key
ANTHROPIC_MODEL=claude-3-opus-20240229

# Gemini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-1.5-pro

# Local (Ollama)
LLM_PROVIDER=local
LOCAL_LLM_URL=http://localhost:11434
LOCAL_MODEL=llama2
```

### Content Settings

Adjust content rating and features:

```env
CONTENT_RATING=mature  # pg, teen, mature, explicit
ENABLE_ROMANCE=true
ENABLE_COMBAT=true
```

## Development

### Running Tests
```bash
# Basic functionality test
python test_simple.py

# Full integration test
python test_comprehensive.py

# Specific feature tests
python test_dialogue_generation.py
python test_continuity.py
```

### Creating Scene Templates

Add new templates to `backend/world_templates/`:

```python
{
    "name": "haunted_mansion",
    "description": "A decrepit Victorian mansion",
    "locations": {
        "foyer": {
            "description": "Grand entrance with cobwebs",
            "exits": {"north": "hallway", "up": "stairs"}
        }
    }
}
```

## API Documentation

Interactive API docs available at `http://localhost:8000/docs`

### Core Endpoints

- `POST /session/create` - Start new narrative session
- `POST /session/{session_id}/action` - Submit player action
- `GET /session/{session_id}/state` - Get current world state
- `POST /session/{session_id}/checkpoint` - Save narrative state
- `POST /session/{session_id}/rollback` - Restore previous state

## Contributing

Contributions are welcome! Please ensure:
1. Code follows existing patterns
2. Tests pass before submitting PR
3. Documentation is updated for new features

## License

MIT License - See LICENSE file for details

## Acknowledgments

Built with inspiration from:
- The Council narrative framework
- SillyTavern's character system
- Disco Elysium's internal dialogue mechanics

## Support

For issues and feature requests, please use the GitHub issue tracker.