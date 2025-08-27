"""
LLM Integration Service for narrative generation
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Provider imports (will be optional based on what's installed)
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    LOCAL = "local"  # For Ollama, LM Studio, etc.


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # For local models
    
    # Generation parameters
    temperature: float = 0.9
    max_tokens: int = 2000
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Custom settings per provider
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NarrativeContext:
    """Context for narrative generation"""
    # World state
    world_summary: str
    location_description: str
    time_of_day: str
    weather: str
    
    # Characters present
    characters: List[Dict[str, Any]]
    active_character: str  # Who is acting
    
    # Recent history
    recent_actions: List[str] = field(default_factory=list)
    recent_dialogue: List[str] = field(default_factory=list)
    
    # Scene information
    scene_type: str = "exploration"
    tension_level: int = 0
    objectives: List[str] = field(default_factory=list)
    
    # Special directives
    style_hints: List[str] = field(default_factory=list)
    content_rating: str = "mature"  # pg, teen, mature, explicit
    
    def to_prompt(self) -> str:
        """Convert context to prompt text"""
        prompt_parts = []
        
        # Location and time
        prompt_parts.append(f"LOCATION: {self.location_description}")
        prompt_parts.append(f"TIME: {self.time_of_day}, {self.weather}")
        
        # Characters
        if self.characters:
            char_descriptions = []
            for char in self.characters:
                desc = f"- {char['name']}"
                if char.get('description'):
                    desc += f" ({char['description']})"
                desc += f": {char.get('position', 'present')}"
                if char.get('clothing'):
                    desc += f", wearing {char['clothing']}"
                if char.get('emotional_state'):
                    desc += f", feeling {char['emotional_state']}"
                char_descriptions.append(desc)
            prompt_parts.append("CHARACTERS PRESENT:\n" + "\n".join(char_descriptions))
        
        # Recent context
        if self.recent_actions:
            prompt_parts.append("RECENT ACTIONS:\n" + "\n".join(self.recent_actions[-3:]))
        
        # Scene objectives
        if self.objectives:
            prompt_parts.append("SCENE OBJECTIVES:\n" + "\n".join(f"- {obj}" for obj in self.objectives))
        
        # Style hints
        if self.style_hints:
            prompt_parts.append("STYLE: " + ", ".join(self.style_hints))
        
        return "\n\n".join(prompt_parts)


class LLMService(ABC):
    """Abstract base for LLM services"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    async def generate_narrative(self, 
                                action: str, 
                                context: NarrativeContext) -> str:
        """Generate narrative response to an action"""
        pass
    
    @abstractmethod
    async def generate_npc_action(self, 
                                 npc_name: str,
                                 context: NarrativeContext) -> Dict[str, str]:
        """Generate autonomous NPC action"""
        pass
    
    def build_system_prompt(self, context: NarrativeContext) -> str:
        """Build the system prompt for narrative generation"""
        return f"""You are a narrative engine for an immersive text-based experience.

CONTENT RATING: {context.content_rating}
SCENE TYPE: {context.scene_type}

Your role:
- Generate vivid, sensory-rich narrative responses
- Maintain character consistency and spatial awareness
- React naturally to player actions
- Drive the story forward with interesting developments
- Respect the content rating while being authentic

Key principles:
- Show don't tell - use specific sensory details
- Characters have agency and act on their own desires
- The world feels alive and reactive
- Maintain tension and pacing appropriate to the scene
- Track positions, clothing states, and relationships accurately

Current tension level: {context.tension_level}/100"""


class OpenAIService(LLMService):
    """OpenAI GPT implementation"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not openai:
            raise ImportError("openai package not installed")
        
        self.client = openai.AsyncOpenAI(
            api_key=config.api_key or os.getenv("OPENAI_API_KEY")
        )
    
    async def generate_narrative(self, action: str, context: NarrativeContext) -> str:
        """Generate narrative using GPT"""
        messages = [
            {"role": "system", "content": self.build_system_prompt(context)},
            {"role": "user", "content": f"{context.to_prompt()}\n\nPLAYER ACTION: {action}\n\nGenerate a narrative response:"}
        ]
        
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty
        )
        
        return response.choices[0].message.content
    
    async def generate_npc_action(self, npc_name: str, context: NarrativeContext) -> Dict[str, str]:
        """Generate NPC action using GPT"""
        messages = [
            {"role": "system", "content": f"""You are simulating {npc_name}'s autonomous actions.
            
{self.build_system_prompt(context)}

Respond in JSON format:
{{
    "action": "what the character does",
    "dialogue": "what they say (if anything)",
    "inner_thought": "their private thoughts"
}}"""},
            {"role": "user", "content": f"{context.to_prompt()}\n\nWhat does {npc_name} do next?"}
        ]
        
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)


class AnthropicService(LLMService):
    """Anthropic Claude implementation"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not anthropic:
            raise ImportError("anthropic package not installed")
        
        self.client = anthropic.AsyncAnthropic(
            api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY")
        )
    
    async def generate_narrative(self, action: str, context: NarrativeContext) -> str:
        """Generate narrative using Claude"""
        prompt = f"""<context>
{context.to_prompt()}
</context>

<player_action>
{action}
</player_action>

Generate a vivid narrative response to the player's action. Focus on:
- Sensory details and atmosphere
- Character reactions and emotions
- Natural story progression
- Maintaining spatial and state consistency"""
        
        response = await self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=self.build_system_prompt(context),
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    async def generate_npc_action(self, npc_name: str, context: NarrativeContext) -> Dict[str, str]:
        """Generate NPC action using Claude"""
        prompt = f"""<context>
{context.to_prompt()}
</context>

You are simulating {npc_name}'s autonomous actions and thoughts.

Generate their next action in this exact JSON format:
{{
    "action": "specific physical action they take",
    "dialogue": "exact words they speak (or empty string if silent)",  
    "inner_thought": "their private thoughts about the situation"
}}

Make {npc_name} act according to their personality and the current situation."""
        
        response = await self.client.messages.create(
            model=self.config.model,
            max_tokens=500,
            temperature=self.config.temperature,
            system=self.build_system_prompt(context),
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse JSON from response
        text = response.content[0].text
        # Try to extract JSON if it's embedded in text
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(text)


class GeminiService(LLMService):
    """Google Gemini implementation with Gemini 2.0 thinking support"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not genai:
            raise ImportError("google-generativeai package not installed")
        
        genai.configure(api_key=config.api_key or os.getenv("GEMINI_API_KEY"))
        
        # Check if we're using a thinking model
        self.thinking_mode = "thinking" in config.model.lower()
        
        generation_config = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_output_tokens": config.max_tokens,
        }
        
        self.model = genai.GenerativeModel(
            config.model,
            generation_config=generation_config
        )
    
    async def generate_narrative(self, action: str, context: NarrativeContext) -> str:
        """Generate narrative using Gemini with optional thinking mode"""
        
        # Enhanced prompt for Gemini 2.0 with thinking
        if self.thinking_mode:
            prompt = f"""<thinking>
Consider the current scene carefully:
- Where exactly is everyone positioned?
- What is the emotional state and clothing of each character?
- What are the relationships and recent interactions?
- How would each character realistically react to this action?
- What sensory details would make this scene vivid?
</thinking>

{self.build_system_prompt(context)}

Current Situation:
{context.to_prompt()}

Player Action: {action}

Generate a richly detailed narrative response that:
1. Acknowledges the player's specific action with sensory detail
2. Shows immediate physical and emotional consequences
3. Includes authentic NPC reactions based on their personalities
4. Maintains perfect spatial and state consistency
5. Drives the story forward with tension or development

Focus on showing through specific details rather than telling."""
        else:
            prompt = f"""{self.build_system_prompt(context)}

Current Situation:
{context.to_prompt()}

Player Action: {action}

Generate a narrative response that:
1. Acknowledges the player's action
2. Describes the immediate consequences
3. Shows NPC reactions if relevant
4. Maintains world consistency
5. Drives the story forward"""
        
        response = await self.model.generate_content_async(prompt)
        
        # Extract the actual response (excluding thinking if present)
        text = response.text
        if self.thinking_mode and "<thinking>" in text:
            # Remove thinking tags if they leak into output
            import re
            text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL).strip()
        
        return text
    
    async def generate_npc_action(self, npc_name: str, context: NarrativeContext) -> Dict[str, str]:
        """Generate NPC action using Gemini"""
        prompt = f"""{self.build_system_prompt(context)}

Current Situation:
{context.to_prompt()}

You are determining {npc_name}'s next autonomous action.

Respond with a JSON object containing:
- action: what {npc_name} physically does
- dialogue: what {npc_name} says (can be empty string)
- inner_thought: {npc_name}'s private thoughts

Example format:
{{"action": "walks to the bar", "dialogue": "Another drink, please", "inner_thought": "I need to calm my nerves"}}"""
        
        response = await self.model.generate_content_async(prompt)
        
        # Parse JSON from response
        import re
        text = response.text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(text)


class LocalLLMService(LLMService):
    """Local model implementation (Ollama, LM Studio, etc.)"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"  # Ollama default
    
    async def generate_narrative(self, action: str, context: NarrativeContext) -> str:
        """Generate narrative using local model"""
        import aiohttp
        
        prompt = f"""{self.build_system_prompt(context)}

{context.to_prompt()}

Player Action: {action}

Response:"""
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "stream": False
                }
            ) as response:
                data = await response.json()
                return data.get("response", "")
    
    async def generate_npc_action(self, npc_name: str, context: NarrativeContext) -> Dict[str, str]:
        """Generate NPC action using local model"""
        import aiohttp
        
        prompt = f"""Generate {npc_name}'s next action as JSON.

{context.to_prompt()}

Respond only with JSON in this format:
{{"action": "...", "dialogue": "...", "inner_thought": "..."}}"""
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "temperature": self.config.temperature,
                    "max_tokens": 500,
                    "stream": False
                }
            ) as response:
                data = await response.json()
                return json.loads(data.get("response", "{}"))


class LLMFactory:
    """Factory for creating LLM services"""
    
    @staticmethod
    def create(config: LLMConfig) -> LLMService:
        """Create appropriate LLM service based on config"""
        if config.provider == LLMProvider.OPENAI:
            return OpenAIService(config)
        elif config.provider == LLMProvider.ANTHROPIC:
            return AnthropicService(config)
        elif config.provider == LLMProvider.GEMINI:
            return GeminiService(config)
        elif config.provider == LLMProvider.LOCAL:
            return LocalLLMService(config)
        else:
            raise ValueError(f"Unknown provider: {config.provider}")
    
    @staticmethod
    def create_from_env() -> LLMService:
        """Create LLM service from environment variables"""
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        
        if provider == "openai":
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.9"))
            )
        elif provider == "anthropic":
            config = LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229"),
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.9"))
            )
        elif provider == "gemini":
            config = LLMConfig(
                provider=LLMProvider.GEMINI,
                model=os.getenv("GEMINI_MODEL", "gemini-2.0-pro-exp"),
                api_key=os.getenv("GEMINI_API_KEY"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "1.2")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2400"))
            )
        elif provider == "local":
            config = LLMConfig(
                provider=LLMProvider.LOCAL,
                model=os.getenv("LOCAL_MODEL", "llama2"),
                base_url=os.getenv("LOCAL_LLM_URL", "http://localhost:11434"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.9"))
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
        
        return LLMFactory.create(config)


# Example usage
async def test_llm_service():
    """Test the LLM service"""
    
    # Create config
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-3.5-turbo",
        temperature=0.9
    )
    
    # Create service
    service = LLMFactory.create(config)
    
    # Create context
    context = NarrativeContext(
        world_summary="A fantasy realm",
        location_description="A cozy tavern with a crackling fireplace",
        time_of_day="Evening",
        weather="Rainy",
        characters=[
            {"name": "Alice", "position": "sitting at the bar"},
            {"name": "Bob", "position": "standing near the fireplace"}
        ],
        active_character="Player",
        scene_type="dialogue",
        style_hints=["warm", "atmospheric", "character-focused"]
    )
    
    # Generate narrative
    response = await service.generate_narrative(
        action="I walk over to Alice and greet her warmly",
        context=context
    )
    
    print("Narrative Response:")
    print(response)
    
    # Generate NPC action
    npc_action = await service.generate_npc_action("Alice", context)
    print("\nAlice's Action:")
    print(json.dumps(npc_action, indent=2))


if __name__ == "__main__":
    # Test the service
    asyncio.run(test_llm_service())