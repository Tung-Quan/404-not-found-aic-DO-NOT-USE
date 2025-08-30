"""
AI Agent Manager - Tích hợp đầy đủ AI agents cho video search
Hỗ trợ OpenAI, Anthropic, và local models
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
import asyncio

# Core AI libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from langchain.llms import OpenAI as LangChainOpenAI
    from langchain.chat_models import ChatOpenAI, ChatAnthropic
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.agents import initialize_agent, Tool
    from langchain.agents import AgentType
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Local AI models
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    from transformers import BlipProcessor, BlipForConditionalGeneration
    LOCAL_MODELS_AVAILABLE = True
except ImportError:
    LOCAL_MODELS_AVAILABLE = False

@dataclass
class AgentConfig:
    """Configuration cho AI agent"""
    name: str
    provider: str  # 'openai', 'anthropic', 'local', 'langchain'
    model: str
    api_key: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    capabilities: List[str] = None

class AIAgentManager:
    """Manager cho tất cả AI agents"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.agents: Dict[str, Any] = {}
        self.configs: Dict[str, AgentConfig] = {}
        
        # Load environment variables
        self.api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        }
        
        # Initialize agents
        self._setup_default_agents()
        
        if config_path and config_path.exists():
            self._load_config(config_path)
    
    def _setup_default_agents(self):
        """Setup default AI agents"""
        
        # OpenAI GPT-4 Agent
        if OPENAI_AVAILABLE and self.api_keys["openai"]:
            self.configs["gpt4_vision"] = AgentConfig(
                name="GPT-4 Vision",
                provider="openai",
                model="gpt-4-vision-preview",
                api_key=self.api_keys["openai"],
                max_tokens=1000,
                temperature=0.3,
                system_prompt="You are an expert video frame analyzer. Describe what you see in detail.",
                capabilities=["vision", "text_generation", "analysis"]
            )
            
            self.configs["gpt4_text"] = AgentConfig(
                name="GPT-4 Text",
                provider="openai", 
                model="gpt-4",
                api_key=self.api_keys["openai"],
                max_tokens=1500,
                temperature=0.5,
                system_prompt="You are a helpful assistant for video search and analysis.",
                capabilities=["text_generation", "analysis", "summarization"]
            )
        
        # Anthropic Claude Agent
        if ANTHROPIC_AVAILABLE and self.api_keys["anthropic"]:
            self.configs["claude3"] = AgentConfig(
                name="Claude 3",
                provider="anthropic",
                model="claude-3-opus-20240229",
                api_key=self.api_keys["anthropic"],
                max_tokens=1000,
                temperature=0.4,
                system_prompt="You are Claude, an AI assistant specialized in video content analysis.",
                capabilities=["text_generation", "analysis", "reasoning"]
            )
        
        # Local BLIP Agent (for image captioning)
        if LOCAL_MODELS_AVAILABLE:
            self.configs["blip_local"] = AgentConfig(
                name="BLIP Local",
                provider="local",
                model="Salesforce/blip-image-captioning-base",
                capabilities=["image_captioning", "vision"]
            )
            
            self.configs["llama_local"] = AgentConfig(
                name="LLaMA Local",
                provider="local", 
                model="microsoft/DialoGPT-large",
                capabilities=["text_generation", "conversation"]
            )
    
    def initialize_agent(self, agent_name: str) -> bool:
        """Initialize specific agent"""
        if agent_name not in self.configs:
            self.logger.error(f"Agent {agent_name} not configured")
            return False
        
        config = self.configs[agent_name]
        
        try:
            if config.provider == "openai":
                self.agents[agent_name] = openai.OpenAI(api_key=config.api_key)
                
            elif config.provider == "anthropic":
                self.agents[agent_name] = anthropic.Anthropic(api_key=config.api_key)
                
            elif config.provider == "local":
                if "blip" in config.model.lower():
                    processor = BlipProcessor.from_pretrained(config.model)
                    model = BlipForConditionalGeneration.from_pretrained(config.model)
                    self.agents[agent_name] = {"processor": processor, "model": model}
                else:
                    tokenizer = AutoTokenizer.from_pretrained(config.model)
                    model = AutoModelForCausalLM.from_pretrained(config.model)
                    self.agents[agent_name] = {"tokenizer": tokenizer, "model": model}
            
            self.logger.info(f"✅ Initialized agent: {config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize {agent_name}: {e}")
            return False
    
    def analyze_frame(self, image_path: str, agent_name: str = "gpt4_vision") -> Optional[str]:
        """Analyze video frame using specified agent"""
        if agent_name not in self.agents:
            if not self.initialize_agent(agent_name):
                return None
        
        config = self.configs[agent_name]
        agent = self.agents[agent_name]
        
        try:
            if config.provider == "openai" and "vision" in config.capabilities:
                # GPT-4 Vision analysis
                with open(image_path, "rb") as image_file:
                    import base64
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                response = agent.chat.completions.create(
                    model=config.model,
                    messages=[
                        {
                            "role": "system",
                            "content": config.system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe this video frame in detail."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )
                return response.choices[0].message.content
                
            elif config.provider == "local" and "blip" in config.model.lower():
                # Local BLIP analysis
                from PIL import Image
                
                image = Image.open(image_path).convert('RGB')
                inputs = agent["processor"](image, return_tensors="pt")
                out = agent["model"].generate(**inputs, max_length=50)
                caption = agent["processor"].decode(out[0], skip_special_tokens=True)
                return caption
                
        except Exception as e:
            self.logger.error(f"Frame analysis failed: {e}")
            return None
    
    def generate_search_query(self, user_query: str, agent_name: str = "gpt4_text") -> Optional[str]:
        """Generate optimized search query"""
        if agent_name not in self.agents:
            if not self.initialize_agent(agent_name):
                return None
        
        config = self.configs[agent_name]
        agent = self.agents[agent_name]
        
        prompt = f"""
        Given this user query: "{user_query}"
        
        Generate an optimized search query for finding relevant video frames.
        Consider:
        - Visual elements that might appear in frames
        - Objects, people, actions, scenes
        - Color, composition, style
        - Context and meaning
        
        Return only the optimized query, nothing else.
        """
        
        try:
            if config.provider == "openai":
                response = agent.chat.completions.create(
                    model=config.model,
                    messages=[
                        {"role": "system", "content": config.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
                
            elif config.provider == "anthropic":
                response = agent.messages.create(
                    model=config.model,
                    max_tokens=200,
                    temperature=0.3,
                    system=config.system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
                
        except Exception as e:
            self.logger.error(f"Query generation failed: {e}")
            return user_query  # Fallback to original query
    
    def summarize_results(self, search_results: List[Dict], agent_name: str = "gpt4_text") -> Optional[str]:
        """Summarize search results"""
        if not search_results:
            return "No results found."
        
        if agent_name not in self.agents:
            if not self.initialize_agent(agent_name):
                return None
        
        # Format results for analysis
        results_text = "\n".join([
            f"Frame {i+1}: {result.get('description', 'No description')} (Score: {result.get('score', 'N/A')})"
            for i, result in enumerate(search_results[:5])
        ])
        
        prompt = f"""
        Analyze these video search results and provide a concise summary:
        
        {results_text}
        
        Summarize:
        - What type of content was found
        - Common themes or patterns
        - Key visual elements
        - Overall relevance to the search
        
        Keep the summary under 200 words.
        """
        
        config = self.configs[agent_name]
        agent = self.agents[agent_name]
        
        try:
            if config.provider == "openai":
                response = agent.chat.completions.create(
                    model=config.model,
                    messages=[
                        {"role": "system", "content": config.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.4
                )
                return response.choices[0].message.content
                
            elif config.provider == "anthropic":
                response = agent.messages.create(
                    model=config.model,
                    max_tokens=300,
                    temperature=0.4,
                    system=config.system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            return f"Found {len(search_results)} relevant frames."
    
    def get_available_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available agents"""
        return {
            name: {
                "name": config.name,
                "provider": config.provider,
                "model": config.model,
                "capabilities": config.capabilities,
                "initialized": name in self.agents
            }
            for name, config in self.configs.items()
        }
    
    def set_api_key(self, provider: str, api_key: str):
        """Set API key for provider"""
        self.api_keys[provider] = api_key
        # Update existing configs
        for config in self.configs.values():
            if config.provider == provider:
                config.api_key = api_key
    
    def _load_config(self, config_path: Path):
        """Load agent configuration from file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            for name, data in config_data.get("agents", {}).items():
                self.configs[name] = AgentConfig(**data)
                
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")

# Global instance
ai_agent_manager = AIAgentManager()

def test_ai_agents():
    """Test AI agent functionality"""
    print("🧪 Testing AI Agent Manager...")
    
    manager = AIAgentManager()
    
    # Show available agents
    agents = manager.get_available_agents()
    print(f"📋 Available agents: {len(agents)}")
    
    for name, info in agents.items():
        status = "✅" if info["initialized"] else "⚠️"
        print(f"  {status} {info['name']} ({info['provider']}) - {info['capabilities']}")
    
    # Test query generation (if available)
    optimized = manager.generate_search_query("show me a happy dog running in the park")
    if optimized:
        print(f"🔍 Query optimization: '{optimized}'")
    
    print("✅ AI Agent Manager test completed")

if __name__ == "__main__":
    test_ai_agents()
