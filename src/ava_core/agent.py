#!/usr/bin/env python3
"""
AVA Core Agent Implementation
Local agentic AI optimized for NVIDIA RTX A2000 4GB VRAM

This module implements the main Agent class that orchestrates all agentic capabilities
including dialogue management, function calling, reasoning, and LLM interaction.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available. Install with: pip install ollama")

from .dialogue_manager import DialogueManager
from .function_calling import FunctionCaller
from .reasoning import ReasoningEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration class for AVA Agent."""
    model_name: str = "ava-agent:latest"
    max_history_length: int = 20
    temperature: float = 0.7
    max_tokens: int = 2048
    stream_response: bool = True
    enable_reasoning: bool = True
    enable_function_calling: bool = True
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    timeout: float = 30.0


class AgentError(Exception):
    """Custom exception for Agent-related errors."""
    pass


class Agent:
    """
    Core AVA Agent implementing agentic workflow orchestration.
    
    Handles the complete agentic loop:
    1. Perception (input processing)
    2. Understanding (context building)
    3. Planning (reasoning about actions)
    4. Action (function calling, tool use)
    5. Response (generating output)
    6. Learning (updating context/history)
    """
    
    def __init__(
        self, 
        config: Optional[AgentConfig] = None,
        dialogue_manager: Optional[DialogueManager] = None,
        function_caller: Optional[FunctionCaller] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        available_tools: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the AVA Agent.

        Args:
            config: Agent configuration settings
            dialogue_manager: Handles conversation flow and context
            function_caller: Manages tool/function execution
            reasoning_engine: Handles reasoning processes like CoT
            available_tools: Dictionary of available tools for function calling
        """
        self.config = config or AgentConfig()
        
        # Initialize components
        self.dialogue_manager = dialogue_manager or DialogueManager(
            max_history_length=self.config.max_history_length
        )
        
        # Initialize function caller with tools
        if available_tools is None:
            available_tools = self._get_default_tools()
        
        self.function_caller = function_caller or FunctionCaller(
            available_tools=available_tools
        )
        
        self.reasoning_engine = reasoning_engine or ReasoningEngine()
        
        # Initialize Ollama client if available
        self.ollama_client = None
        if OLLAMA_AVAILABLE:
            try:
                self.ollama_client = ollama.Client(
                    host=f"http://{self.config.ollama_host}:{self.config.ollama_port}"
                )
                # Test connection
                self.ollama_client.list()
                logger.info(f"Connected to Ollama at {self.config.ollama_host}:{self.config.ollama_port}")
            except Exception as e:
                logger.warning(f"Could not connect to Ollama: {e}")
                self.ollama_client = None
        
        logger.info(f"AVA Agent initialized with model: {self.config.model_name}")

    def _get_default_tools(self) -> Dict[str, Any]:
        """Get default tools for the agent."""
        from .function_calling import get_weather, Calculator
        
        return {
            "get_weather": get_weather,
            "calculator": Calculator().run,
        }

    async def process_input_async(self, user_input: str) -> str:
        """
        Asynchronously process user input through the complete agentic workflow.
        
        Args:
            user_input: The user's input message
            
        Returns:
            The agent's response
        """
        try:
            logger.info(f"Processing input: {user_input[:100]}...")
            
            # Step 1: Update dialogue context (Perception)
            self.dialogue_manager.add_message("user", user_input)
            
            # Step 2: Get current context (Understanding)
            context = self.dialogue_manager.get_context(include_system_prompt=True)
            
            # Step 3: Apply reasoning if enabled (Planning)
            if self.config.enable_reasoning:
                reasoning_prompt = self.reasoning_engine.apply_chain_of_thought(
                    user_input, context
                )
                if reasoning_prompt:
                    context = reasoning_prompt
            
            # Step 4: Query LLM with context (Initial Response Generation)
            raw_llm_output = await self._query_llm_async(context)
            
            # Step 5: Check for function calls (Action)
            if self.config.enable_function_calling and self.function_caller.needs_function_call(raw_llm_output):
                final_response = await self._handle_function_call_async(raw_llm_output)
            else:
                final_response = raw_llm_output
            
            # Step 6: Update dialogue with response (Learning)
            self.dialogue_manager.add_message("assistant", final_response)
            
            logger.info("Input processed successfully")
            return final_response
            
        except Exception as e:
            error_msg = f"Error processing input: {str(e)}"
            logger.error(error_msg)
            return f"I encountered an error: {error_msg}"

    def process_input(self, user_input: str) -> str:
        """
        Synchronous wrapper for processing user input.
        
        Args:
            user_input: The user's input message
            
        Returns:
            The agent's response
        """
        try:
            return asyncio.run(self.process_input_async(user_input))
        except Exception as e:
            logger.error(f"Error in synchronous processing: {e}")
            return f"I encountered an error: {str(e)}"

    async def _handle_function_call_async(self, llm_output: str) -> str:
        """Handle function calls asynchronously."""
        try:
            tool_name, tool_args = self.function_caller.parse_function_call(llm_output)
            
            if tool_name and tool_args is not None:
                # Execute the function
                tool_result = self.function_caller.execute_function(tool_name, tool_args)
                
                # Get context with tool result for final response
                context_with_result = self.dialogue_manager.get_context_with_tool_result(
                    tool_name, tool_result
                )
                
                # Generate final response incorporating tool result
                final_response = await self._query_llm_async(context_with_result)
                return final_response
            else:
                logger.warning("Could not parse function call from LLM output")
                return llm_output
                
        except Exception as e:
            logger.error(f"Error handling function call: {e}")
            return f"I encountered an error while using a tool: {str(e)}"

    async def _query_llm_async(self, context: Union[str, List[Dict]]) -> str:
        """
        Query the LLM asynchronously with the given context.
        
        Args:
            context: The conversation context or prompt
            
        Returns:
            The LLM's response
        """
        if not self.ollama_client:
            # Fallback to simulation if Ollama is not available
            return self._simulate_llm_response(context)
        
        try:
            # Prepare messages format for Ollama
            if isinstance(context, str):
                messages = [{"role": "user", "content": context}]
            else:
                messages = context
            
            # Query Ollama
            if self.config.stream_response:
                response_chunks = []
                async for chunk in self._stream_ollama_response(messages):
                    response_chunks.append(chunk)
                return "".join(response_chunks)
            else:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.ollama_client.chat,
                        model=self.config.model_name,
                        messages=messages,
                        options={
                            "temperature": self.config.temperature,
                            "num_predict": self.config.max_tokens,
                        }
                    ),
                    timeout=self.config.timeout
                )
                return response["message"]["content"]
                
        except asyncio.TimeoutError:
            logger.error(f"LLM query timed out after {self.config.timeout}s")
            return "I'm sorry, my response took too long. Please try again."
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return self._simulate_llm_response(context)

    async def _stream_ollama_response(self, messages: List[Dict]) -> str:
        """Stream response from Ollama."""
        try:
            stream = await asyncio.to_thread(
                self.ollama_client.chat,
                model=self.config.model_name,
                messages=messages,
                stream=True,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                }
            )
            
            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]
                    
        except Exception as e:
            logger.error(f"Error streaming from Ollama: {e}")
            yield "I encountered an error while generating my response."

    def _simulate_llm_response(self, context: Union[str, List[Dict]]) -> str:
        """Simulate LLM response when Ollama is not available."""
        context_str = str(context).lower()
        
        # Simple keyword-based responses for testing
        if "weather" in context_str:
            return "[TOOL_CALL: get_weather(location='London')] I'll check the weather for you."
        elif "calculate" in context_str or "math" in context_str:
            return "[TOOL_CALL: calculator(operation='add', num1=5, num2=3)] Let me calculate that for you."
        elif "hello" in context_str or "hi" in context_str:
            return "Hello! I'm AVA, your local AI assistant. How can I help you today?"
        elif "tool_result" in context_str:
            return "Based on the tool result, here's what I found for you."
        else:
            return "I understand your request. This is a simulated response since Ollama is not available."

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent and its components."""
        status = {
            "agent_initialized": True,
            "model_name": self.config.model_name,
            "ollama_available": OLLAMA_AVAILABLE,
            "ollama_connected": self.ollama_client is not None,
            "dialogue_history_length": len(self.dialogue_manager.history),
            "available_tools": list(self.function_caller.available_tools.keys()),
            "config": self.config.__dict__,
        }
        
        if self.ollama_client:
            try:
                models = self.ollama_client.list()
                status["available_models"] = [model["name"] for model in models["models"]]
            except Exception as e:
                status["model_list_error"] = str(e)
        
        return status

    def clear_history(self):
        """Clear the dialogue history."""
        self.dialogue_manager.clear_history()
        logger.info("Dialogue history cleared")

    def update_config(self, **kwargs):
        """Update agent configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")


# Testing and demonstration
if __name__ == "__main__":
    import asyncio
    
    async def test_agent():
        """Test the enhanced agent implementation."""
        print("=== AVA Core Agent Enhanced Test ===")
        
        # Create agent with default configuration
        agent = Agent()
        
        # Test status
        print("\nAgent Status:")
        status = agent.get_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Test interactions
        test_inputs = [
            "Hello AVA!",
            "What's the weather like in London?",
            "Can you calculate 15 + 27?",
            "Tell me about yourself",
        ]
        
        for user_input in test_inputs:
            print(f"\n{'='*50}")
            print(f"User: {user_input}")
            response = await agent.process_input_async(user_input)
            print(f"AVA: {response}")
        
        print(f"\n{'='*50}")
        print("Test completed successfully!")

    # Run the test
    asyncio.run(test_agent()) 