#!/usr/bin/env python3
"""
Enhanced Dialogue Manager for AVA
Manages conversation history, context, and multi-turn interactions
for local agentic AI on RTX A2000 4GB VRAM constraints.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Enumeration for message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


@dataclass
class Message:
    """Represents a single message in the conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_call_id: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class ConversationState:
    """Tracks the current state of the conversation."""
    total_messages: int = 0
    user_messages: int = 0
    assistant_messages: int = 0
    tool_calls_made: int = 0
    errors_encountered: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class DialogueManager:
    """
    Enhanced Dialogue Manager for AVA's conversational capabilities.
    Manages conversation history, context formatting, and state tracking.
    """
    
    def __init__(
        self, 
        max_history_length: int = 20,
        system_prompt: Optional[str] = None,
        preserve_system: bool = True,
        auto_summarize: bool = False,
        summary_threshold: int = 50
    ):
        """
        Initialize the Dialogue Manager.
        
        Args:
            max_history_length: Maximum number of messages to keep in history
            system_prompt: Default system prompt for the conversation
            preserve_system: Whether to always keep system messages
            auto_summarize: Whether to automatically summarize old conversations
            summary_threshold: Number of messages before auto-summarization kicks in
        """
        self.max_history_length = max_history_length
        self.preserve_system = preserve_system
        self.auto_summarize = auto_summarize
        self.summary_threshold = summary_threshold
        
        self.messages: List[Message] = []
        self.state = ConversationState()
        
        # Set default system prompt
        self._default_system_prompt = system_prompt or self._get_default_system_prompt()
        
        # Initialize with system message if provided
        if self._default_system_prompt:
            self.add_message(
                role=MessageRole.SYSTEM,
                content=self._default_system_prompt,
                metadata={"default": True}
            )
        
        logger.info(f"Dialogue Manager initialized with max_history={max_history_length}")
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for AVA."""
        return (
            "You are AVA, a helpful and highly capable AI assistant optimized for agentic tasks. "
            "You are designed to be concise, accurate, and helpful. When you need to use a tool, "
            "use the format [TOOL_CALL: tool_name(param1=value1, param2=value2)]. "
            "Always explain your reasoning when performing complex tasks."
        )
    
    def add_message(
        self, 
        role: Union[MessageRole, str], 
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tool_call_id: Optional[str] = None,
        function_call: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Add a message to the dialogue history.
        
        Args:
            role: The role of the message sender
            content: The message content
            metadata: Additional metadata for the message
            tool_call_id: ID of associated tool call (for tool messages)
            function_call: Function call data (legacy format)
            tool_calls: List of tool calls (modern format)
            
        Returns:
            The message ID of the added message
        """
        try:
            # Convert string role to enum
            if isinstance(role, str):
                role = MessageRole(role.lower())
            
            # Validate content
            if not content or not content.strip():
                raise ValueError("Message content cannot be empty")
            
            # Create message
            message = Message(
                role=role,
                content=content.strip(),
                metadata=metadata or {},
                tool_call_id=tool_call_id,
                function_call=function_call,
                tool_calls=tool_calls
            )
            
            # Add to history
            self.messages.append(message)
            
            # Update state
            self._update_conversation_state(message)
            
            # Manage history length
            self._manage_history_length()
            
            logger.debug(f"Added {role.value} message: \"{content[:50]}...\" (ID: {message.message_id})")
            return message.message_id
        
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            self.state.errors_encountered += 1
            raise
    
    def _update_conversation_state(self, message: Message):
        """Update conversation state with new message."""
        self.state.total_messages += 1
        self.state.last_activity = datetime.now()
        
        if message.role == MessageRole.USER:
            self.state.user_messages += 1
        elif message.role == MessageRole.ASSISTANT:
            self.state.assistant_messages += 1
        
        # Count tool calls
        if message.tool_calls:
            self.state.tool_calls_made += len(message.tool_calls)
        elif message.function_call:
            self.state.tool_calls_made += 1
    
    def _manage_history_length(self):
        """Manage conversation history length according to settings."""
        if len(self.messages) <= self.max_history_length:
            return
        
        # Identify system messages to preserve
        system_messages = []
        non_system_messages = []
        
        for msg in self.messages:
            if msg.role == MessageRole.SYSTEM and self.preserve_system:
                system_messages.append(msg)
            else:
                non_system_messages.append(msg)
        
        # Calculate how many non-system messages to keep
        max_non_system = self.max_history_length - len(system_messages)
        
        if len(non_system_messages) > max_non_system:
            # Auto-summarize if enabled
            if self.auto_summarize and len(non_system_messages) > self.summary_threshold:
                summary = self._create_conversation_summary(
                    non_system_messages[:-max_non_system]
                )
                
                # Add summary as a system message
                summary_msg = Message(
                    role=MessageRole.SYSTEM,
                    content=f"[Conversation Summary]: {summary}",
                    metadata={"type": "auto_summary", "summarized_messages": len(non_system_messages) - max_non_system}
                )
                system_messages.append(summary_msg)
            
            # Keep only recent non-system messages
            non_system_messages = non_system_messages[-max_non_system:]
        
        # Rebuild messages list
        self.messages = system_messages + non_system_messages
        logger.debug(f"Managed history: kept {len(self.messages)} messages")
    
    def _create_conversation_summary(self, messages: List[Message]) -> str:
        """Create a summary of older messages."""
        if not messages:
            return "No previous conversation to summarize."
        
        # Simple summarization - in practice, this could use the LLM itself
        user_queries = [msg.content for msg in messages if msg.role == MessageRole.USER]
        assistant_responses = [msg.content for msg in messages if msg.role == MessageRole.ASSISTANT]
        tool_calls = sum(1 for msg in messages if msg.tool_calls or msg.function_call)
        
        summary = (
            f"Previous conversation had {len(user_queries)} user queries, "
            f"{len(assistant_responses)} assistant responses, and {tool_calls} tool calls. "
        )
        
        if user_queries:
            summary += f"Main topics discussed: {', '.join(user_queries[:3])}..."
        
        return summary
    
    def get_context(
        self, 
        include_system_prompt: bool = True,
        format_type: str = "messages",
        include_metadata: bool = False
    ) -> Union[List[Dict[str, Any]], str]:
        """
        Get the current conversational context.
        
        Args:
            include_system_prompt: Whether to include system messages
            format_type: Format of output ("messages", "string", "chat_ml")
            include_metadata: Whether to include message metadata
            
        Returns:
            Formatted context for the LLM
        """
        try:
            # Filter messages based on settings
            context_messages = []
            
            for message in self.messages:
                if not include_system_prompt and message.role == MessageRole.SYSTEM:
                    continue
                context_messages.append(message)
            
            # Format according to requested type
            if format_type == "messages":
                return self._format_as_messages(context_messages, include_metadata)
            elif format_type == "string":
                return self._format_as_string(context_messages)
            elif format_type == "chat_ml":
                return self._format_as_chat_ml(context_messages)
            else:
                raise ValueError(f"Unsupported format_type: {format_type}")
        
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return []
    
    def _format_as_messages(self, messages: List[Message], include_metadata: bool) -> List[Dict[str, Any]]:
        """Format messages as list of dictionaries."""
        formatted = []
        
        for msg in messages:
            msg_dict = {
                "role": msg.role.value,
                "content": msg.content
            }
            
            # Add tool-related fields if present
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            if msg.function_call:
                msg_dict["function_call"] = msg.function_call
            if msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            
            # Add metadata if requested
            if include_metadata:
                msg_dict["metadata"] = {
                    "timestamp": msg.timestamp.isoformat(),
                    "message_id": msg.message_id,
                    **msg.metadata
                }
            
            formatted.append(msg_dict)
        
        return formatted
    
    def _format_as_string(self, messages: List[Message]) -> str:
        """Format messages as a single string."""
        formatted_lines = []
        
        for msg in messages:
            role_name = msg.role.value.title()
            formatted_lines.append(f"{role_name}: {msg.content}")
        
        return "\n".join(formatted_lines)
    
    def _format_as_chat_ml(self, messages: List[Message]) -> str:
        """Format messages in ChatML format."""
        formatted_lines = []
        
        for msg in messages:
            formatted_lines.append(f"<|im_start|>{msg.role.value}")
            formatted_lines.append(msg.content)
            formatted_lines.append("<|im_end|>")
        
        return "\n".join(formatted_lines)
    
    def get_context_with_tool_result(
        self, 
        tool_name: str, 
        tool_result: Any,
        tool_call_id: Optional[str] = None,
        format_type: str = "messages"
    ) -> Union[List[Dict[str, Any]], str]:
        """
        Get context including a tool execution result.
        
        Args:
            tool_name: Name of the executed tool
            tool_result: Result from the tool execution
            tool_call_id: ID of the tool call
            format_type: Format of output
            
        Returns:
            Context with tool result included
        """
        try:
            # Format tool result
            if hasattr(tool_result, 'result'):
                # ToolResult object
                result_content = json.dumps(tool_result.result) if tool_result.success else f"Error: {tool_result.error_message}"
            else:
                # Raw result
                result_content = json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
            
            # Create temporary tool message
            temp_message = Message(
                role=MessageRole.TOOL,
                content=result_content,
                tool_call_id=tool_call_id or f"temp_{tool_name}",
                metadata={"tool_name": tool_name, "temporary": True}
            )
            
            # Get current context and add tool result
            context_messages = [msg for msg in self.messages]
            context_messages.append(temp_message)
            
            # Format according to requested type
            if format_type == "messages":
                return self._format_as_messages(context_messages, include_metadata=False)
            elif format_type == "string":
                return self._format_as_string(context_messages)
            elif format_type == "chat_ml":
                return self._format_as_chat_ml(context_messages)
            else:
                raise ValueError(f"Unsupported format_type: {format_type}")
        
        except Exception as e:
            logger.error(f"Error creating context with tool result: {e}")
            return self.get_context(format_type=format_type)
    
    def clear_history(self, keep_system: bool = True):
        """
        Clear the conversation history.
        
        Args:
            keep_system: Whether to preserve system messages
        """
        try:
            if keep_system:
                # Keep only system messages
                self.messages = [msg for msg in self.messages if msg.role == MessageRole.SYSTEM]
            else:
                self.messages = []
            
            # Reset state
            self.state = ConversationState(conversation_id=self.state.conversation_id)
            
            logger.info(f"History cleared (kept {len(self.messages)} system messages)")
        
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
    
    def update_system_prompt(self, new_prompt: str):
        """Update the system prompt."""
        try:
            # Remove existing default system messages
            self.messages = [
                msg for msg in self.messages 
                if not (msg.role == MessageRole.SYSTEM and msg.metadata.get("default"))
            ]
            
            # Add new system message
            self.add_message(
                role=MessageRole.SYSTEM,
                content=new_prompt,
                metadata={"default": True, "updated": True}
            )
            
            logger.info("System prompt updated")
        
        except Exception as e:
            logger.error(f"Error updating system prompt: {e}")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current conversation."""
        return {
            "state": {
                "total_messages": self.state.total_messages,
                "user_messages": self.state.user_messages,
                "assistant_messages": self.state.assistant_messages,
                "tool_calls_made": self.state.tool_calls_made,
                "errors_encountered": self.state.errors_encountered,
                "conversation_id": self.state.conversation_id,
                "last_activity": self.state.last_activity.isoformat()
            },
            "current_history": {
                "messages_in_memory": len(self.messages),
                "system_messages": len([m for m in self.messages if m.role == MessageRole.SYSTEM]),
                "user_messages_in_memory": len([m for m in self.messages if m.role == MessageRole.USER]),
                "assistant_messages_in_memory": len([m for m in self.messages if m.role == MessageRole.ASSISTANT])
            },
            "settings": {
                "max_history_length": self.max_history_length,
                "preserve_system": self.preserve_system,
                "auto_summarize": self.auto_summarize,
                "summary_threshold": self.summary_threshold
            }
        }
    
    def export_conversation(self, include_metadata: bool = True) -> Dict[str, Any]:
        """Export the full conversation for backup or analysis."""
        return {
            "conversation_id": self.state.conversation_id,
            "export_timestamp": datetime.now().isoformat(),
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "message_id": msg.message_id,
                    "metadata": msg.metadata if include_metadata else {},
                    "tool_call_id": msg.tool_call_id,
                    "function_call": msg.function_call,
                    "tool_calls": msg.tool_calls
                }
                for msg in self.messages
            ],
            "state": self.get_conversation_stats()
        }


# --- Test Functions ---
def test_dialogue_manager():
    """Test the enhanced dialogue manager."""
    print("--- Enhanced Dialogue Manager Test ---")
    
    # Initialize manager
    dm = DialogueManager(max_history_length=5, auto_summarize=True)
    
    # Test basic message addition
    print("\nTest 1: Basic message flow")
    dm.add_message(MessageRole.USER, "Hello AVA!")
    dm.add_message(MessageRole.ASSISTANT, "Hello! How can I help you today?")
    dm.add_message(MessageRole.USER, "What's the weather like?")
    dm.add_message(
        MessageRole.ASSISTANT, 
        "I'll check the weather for you.", 
        tool_calls=[{"id": "call_1", "function": {"name": "get_weather", "arguments": "{\"location\": \"default\"}"}}]
    )
    
    # Test tool result context
    print("\nTest 2: Tool result context")
    context_with_tool = dm.get_context_with_tool_result(
        tool_name="get_weather",
        tool_result={"temperature": 22, "condition": "sunny"},
        tool_call_id="call_1"
    )
    print(f"Context with tool result: {len(context_with_tool)} messages")
    
    # Test history management
    print("\nTest 3: History management")
    for i in range(10):
        dm.add_message(MessageRole.USER, f"Message {i}")
        dm.add_message(MessageRole.ASSISTANT, f"Response {i}")
    
    print(f"Messages after overflow: {len(dm.messages)}")
    
    # Test statistics
    print("\nTest 4: Conversation statistics")
    stats = dm.get_conversation_stats()
    print(f"Total messages processed: {stats['state']['total_messages']}")
    print(f"Messages in memory: {stats['current_history']['messages_in_memory']}")
    
    # Test export
    print("\nTest 5: Conversation export")
    export_data = dm.export_conversation()
    print(f"Exported conversation with {len(export_data['messages'])} messages")
    
    print("\n--- Test Complete ---")


if __name__ == "__main__":
    test_dialogue_manager()

print("Placeholder for AVA dialogue manager (src/ava_core/dialogue_manager.py)")
import json # Added for testing print output 