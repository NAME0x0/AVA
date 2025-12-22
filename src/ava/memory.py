"""
AVA Memory System
=================

Handles conversation memory, context management, and learning.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import MemoryConfig  # Import from config to avoid duplication

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            metadata=data.get("metadata", {})
        )


class ConversationMemory:
    """
    Manages conversation history and context.
    
    Features:
    - Rolling window of recent messages
    - Automatic persistence
    - Context summarization (future)
    """
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.messages: List[Message] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        if self.config.persist_conversations:
            Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ConversationMemory initialized - Session: {self.session_id}")
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> Message:
        """Add a message to history."""
        msg = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(msg)
        
        # Trim if needed
        if len(self.messages) > self.config.max_history:
            self.messages = self.messages[-self.config.max_history:]
        
        return msg
    
    def add_exchange(self, user_msg: str, assistant_msg: str, metadata: Dict[str, Any] = None):
        """Add a user-assistant exchange."""
        self.add_message("user", user_msg, metadata)
        self.add_message("assistant", assistant_msg, metadata)
    
    def get_history(self, n: int = None) -> List[Dict[str, str]]:
        """Get recent history in chat format."""
        messages = self.messages[-n:] if n else self.messages
        return [{"role": m.role, "content": m.content} for m in messages]
    
    def get_context_string(self, n: int = 10) -> str:
        """Get recent context as formatted string."""
        recent = self.messages[-n:]
        lines = []
        for m in recent:
            prefix = "User: " if m.role == "user" else "AVA: "
            lines.append(f"{prefix}{m.content}")
        return "\n".join(lines)
    
    def clear(self):
        """Clear all messages."""
        self.messages = []
    
    def save(self, filename: str = None):
        """Save conversation to file."""
        if not self.config.persist_conversations:
            return
        
        filename = filename or f"conversation_{self.session_id}.json"
        path = Path(self.config.data_dir) / filename
        
        data = {
            "session_id": self.session_id,
            "saved_at": datetime.now().isoformat(),
            "messages": [m.to_dict() for m in self.messages]
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Conversation saved to {path}")
    
    def load(self, filename: str) -> bool:
        """Load conversation from file."""
        path = Path(self.config.data_dir) / filename
        
        if not path.exists():
            logger.warning(f"Conversation file not found: {path}")
            return False
        
        try:
            with open(path) as f:
                data = json.load(f)
            
            self.messages = [Message.from_dict(m) for m in data.get("messages", [])]
            self.session_id = data.get("session_id", self.session_id)
            logger.info(f"Loaded {len(self.messages)} messages from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return False
    
    def __len__(self) -> int:
        return len(self.messages)
