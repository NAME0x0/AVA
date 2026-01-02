"""
AVA TUI - Conversation Persistence
===================================

Handles saving and loading conversation history to SQLite database.
"""

import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single chat message."""
    role: str  # "user", "assistant", "error"
    content: str
    timestamp: str
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata"),
        )


@dataclass
class Session:
    """A conversation session."""
    id: str
    name: str
    created_at: str
    updated_at: str
    messages: list[Message]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
        }


class ConversationStore:
    """SQLite-based conversation persistence."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the conversation store.
        
        Args:
            db_path: Path to SQLite database. Defaults to data/conversations.db
        """
        if db_path is None:
            # Default to project data directory
            db_path = Path(__file__).parent.parent / "data" / "conversations" / "tui_sessions.db"
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session 
                ON messages(session_id)
            """)
            conn.commit()

    def create_session(self, name: Optional[str] = None) -> str:
        """Create a new conversation session.
        
        Args:
            name: Optional session name. Defaults to timestamp.
            
        Returns:
            Session ID
        """
        import uuid
        
        session_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        
        if name is None:
            name = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO sessions (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, name, now, now),
            )
            conn.commit()
        
        logger.info(f"Created session: {session_id}")
        return session_id

    def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str, 
        metadata: Optional[dict] = None
    ) -> None:
        """Add a message to a session.
        
        Args:
            session_id: Session ID
            role: Message role (user, assistant, error)
            content: Message content
            metadata: Optional metadata (cognitive_state, etc.)
        """
        now = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, now, metadata_json),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            conn.commit()

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session with all messages.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session object or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get session info
            session_row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            
            if not session_row:
                return None
            
            # Get messages
            message_rows = conn.execute(
                """
                SELECT role, content, timestamp, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
            
            messages = []
            for row in message_rows:
                metadata = json.loads(row["metadata"]) if row["metadata"] else None
                messages.append(Message(
                    role=row["role"],
                    content=row["content"],
                    timestamp=row["timestamp"],
                    metadata=metadata,
                ))
            
            return Session(
                id=session_row["id"],
                name=session_row["name"],
                created_at=session_row["created_at"],
                updated_at=session_row["updated_at"],
                messages=messages,
            )

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """List recent sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session summaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT s.id, s.name, s.created_at, s.updated_at,
                       COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                GROUP BY s.id
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            
            return [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "message_count": row["message_count"],
                }
                for row in rows
            ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            # Delete messages first (foreign key)
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            result = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            return result.rowcount > 0

    def search_messages(self, query: str, limit: int = 50) -> list[dict]:
        """Search messages across all sessions.
        
        Args:
            query: Search query (case-insensitive substring match)
            limit: Maximum results
            
        Returns:
            List of matching messages with session info
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT m.*, s.name as session_name
                FROM messages m
                JOIN sessions s ON m.session_id = s.id
                WHERE m.content LIKE ?
                ORDER BY m.timestamp DESC
                LIMIT ?
                """,
                (f"%{query}%", limit),
            ).fetchall()
            
            return [
                {
                    "session_id": row["session_id"],
                    "session_name": row["session_name"],
                    "role": row["role"],
                    "content": row["content"][:200] + "..." if len(row["content"]) > 200 else row["content"],
                    "timestamp": row["timestamp"],
                }
                for row in rows
            ]

    def get_latest_session(self) -> Optional[str]:
        """Get the ID of the most recent session.
        
        Returns:
            Session ID or None if no sessions exist
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM sessions ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else None

    def export_session(self, session_id: str, format: str = "json") -> Optional[str]:
        """Export a session to a string.
        
        Args:
            session_id: Session ID
            format: Export format (json, markdown)
            
        Returns:
            Exported content or None if session not found
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        if format == "json":
            return json.dumps(session.to_dict(), indent=2)
        elif format == "markdown":
            lines = [
                f"# {session.name}",
                f"Created: {session.created_at}",
                "",
            ]
            for msg in session.messages:
                role_prefix = "**You:**" if msg.role == "user" else "**AVA:**"
                if msg.role == "error":
                    role_prefix = "**Error:**"
                lines.append(f"{role_prefix}")
                lines.append(msg.content)
                lines.append("")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unknown format: {format}")
