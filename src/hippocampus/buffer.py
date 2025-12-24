"""
EPISODIC BUFFER: SQLite-backed Replay Buffer for High-Surprise Events
======================================================================

This module implements a persistent storage system for episodic memories
that are flagged as high-surprise by the Entropix metacognitive module.

Key Features:
1. SQLite persistence for durability across sessions
2. Priority sampling based on surprise values
3. Temporal decay for older memories
4. Integration with the Nightmare Engine for offline consolidation

The episodic buffer serves as the input to the "dreaming" process,
where high-value experiences are replayed and consolidated into
the model's long-term knowledge via QLoRA fine-tuning.

Reference: Experience Replay (Mnih et al., 2015)
"""

import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """
    A single episodic memory for replay.

    Episodes capture complete interaction cycles that can be
    replayed during offline consolidation (dreaming).

    Attributes:
        episode_id: Unique identifier
        timestamp: When the episode occurred
        prompt: User input that triggered this episode
        response: AVA's generated response
        embedding: Embedding vector (serialized)
        surprise: Entropix surprise signal
        entropy: Shannon entropy at generation
        varentropy: Variance of entropy
        cognitive_state: Classified cognitive state label
        quality_score: Post-hoc quality assessment
        used_tools: Whether tools were used
        tool_calls: Serialized tool call information
        metadata: Additional context
    """
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Core content
    prompt: str = ""
    response: str = ""

    # Embedding (stored as JSON list)
    embedding: list[float] | None = None

    # Entropix metrics
    surprise: float = 0.0
    entropy: float = 0.0
    varentropy: float = 0.0
    cognitive_state: str = "neutral"

    # Quality metrics
    quality_score: float = 0.5

    # Tool usage
    used_tools: bool = False
    tool_calls: list[dict[str, Any]] | None = None

    # Additional context
    metadata: dict[str, Any] = field(default_factory=dict)

    # Replay statistics
    replay_count: int = 0
    last_replayed: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "prompt": self.prompt,
            "response": self.response,
            "embedding": self.embedding,
            "surprise": self.surprise,
            "entropy": self.entropy,
            "varentropy": self.varentropy,
            "cognitive_state": self.cognitive_state,
            "quality_score": self.quality_score,
            "used_tools": self.used_tools,
            "tool_calls": self.tool_calls,
            "metadata": self.metadata,
            "replay_count": self.replay_count,
            "last_replayed": self.last_replayed,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        """Create from dictionary."""
        return cls(
            episode_id=data.get("episode_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            prompt=data.get("prompt", ""),
            response=data.get("response", ""),
            embedding=data.get("embedding"),
            surprise=data.get("surprise", 0.0),
            entropy=data.get("entropy", 0.0),
            varentropy=data.get("varentropy", 0.0),
            cognitive_state=data.get("cognitive_state", "neutral"),
            quality_score=data.get("quality_score", 0.5),
            used_tools=data.get("used_tools", False),
            tool_calls=data.get("tool_calls"),
            metadata=data.get("metadata", {}),
            replay_count=data.get("replay_count", 0),
            last_replayed=data.get("last_replayed"),
        )

    @property
    def priority_score(self) -> float:
        """
        Calculate priority score for replay sampling.

        Higher priority = more likely to be sampled for replay.
        Based on: surprise, quality, recency, and replay count.
        """
        # Surprise contribution (higher = more priority)
        surprise_factor = min(1.0, self.surprise / 3.0)

        # Quality contribution
        quality_factor = self.quality_score

        # Recency contribution (newer = higher)
        try:
            ts = datetime.fromisoformat(self.timestamp)
            days_ago = (datetime.now() - ts).days
            recency_factor = max(0.0, 1.0 - (days_ago * 0.05))
        except (ValueError, TypeError):
            recency_factor = 0.5

        # Replay count (less replayed = higher priority)
        replay_factor = max(0.1, 1.0 - (self.replay_count * 0.1))

        # Weighted combination
        priority = (
            0.4 * surprise_factor +
            0.3 * quality_factor +
            0.2 * recency_factor +
            0.1 * replay_factor
        )

        return priority


@dataclass
class BufferConfig:
    """
    Configuration for the Episodic Buffer.

    Attributes:
        db_path: Path to SQLite database file
        max_episodes: Maximum episodes to store
        surprise_threshold: Minimum surprise for storage
        cleanup_interval: How often to clean up old episodes
        priority_alpha: Alpha for priority sampling
    """
    db_path: str = "data/memory/episodic/replay_buffer.db"
    max_episodes: int = 10000
    surprise_threshold: float = 0.5
    cleanup_interval: int = 100  # Cleanup every N additions
    priority_alpha: float = 0.6  # 0 = uniform, 1 = pure priority


class EpisodicBuffer:
    """
    SQLite-backed replay buffer for high-surprise episodes.

    This buffer stores episodic memories for offline consolidation
    during the "dreaming" phase. Episodes are sampled based on
    priority (surprise, quality, recency) for replay.

    Usage:
        buffer = EpisodicBuffer()

        # Add an episode
        episode = Episode(
            prompt="What is the capital of France?",
            response="Paris is the capital of France.",
            surprise=2.5,
            quality_score=0.9,
        )
        buffer.add(episode)

        # Sample for replay
        batch = buffer.sample(batch_size=32)
    """

    def __init__(self, config: BufferConfig | None = None):
        """
        Initialize the episodic buffer.

        Args:
            config: Buffer configuration
        """
        self.config = config or BufferConfig()

        # Ensure directory exists
        db_dir = os.path.dirname(self.config.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        # Initialize database
        self._init_database()

        # Statistics
        self._add_count = 0

        logger.info(f"EpisodicBuffer initialized with db: {self.config.db_path}")

    def _init_database(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                timestamp TEXT,
                prompt TEXT,
                response TEXT,
                embedding TEXT,
                surprise REAL,
                entropy REAL,
                varentropy REAL,
                cognitive_state TEXT,
                quality_score REAL,
                used_tools INTEGER,
                tool_calls TEXT,
                metadata TEXT,
                replay_count INTEGER,
                last_replayed TEXT,
                priority_score REAL
            )
        """)

        # Create indexes for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_surprise
            ON episodes(surprise DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_priority
            ON episodes(priority_score DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON episodes(timestamp DESC)
        """)

        conn.commit()
        conn.close()

    def add(self, episode: Episode) -> bool:
        """
        Add an episode to the buffer.

        Episodes below the surprise threshold are rejected.

        Args:
            episode: Episode to add

        Returns:
            True if added, False if rejected
        """
        # Check surprise threshold
        if episode.surprise < self.config.surprise_threshold:
            logger.debug(
                f"Episode rejected: surprise {episode.surprise:.3f} "
                f"< threshold {self.config.surprise_threshold}"
            )
            return False

        # Calculate priority score
        priority = episode.priority_score

        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO episodes (
                    episode_id, timestamp, prompt, response, embedding,
                    surprise, entropy, varentropy, cognitive_state,
                    quality_score, used_tools, tool_calls, metadata,
                    replay_count, last_replayed, priority_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode.episode_id,
                episode.timestamp,
                episode.prompt,
                episode.response,
                json.dumps(episode.embedding) if episode.embedding else None,
                episode.surprise,
                episode.entropy,
                episode.varentropy,
                episode.cognitive_state,
                episode.quality_score,
                1 if episode.used_tools else 0,
                json.dumps(episode.tool_calls) if episode.tool_calls else None,
                json.dumps(episode.metadata),
                episode.replay_count,
                episode.last_replayed,
                priority,
            ))

            conn.commit()

            # Periodic cleanup
            self._add_count += 1
            if self._add_count % self.config.cleanup_interval == 0:
                self._cleanup(cursor)
                conn.commit()

            logger.debug(f"Episode added: {episode.episode_id[:8]}... (priority: {priority:.3f})")
            return True

        except sqlite3.Error as e:
            logger.error(f"Database error adding episode: {e}")
            return False
        finally:
            conn.close()

    def _cleanup(self, cursor: sqlite3.Cursor):
        """Remove oldest episodes if over capacity."""
        cursor.execute("SELECT COUNT(*) FROM episodes")
        count = cursor.fetchone()[0]

        if count > self.config.max_episodes:
            # Delete oldest episodes (by timestamp)
            to_delete = count - self.config.max_episodes
            cursor.execute("""
                DELETE FROM episodes WHERE episode_id IN (
                    SELECT episode_id FROM episodes
                    ORDER BY timestamp ASC
                    LIMIT ?
                )
            """, (to_delete,))
            logger.info(f"Cleaned up {to_delete} old episodes")

    def sample(
        self,
        batch_size: int = 32,
        min_surprise: float | None = None,
        cognitive_states: list[str] | None = None,
        prioritized: bool = True,
    ) -> list[Episode]:
        """
        Sample episodes from the buffer.

        Args:
            batch_size: Number of episodes to sample
            min_surprise: Minimum surprise filter
            cognitive_states: Filter by cognitive states
            prioritized: Use priority sampling vs uniform

        Returns:
            List of sampled episodes
        """
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()

        # Build query
        conditions = []
        params = []

        if min_surprise is not None:
            conditions.append("surprise >= ?")
            params.append(min_surprise)

        if cognitive_states:
            placeholders = ",".join("?" * len(cognitive_states))
            conditions.append(f"cognitive_state IN ({placeholders})")
            params.extend(cognitive_states)

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        if prioritized:
            # Priority sampling: sample with probability proportional to priority
            # Approximate with weighted random selection
            query = f"""
                SELECT * FROM episodes {where_clause}
                ORDER BY priority_score * RANDOM() DESC
                LIMIT ?
            """
        else:
            # Uniform random sampling
            query = f"""
                SELECT * FROM episodes {where_clause}
                ORDER BY RANDOM()
                LIMIT ?
            """

        params.append(batch_size)
        cursor.execute(query, params)

        episodes = []
        for row in cursor.fetchall():
            episode = self._row_to_episode(row)
            episodes.append(episode)

        conn.close()

        logger.debug(f"Sampled {len(episodes)} episodes")
        return episodes

    def _row_to_episode(self, row: tuple) -> Episode:
        """Convert database row to Episode object."""
        return Episode(
            episode_id=row[0],
            timestamp=row[1],
            prompt=row[2],
            response=row[3],
            embedding=json.loads(row[4]) if row[4] else None,
            surprise=row[5],
            entropy=row[6],
            varentropy=row[7],
            cognitive_state=row[8],
            quality_score=row[9],
            used_tools=bool(row[10]),
            tool_calls=json.loads(row[11]) if row[11] else None,
            metadata=json.loads(row[12]) if row[12] else {},
            replay_count=row[13],
            last_replayed=row[14],
        )

    def mark_replayed(self, episode_ids: list[str]):
        """
        Mark episodes as replayed (updates replay count and timestamp).

        Args:
            episode_ids: List of episode IDs that were replayed
        """
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        for episode_id in episode_ids:
            # Update replay count and recalculate priority
            cursor.execute("""
                UPDATE episodes
                SET replay_count = replay_count + 1,
                    last_replayed = ?,
                    priority_score = priority_score * 0.9
                WHERE episode_id = ?
            """, (now, episode_id))

        conn.commit()
        conn.close()

        logger.debug(f"Marked {len(episode_ids)} episodes as replayed")

    def get_high_priority_episodes(
        self,
        limit: int = 100,
        min_surprise: float = 1.0,
    ) -> list[Episode]:
        """
        Get highest priority episodes for consolidation.

        Args:
            limit: Maximum episodes to return
            min_surprise: Minimum surprise threshold

        Returns:
            List of high-priority episodes
        """
        return self.sample(
            batch_size=limit,
            min_surprise=min_surprise,
            prioritized=True,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get buffer statistics."""
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM episodes")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(surprise), AVG(quality_score), AVG(priority_score) FROM episodes")
        row = cursor.fetchone()
        avg_surprise = row[0] or 0.0
        avg_quality = row[1] or 0.0
        avg_priority = row[2] or 0.0

        cursor.execute("SELECT cognitive_state, COUNT(*) FROM episodes GROUP BY cognitive_state")
        state_dist = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute("SELECT SUM(replay_count) FROM episodes")
        total_replays = cursor.fetchone()[0] or 0

        conn.close()

        return {
            "total_episodes": total,
            "avg_surprise": avg_surprise,
            "avg_quality": avg_quality,
            "avg_priority": avg_priority,
            "cognitive_state_distribution": state_dist,
            "total_replays": total_replays,
            "max_capacity": self.config.max_episodes,
            "fill_ratio": total / self.config.max_episodes,
        }

    def clear(self):
        """Clear all episodes from the buffer."""
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM episodes")
        conn.commit()
        conn.close()
        logger.info("Episodic buffer cleared")

    def export_for_training(
        self,
        output_path: str,
        min_quality: float = 0.7,
        format: str = "jsonl",
    ) -> int:
        """
        Export episodes as training data.

        Args:
            output_path: Path to output file
            min_quality: Minimum quality score to include
            format: Output format ('jsonl' or 'json')

        Returns:
            Number of episodes exported
        """
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM episodes
            WHERE quality_score >= ?
            ORDER BY quality_score DESC
        """, (min_quality,))

        episodes = [self._row_to_episode(row) for row in cursor.fetchall()]
        conn.close()

        # Convert to training format
        training_data = []
        for ep in episodes:
            item = {
                "prompt": ep.prompt,
                "response": ep.response,
                "quality": ep.quality_score,
                "surprise": ep.surprise,
                "cognitive_state": ep.cognitive_state,
            }
            training_data.append(item)

        # Write output
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if format == "jsonl":
            with open(output_path, 'w') as f:
                for item in training_data:
                    f.write(json.dumps(item) + '\n')
        else:
            with open(output_path, 'w') as f:
                json.dump(training_data, f, indent=2)

        logger.info(f"Exported {len(training_data)} episodes to {output_path}")
        return len(training_data)


# Factory function
def create_episodic_buffer(
    db_path: str = "data/memory/episodic/replay_buffer.db",
    max_episodes: int = 10000,
    surprise_threshold: float = 0.5,
) -> EpisodicBuffer:
    """
    Factory function to create an episodic buffer.

    Args:
        db_path: Path to database file
        max_episodes: Maximum episodes to store
        surprise_threshold: Minimum surprise for storage

    Returns:
        Configured EpisodicBuffer
    """
    config = BufferConfig(
        db_path=db_path,
        max_episodes=max_episodes,
        surprise_threshold=surprise_threshold,
    )

    return EpisodicBuffer(config)
