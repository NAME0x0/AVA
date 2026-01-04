"""
AVA CORE SYSTEM - The Unified Cortex-Medulla Architecture
==========================================================

This module integrates all components of the Cortex-Medulla architecture
into a unified system that implements continuous, autonomous operation.

System Flow:
============

1. SENSING (Medulla):
   - Continuous ingestion of sensory inputs (text, audio, logs)
   - Update SSM hidden state with O(1) memory
   - Calculate surprise signal

2. PERCEIVING (Medulla + Titans):
   - Update Titans neural memory based on surprise
   - Retrieve relevant memories

3. DECIDING (Agency):
   - Active Inference calculates Expected Free Energy
   - Select optimal policy (Reflex, DeepThought, Tool, Wait, etc.)

4. ACTING:
   - If REFLEX: Medulla generates quick response
   - If DEEP_THOUGHT: Bridge projects state → Cortex generates
   - If TOOL: Execute tool and feed results back

5. LEARNING (Titans):
   - Compress experience into neural memory weights
   - No context window growth

Hardware Budget (RTX A2000 4GB):
================================
- System Overhead:    ~300 MB
- Medulla (Mamba):    ~800 MB
- Titans Memory:      ~200 MB
- Bridge Adapter:     ~50 MB
- Cortex Buffer:      ~1,600 MB (paged)
- Headroom:           ~1,050 MB
- TOTAL:              ~3,000 MB / 4,096 MB available
"""

import asyncio
import logging
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .agency import ActiveInferenceController, AgencyConfig, BeliefState, Observation, PolicyType
from .bridge import Bridge, BridgeConfig
from .cortex import Cortex, CortexConfig

# Import architecture components
from .medulla import Medulla, MedullaConfig, ThermalMonitor

# Import existing systems for integration
try:
    from ..hippocampus.titans import TitansSidecarNumpy as TitansSidecar
except ImportError:
    TitansSidecar = None

# Import tools for Search-First workflow
try:
    from ..ava.tools import FactVerificationTool, WebBrowseTool, WebSearchTool
except ImportError:
    WebSearchTool = None
    WebBrowseTool = None
    FactVerificationTool = None

# Import episodic memory for experience storage
try:
    from ..hippocampus.titans import EpisodicMemory, EpisodicMemoryStore
except ImportError:
    EpisodicMemoryStore = None
    EpisodicMemory = None

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """Overall system operating state."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"


@dataclass
class CoreConfig:
    """
    Master configuration for the AVA Core System.

    This aggregates all component configs and provides system-wide settings.
    """

    # Component Configurations
    medulla_config: MedullaConfig = field(default_factory=MedullaConfig)
    cortex_config: CortexConfig = field(default_factory=CortexConfig)
    bridge_config: BridgeConfig = field(default_factory=BridgeConfig)
    agency_config: AgencyConfig = field(default_factory=AgencyConfig)

    # System Settings
    data_dir: str = "data"
    log_level: str = "INFO"

    # Main Loop Settings
    main_loop_interval: float = 0.1  # 100ms tick rate
    idle_loop_interval: float = 1.0  # 1s when idle

    # Sensory Input Configuration
    enable_audio_input: bool = False  # Requires faster-whisper
    enable_log_monitoring: bool = True  # Monitor system logs

    # Output Configuration
    enable_voice_output: bool = False  # Requires TTS

    # Safety Settings
    max_cortex_time: float = 300.0  # 5 min max per Cortex call
    emergency_shutdown_phrase: str = "ava shutdown"

    # Persistence
    autosave_interval: int = 100  # Save state every N interactions

    # ========== SEARCH-FIRST SETTINGS ==========
    # Web search is the default action for unknown queries
    search_first_enabled: bool = True
    search_min_sources: int = 3  # Minimum sources to check
    search_max_sources: int = 10  # Maximum sources to check
    fact_convergence_threshold: float = 0.7  # Agreement threshold for facts

    # ========== THERMAL MANAGEMENT ==========
    thermal_monitoring_enabled: bool = True
    thermal_check_interval: float = 5.0  # Check every 5 seconds
    max_gpu_power_percent: float = 15.0  # 15% max GPU power (RTX A2000)

    # ========== SELF-PRESERVATION ==========
    self_health_monitoring: bool = True
    health_check_interval: float = 60.0  # Check health every minute

    # ========== EPISODIC MEMORY ==========
    episodic_memory_enabled: bool = True
    episodic_memory_path: str = "data/memory/episodic"
    episodic_max_entries: int = 10000


@dataclass
class InteractionRecord:
    """Record of a single interaction for logging and learning."""

    timestamp: datetime = field(default_factory=datetime.now)

    # Input
    user_input: str = ""
    input_type: str = "text"  # text, audio, system

    # Processing
    surprise_value: float = 0.0
    selected_policy: str = ""
    used_cortex: bool = False

    # Output
    response: str = ""
    response_time_ms: float = 0.0

    # State
    belief_entropy: float = 0.0
    medulla_state_norm: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_input": self.user_input,
            "input_type": self.input_type,
            "surprise_value": self.surprise_value,
            "selected_policy": self.selected_policy,
            "used_cortex": self.used_cortex,
            "response": self.response,
            "response_time_ms": self.response_time_ms,
            "belief_entropy": self.belief_entropy,
        }


class AVACoreSystem:
    """
    The AVA Core System - Unified Cortex-Medulla Architecture.

    This is the main entry point for the v3 architecture. It orchestrates:

    1. Medulla: Always-on reflexive processing
    2. Cortex: Deep reasoning via AirLLM
    3. Bridge: Neural state projection for context transfer
    4. Agency: Active Inference for autonomous behavior
    5. Titans: Test-time learning for infinite memory
    6. Search-First: Web search as default for unknown queries
    7. Thermal: GPU power monitoring (15% max for RTX A2000)
    8. Episodic: Timestamp-based JSON memory storage

    The system runs continuously, responding to user input and proactively
    managing its own operation through Free Energy minimization.
    """

    def __init__(
        self,
        config: CoreConfig | None = None,
    ):
        """
        Initialize the AVA Core System.

        Args:
            config: Master configuration
        """
        self.config = config or CoreConfig()

        # System state
        self.state = SystemState.INITIALIZING
        self.is_running = False

        # Core components (lazy loading)
        self._medulla: Medulla | None = None
        self._cortex: Cortex | None = None
        self._bridge: Bridge | None = None
        self._agency: ActiveInferenceController | None = None
        self._titans: Any | None = None

        # Search-First & Self-Monitoring components
        self._thermal_monitor: ThermalMonitor | None = None
        self._episodic_store: Any | None = None  # EpisodicMemoryStore

        # Conversation state
        self.conversation_history: list[dict[str, str]] = []
        self.max_history = 50

        # Interaction tracking
        self.interaction_records: list[InteractionRecord] = []
        self.total_interactions = 0
        self.session_start = datetime.now()

        # Timestamps for periodic checks
        self._last_thermal_check = time.time()
        self._last_health_check = time.time()

        # Callbacks for external integration
        self._output_callback: Callable[[str], None] | None = None
        self._input_callback: Callable[[], str | None] | None = None

        logger.info("AVA Core System created (Search-First enabled)")

    async def initialize(self) -> None:
        """
        Initialize all system components.

        This performs startup checks and loads models:
        1. Initialize Titans memory
        2. Initialize Medulla (loads to VRAM)
        3. Initialize Cortex (loads to System RAM)
        4. Initialize Bridge
        5. Initialize Agency controller
        6. Initialize Thermal Monitor (Search-First)
        7. Initialize Episodic Memory Store (Search-First)
        8. Register action callbacks
        """
        logger.info("=" * 60)
        logger.info("INITIALIZING AVA CORE SYSTEM v3")
        logger.info("Cortex-Medulla Architecture + Search-First")
        logger.info("=" * 60)

        try:
            # 1. Initialize Titans Memory
            logger.info("[1/7] Initializing Titans Neural Memory...")
            await self._init_titans()

            # 2. Initialize Medulla (VRAM-resident)
            logger.info("[2/7] Initializing Medulla (Reflexive Core)...")
            await self._init_medulla()

            # 3. Initialize Cortex (System RAM)
            logger.info("[3/7] Initializing Cortex (Reflective Core)...")
            await self._init_cortex()

            # 4. Initialize Bridge
            logger.info("[4/7] Initializing Bridge (State Projection)...")
            await self._init_bridge()

            # 5. Initialize Agency
            logger.info("[5/7] Initializing Agency (Active Inference)...")
            await self._init_agency()

            # 6. Initialize Thermal Monitor (Search-First component)
            logger.info("[6/7] Initializing Thermal Monitor...")
            await self._init_thermal_monitor()

            # 7. Initialize Episodic Memory Store (Search-First component)
            logger.info("[7/7] Initializing Episodic Memory Store...")
            await self._init_episodic_store()

            # Register action callbacks (including Search-First handlers)
            self._register_action_callbacks()

            self.state = SystemState.RUNNING
            logger.info("=" * 60)
            logger.info("AVA CORE SYSTEM INITIALIZED SUCCESSFULLY")
            logger.info(f"Session ID: {self.session_start.strftime('%Y%m%d_%H%M%S')}")
            logger.info(
                f"Search-First: {'ENABLED' if self.config.search_first_enabled else 'DISABLED'}"
            )
            logger.info(f"Thermal Monitoring: {'ENABLED' if self._thermal_monitor else 'DISABLED'}")
            logger.info(f"Episodic Memory: {'ENABLED' if self._episodic_store else 'DISABLED'}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.state = SystemState.ERROR
            raise

    async def _init_titans(self) -> None:
        """Initialize Titans neural memory."""
        if TitansSidecar is not None:
            self._titans = TitansSidecar()
            logger.info("Titans memory initialized")
        else:
            logger.warning("Titans not available - using no persistent memory")
            self._titans = None

    async def _init_medulla(self) -> None:
        """Initialize Medulla with Titans memory."""
        self._medulla = Medulla(
            config=self.config.medulla_config,
            titans_memory=self._titans,
        )
        await self._medulla.initialize()

    async def _init_cortex(self) -> None:
        """Initialize Cortex (loads to System RAM)."""
        self._cortex = Cortex(config=self.config.cortex_config)
        # Don't initialize yet - lazy load on first use
        logger.info("Cortex configured (will load on first use)")

    async def _init_bridge(self) -> None:
        """Initialize Bridge for state projection."""
        self._bridge = Bridge(config=self.config.bridge_config)

    async def _init_agency(self) -> None:
        """Initialize Active Inference controller."""
        self._agency = ActiveInferenceController(config=self.config.agency_config)

    async def _init_thermal_monitor(self) -> None:
        """
        Initialize Thermal Monitor for GPU power management.

        Enforces 15% max GPU power constraint for RTX A2000.
        """
        if not self.config.thermal_monitoring_enabled:
            logger.info("Thermal monitoring disabled by config")
            self._thermal_monitor = None
            return

        try:
            self._thermal_monitor = ThermalMonitor(
                warning_temp=(
                    self.config.medulla_config.thermal_warning_temp
                    if hasattr(self.config.medulla_config, "thermal_warning_temp")
                    else 75.0
                ),
                throttle_temp=(
                    self.config.medulla_config.thermal_throttle_temp
                    if hasattr(self.config.medulla_config, "thermal_throttle_temp")
                    else 80.0
                ),
                pause_temp=(
                    self.config.medulla_config.thermal_pause_temp
                    if hasattr(self.config.medulla_config, "thermal_pause_temp")
                    else 85.0
                ),
            )
            status = self._thermal_monitor.get_status()
            logger.info(
                f"Thermal Monitor initialized - GPU: {status.temperature:.1f}°C, Power: {status.power_percent:.1f}%"
            )
        except Exception as e:
            logger.warning(f"Thermal monitoring not available: {e}")
            self._thermal_monitor = None

    async def _init_episodic_store(self) -> None:
        """
        Initialize Episodic Memory Store for timestamp-based JSON persistence.

        Stores experiences with timestamps for retrieval by date range.
        """
        if not self.config.episodic_memory_enabled:
            logger.info("Episodic memory disabled by config")
            self._episodic_store = None
            return

        if EpisodicMemoryStore is None:
            logger.warning("EpisodicMemoryStore not available")
            self._episodic_store = None
            return

        try:
            storage_path = Path(self.config.episodic_memory_path).resolve()
            allowed_base = Path(self.config.data_dir).resolve()

            # Security: Ensure path is within allowed data directory
            try:
                storage_path.relative_to(allowed_base)
            except ValueError:
                logger.error(
                    f"Security: Episodic memory path '{storage_path}' is outside "
                    f"allowed directory '{allowed_base}'. Using default."
                )
                storage_path = allowed_base / "memory" / "episodic"

            storage_path.mkdir(parents=True, exist_ok=True)

            self._episodic_store = EpisodicMemoryStore(
                storage_path=storage_path, max_memories=self.config.episodic_max_entries
            )

            count = len(self._episodic_store.memories)
            logger.info(f"Episodic Memory Store initialized - {count} existing memories loaded")
        except Exception as e:
            logger.warning(f"Episodic memory store not available: {e}")
            self._episodic_store = None

    def _register_action_callbacks(self) -> None:
        """
        Register action callbacks with the Agency controller.

        This includes Search-First workflow handlers for:
        - PRIMARY_SEARCH: Default web search for unknown queries
        - WEB_SEARCH: Extended search with multiple providers
        - WEB_BROWSE: Content extraction from URLs
        - SELF_MONITOR: Self-health and resource monitoring
        - THERMAL_CHECK: GPU thermal status monitoring
        - SYSTEM_COMMAND: Gated system-level commands
        """

        # ========== CORE ACTION HANDLERS ==========

        async def handle_reflex(obs: Observation, beliefs: BeliefState) -> dict:
            """Handle reflexive response via Medulla."""
            if obs.text:
                surprise, response = await self._medulla.perceive(input_text=obs.text)
                return {"response": response, "surprise": surprise.value}
            return {"response": None}

        async def handle_deep_thought(obs: Observation, beliefs: BeliefState) -> dict:
            """Handle deep reasoning via Cortex."""
            if obs.text:
                response = await self._invoke_cortex(obs.text)
                return {"response": response}
            return {"response": None}

        async def handle_wait(obs: Observation, beliefs: BeliefState) -> dict:
            """Handle waiting (no action)."""
            return {"action": "wait"}

        async def handle_ask_clarification(obs: Observation, beliefs: BeliefState) -> dict:
            """Handle asking for clarification."""
            clarification = "Could you please provide more details about what you need?"
            if self._output_callback:
                self._output_callback(clarification)
            return {"response": clarification}

        # ========== SEARCH-FIRST HANDLERS ==========

        async def handle_primary_search(obs: Observation, beliefs: BeliefState) -> dict:
            """
            Handle PRIMARY_SEARCH - the default action for unknown queries.

            Search-First paradigm: Web search is ALWAYS the first action
            for factual/informational queries before using internal generation.
            """
            if not WebSearchTool or not obs.text:
                return {"response": None, "error": "Search tool not available"}

            query = obs.text
            logger.info(f"[SEARCH-FIRST] Primary search for: {query[:50]}...")

            try:
                search_tool = WebSearchTool()
                results = await search_tool.execute(
                    query=query, num_results=self.config.search_min_sources
                )

                if results.get("error"):
                    logger.warning(f"Search error: {results['error']}")
                    return {"response": None, "search_failed": True}

                # Format search results for response
                search_data = results.get("results", [])
                if not search_data:
                    return {"response": None, "no_results": True}

                # Store search event in episodic memory
                if self._episodic_store:
                    memory = EpisodicMemory(
                        content=f"Searched: {query}",
                        context={"results_count": len(search_data)},
                        importance=0.6,
                        emotional_valence=0.0,
                        is_fact=True,
                    )
                    self._episodic_store.store(memory)

                # Build response from search results
                response_parts = [f"Based on {len(search_data)} sources:\n"]
                for i, result in enumerate(search_data[:5], 1):
                    title = result.get("title", "Unknown")
                    snippet = result.get("snippet", "")
                    url = result.get("url", "")
                    response_parts.append(f"{i}. **{title}**\n   {snippet}\n   Source: {url}\n")

                return {
                    "response": "\n".join(response_parts),
                    "search_results": search_data,
                    "sources_count": len(search_data),
                }

            except Exception as e:
                logger.error(f"Primary search error: {e}")
                return {"response": None, "error": str(e)}

        async def handle_web_search(obs: Observation, beliefs: BeliefState) -> dict:
            """
            Handle WEB_SEARCH - extended search with multiple providers.

            Used when primary search needs more sources or verification.
            """
            if not WebSearchTool or not obs.text:
                return {"response": None, "error": "Search tool not available"}

            query = obs.text
            logger.info(f"[SEARCH-EXTENDED] Extended search for: {query[:50]}...")

            try:
                search_tool = WebSearchTool()
                results = await search_tool.execute(
                    query=query, num_results=self.config.search_max_sources
                )

                return {
                    "search_results": results.get("results", []),
                    "sources_count": len(results.get("results", [])),
                }

            except Exception as e:
                logger.error(f"Extended search error: {e}")
                return {"error": str(e)}

        async def handle_web_browse(obs: Observation, beliefs: BeliefState) -> dict:
            """
            Handle WEB_BROWSE - extract content from a specific URL.

            Used when search results need deeper content extraction.
            """
            if not WebBrowseTool:
                return {"error": "Browse tool not available"}

            # Extract URL from observation context
            url = obs.context.get("url") if obs.context else None
            if not url:
                return {"error": "No URL provided for browsing"}

            logger.info(f"[WEB-BROWSE] Extracting content from: {url}")

            try:
                browse_tool = WebBrowseTool()
                result = await browse_tool.execute(url=url)

                return {
                    "content": result.get("content", ""),
                    "title": result.get("title", ""),
                    "url": url,
                }

            except Exception as e:
                logger.error(f"Browse error: {e}")
                return {"error": str(e)}

        # ========== SELF-MONITORING HANDLERS ==========

        async def handle_self_monitor(obs: Observation, beliefs: BeliefState) -> dict:
            """
            Handle SELF_MONITOR - check system health and resources.

            Self-preservation behavior: Monitor own health metrics.
            """
            logger.info("[SELF-MONITOR] Checking system health...")

            health_report = {"status": "healthy", "checks": {}}

            # Check thermal status
            if self._thermal_monitor:
                thermal_status = self._thermal_monitor.get_status()
                health_report["checks"]["thermal"] = {
                    "temperature": thermal_status.temperature,
                    "power_percent": thermal_status.power_percent,
                    "throttled": thermal_status.is_throttled,
                }
                if thermal_status.is_throttled:
                    health_report["status"] = "throttled"

            # Check memory usage
            try:
                import psutil

                memory = psutil.virtual_memory()
                health_report["checks"]["memory"] = {
                    "used_percent": memory.percent,
                    "available_gb": memory.available / (1024**3),
                }
                if memory.percent > 90:
                    health_report["status"] = "warning"
            except ImportError:
                pass

            # Check interaction stats
            health_report["checks"]["interactions"] = {
                "total": self.total_interactions,
                "uptime_hours": (datetime.now() - self.session_start).total_seconds() / 3600,
            }

            # Store health check in episodic memory
            if self._episodic_store:
                memory = EpisodicMemory(
                    content=f"Health check: {health_report['status']}",
                    context=health_report,
                    importance=0.3,
                    emotional_valence=0.0 if health_report["status"] == "healthy" else -0.3,
                    is_fact=True,
                )
                self._episodic_store.store(memory)

            return health_report

        async def handle_thermal_check(obs: Observation, beliefs: BeliefState) -> dict:
            """
            Handle THERMAL_CHECK - monitor GPU thermal status.

            15% max GPU power constraint for RTX A2000.
            """
            logger.info("[THERMAL-CHECK] Checking GPU thermal status...")

            if not self._thermal_monitor:
                return {"status": "unavailable", "message": "Thermal monitoring not enabled"}

            status = self._thermal_monitor.get_status()

            result = {
                "temperature": status.temperature,
                "power_percent": status.power_percent,
                "power_watts": status.power_draw_watts,
                "is_throttled": status.is_throttled,
                "status": "ok",
            }

            # Check against thresholds
            if self._thermal_monitor.should_pause():
                result["status"] = "critical"
                result["action"] = "pause_processing"
            elif self._thermal_monitor.should_throttle():
                result["status"] = "warning"
                result["action"] = "reduce_load"
            elif self._thermal_monitor.is_power_exceeded():
                result["status"] = "power_limit"
                result["action"] = "reduce_power"

            return result

        async def handle_system_command(obs: Observation, beliefs: BeliefState) -> dict:
            """
            Handle SYSTEM_COMMAND - gated system-level commands.

            IMPORTANT: System commands ALWAYS require user confirmation.
            This is a non-negotiable safety gate.
            """
            if not obs.text:
                return {"error": "No command provided"}

            command = obs.text.strip()
            logger.info(f"[SYSTEM-COMMAND] Request for: {command[:50]}...")

            # Check if command is blocked
            blocked_commands = (
                self.config.agency_config.blocked_system_commands
                if hasattr(self.config.agency_config, "blocked_system_commands")
                else [
                    "rm -rf",
                    "del /f",
                    "format",
                    "shutdown",
                    "reboot",
                    "kill -9",
                    "taskkill /f",
                    "dd if=",
                    "mkfs",
                    "fdisk",
                ]
            )

            for blocked in blocked_commands:
                if blocked.lower() in command.lower():
                    return {
                        "status": "blocked",
                        "message": f"Command contains blocked operation: {blocked}",
                        "requires_confirmation": False,
                    }

            # All non-blocked commands require confirmation
            return {
                "status": "pending_confirmation",
                "command": command,
                "message": "This system command requires your explicit confirmation to execute.",
                "requires_confirmation": True,
            }

        # ========== REGISTER ALL CALLBACKS ==========

        # Core handlers
        self._agency.register_action_callback(PolicyType.REFLEX_REPLY, handle_reflex)
        self._agency.register_action_callback(PolicyType.DEEP_THOUGHT, handle_deep_thought)
        self._agency.register_action_callback(PolicyType.WAIT, handle_wait)
        self._agency.register_action_callback(
            PolicyType.ASK_CLARIFICATION, handle_ask_clarification
        )

        # Search-First handlers
        self._agency.register_action_callback(PolicyType.PRIMARY_SEARCH, handle_primary_search)
        self._agency.register_action_callback(PolicyType.WEB_SEARCH, handle_web_search)
        self._agency.register_action_callback(PolicyType.WEB_BROWSE, handle_web_browse)

        # Self-monitoring handlers
        self._agency.register_action_callback(PolicyType.SELF_MONITOR, handle_self_monitor)
        self._agency.register_action_callback(PolicyType.THERMAL_CHECK, handle_thermal_check)
        self._agency.register_action_callback(PolicyType.SYSTEM_COMMAND, handle_system_command)

        logger.info("All action callbacks registered (including Search-First workflow)")

    def set_output_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for outputting responses."""
        self._output_callback = callback

    def set_input_callback(self, callback: Callable[[], str | None]) -> None:
        """Set callback for getting user input."""
        self._input_callback = callback

    async def process_input(
        self,
        user_input: str,
        force_cortex: bool = False,
        force_search: bool = False,
    ) -> str:
        """
        Process user input and generate a response.

        Search-First Paradigm:
        - Web search is the DEFAULT action for informational queries
        - Only falls back to internal generation if search fails or is inappropriate

        This is the main interaction method that:
        1. Checks thermal status (self-preservation)
        2. Passes input through Medulla for surprise calculation
        3. Uses Agency to select optimal policy (Search-First by default)
        4. Executes the selected policy
        5. Stores experience in episodic memory
        6. Returns the response

        Args:
            user_input: User's input text
            force_cortex: Force Cortex processing regardless of surprise
            force_search: Force web search regardless of query type

        Returns:
            Generated response string
        """
        start_time = time.time()
        record = InteractionRecord(user_input=user_input)

        try:
            # 0. Pre-check: Thermal status (self-preservation)
            if self._thermal_monitor and self.config.thermal_monitoring_enabled:
                thermal_status = self._thermal_monitor.get_status()
                if self._thermal_monitor.should_pause():
                    logger.warning(f"Thermal pause triggered: {thermal_status.temperature}°C")
                    return "I need to pause for thermal management. My GPU temperature is too high. Please wait a moment."
                elif self._thermal_monitor.should_throttle():
                    logger.info(f"Thermal throttling active: {thermal_status.temperature}°C")
                    # Continue with reduced processing

            # 1. Process through Medulla
            surprise, medulla_response = await self._medulla.perceive(input_text=user_input)
            record.surprise_value = surprise.value

            # 2. Create observation for Agency with Search-First context
            observation = Observation(
                text=user_input,
                surprise_signal=surprise.value,
                query_complexity=self._estimate_complexity(user_input),
            )

            # 3. Determine if this is a search-worthy query
            is_informational = self._is_informational_query(user_input)

            # 4. Select policy with Search-First bias
            if force_search or (
                self.config.search_first_enabled and is_informational and not force_cortex
            ):
                # Search-First: Try web search first
                logger.info("[SEARCH-FIRST] Informational query detected, searching...")

                policy = PolicyType.PRIMARY_SEARCH
                result = await self._agency.execute_policy(policy, observation)
                record.selected_policy = policy.name

                search_result = result.get("action_result", {})

                if search_result.get("response"):
                    # Search succeeded - use search results
                    response = search_result["response"]
                    logger.info(
                        f"[SEARCH-FIRST] Found {search_result.get('sources_count', 0)} sources"
                    )
                elif search_result.get("search_failed") or search_result.get("no_results"):
                    # Search failed - fall back to internal generation
                    logger.info("[SEARCH-FIRST] Search failed, falling back to Medulla/Cortex")
                    if surprise.requires_cortex:
                        record.used_cortex = True
                        response = await self._invoke_cortex(user_input)
                    else:
                        response = medulla_response
                else:
                    response = medulla_response

            elif force_cortex or surprise.requires_cortex:
                # Direct Cortex path (high surprise or forced)
                record.selected_policy = PolicyType.DEEP_THOUGHT.name
                record.used_cortex = True
                response = await self._invoke_cortex(user_input)
            else:
                # Standard Agency decision (includes Search-First in policy selection)
                policy, result = await self._agency.process_observation(observation)
                record.selected_policy = policy.name

                if policy == PolicyType.DEEP_THOUGHT:
                    record.used_cortex = True
                    response = await self._invoke_cortex(user_input)
                elif policy == PolicyType.PRIMARY_SEARCH:
                    # Agency selected search
                    search_result = result.get("action_result", {})
                    response = search_result.get("response") or medulla_response
                elif "response" in result.get("action_result", {}):
                    response = result["action_result"]["response"]
                else:
                    response = medulla_response

            # 5. Record interaction
            record.response = response or ""
            record.response_time_ms = (time.time() - start_time) * 1000
            record.belief_entropy = self._agency.beliefs.entropy

            # Update conversation history
            self._update_history(user_input, response)

            # Store in episodic memory
            if self._episodic_store and EpisodicMemory:
                memory = EpisodicMemory(
                    content=f"User: {user_input[:100]}... -> {record.selected_policy}",
                    context={
                        "user_input": user_input,
                        "response_preview": (response or "")[:200],
                        "policy": record.selected_policy,
                        "surprise": record.surprise_value,
                        "used_cortex": record.used_cortex,
                    },
                    importance=min(record.surprise_value + 0.3, 1.0),
                    emotional_valence=0.0,
                    is_fact=False,
                )
                self._episodic_store.store(memory)

            # Store record
            self.interaction_records.append(record)
            self.total_interactions += 1

            # Autosave
            if self.total_interactions % self.config.autosave_interval == 0:
                await self._autosave()

            return response or "I'm not sure how to respond to that."

        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return f"I encountered an error: {str(e)}"

    def _is_informational_query(self, text: str) -> bool:
        """
        Determine if a query is informational and should trigger Search-First.

        Returns True for queries that are likely seeking factual information.
        """
        text_lower = text.lower()

        # Question words that suggest informational queries
        question_indicators = [
            "what is",
            "what are",
            "who is",
            "who are",
            "when did",
            "when was",
            "when is",
            "where is",
            "where are",
            "where did",
            "how does",
            "how do",
            "how did",
            "how to",
            "why is",
            "why do",
            "why did",
            "explain",
            "define",
            "describe",
            "tell me about",
            "what do you know about",
            "facts about",
            "information about",
            "latest",
            "recent",
            "news",
            "current",
        ]

        # Check for question indicators
        for indicator in question_indicators:
            if indicator in text_lower:
                return True

        # Check for question mark
        if "?" in text:
            return True

        # Conversational/command patterns that don't need search
        non_search_patterns = [
            "hello",
            "hi ",
            "hey ",
            "thanks",
            "thank you",
            "please help",
            "can you",
            "could you",
            "i want",
            "i need",
            "i feel",
            "my name",
            "call me",
            "remember",
        ]

        for pattern in non_search_patterns:
            if pattern in text_lower:
                return False

        # Default to search for longer queries (likely seeking information)
        return len(text.split()) > 5

    async def _invoke_cortex(self, query: str) -> str:
        """
        Invoke the Cortex for deep reasoning.

        This method:
        1. Gets Medulla's current state
        2. Uses Bridge to project state to Cortex embeddings
        3. Prepares the full prompt
        4. Calls Cortex for generation

        Args:
            query: User's query

        Returns:
            Cortex-generated response
        """
        logger.info("Invoking Cortex for deep reasoning...")

        # Output waiting message
        if self._output_callback:
            waiting_msg = await self._medulla._generate_waiting_response()
            self._output_callback(f"[Thinking] {waiting_msg}")

        # 1. Get Medulla state
        medulla_state = self._medulla.get_state_vector()

        # 2. Prepare Cortex input via Bridge
        cortex_input = await self._bridge.prepare_cortex_input(
            medulla_state=medulla_state,
            current_query=query,
            conversation_history=self.conversation_history,
        )

        # 3. Generate with Cortex (with timeout protection)
        try:
            result = await asyncio.wait_for(
                self._cortex.generate(
                    prompt=cortex_input["text_prompt"],
                    projected_state=cortex_input["soft_prompts"],
                ),
                timeout=(
                    self.config.max_cortex_time
                    if hasattr(self.config, "max_cortex_time")
                    else 300.0
                ),
            )
        except asyncio.TimeoutError:
            logger.error(f"Cortex generation timed out after {self.config.max_cortex_time}s")
            return "I'm sorry, but that thought took too long. Please try a simpler question or try again."
        except Exception as e:
            logger.exception(f"Cortex generation failed: {e}")
            return f"I encountered an error while thinking: {type(e).__name__}"

        if result.error:
            logger.error(f"Cortex generation error: {result.error}")
            return f"I had trouble thinking about that: {result.error}"

        logger.info(
            f"Cortex complete: {result.output_tokens} tokens in "
            f"{result.total_time_seconds:.1f}s"
        )

        return result.text

    def _estimate_complexity(self, text: str) -> float:
        """Estimate query complexity for Agency decision-making."""
        complexity = 0.0

        # Length factor
        words = len(text.split())
        complexity += min(words / 50, 0.3)

        # Question words
        question_words = ["how", "why", "explain", "analyze", "compare", "evaluate"]
        if any(w in text.lower() for w in question_words):
            complexity += 0.3

        # Technical indicators
        technical_words = ["code", "implement", "algorithm", "function", "debug"]
        if any(w in text.lower() for w in technical_words):
            complexity += 0.3

        # Multiple questions
        complexity += min(text.count("?") * 0.1, 0.2)

        return min(complexity, 1.0)

    def _update_history(self, user_input: str, response: str) -> None:
        """Update conversation history."""
        self.conversation_history.append({"role": "user", "content": user_input})
        if response:
            self.conversation_history.append({"role": "assistant", "content": response})

        # Trim history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history :]

    async def run_forever(self) -> None:
        """
        Run the system in continuous autonomous mode.

        This implements the "always-on" behavior where the system:
        1. Monitors for user input
        2. Periodically checks thermal status (self-preservation)
        3. Periodically checks system health (self-monitoring)
        4. Processes observations through Agency (Search-First enabled)
        5. Takes proactive actions when uncertainty is high
        6. Never exits until explicitly stopped
        """
        logger.info("Starting AVA Core System continuous operation...")
        logger.info(
            f"Search-First: {'ENABLED' if self.config.search_first_enabled else 'DISABLED'}"
        )
        self.is_running = True

        # Set up signal handlers
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            self.is_running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        last_input_time = time.time()
        thermal_paused = False

        while self.is_running:
            try:
                # ========== THERMAL CHECK (Self-Preservation) ==========
                current_time = time.time()
                if (
                    self._thermal_monitor
                    and current_time - self._last_thermal_check
                    >= self.config.thermal_check_interval
                ):

                    self._last_thermal_check = current_time
                    thermal_status = self._thermal_monitor.get_status()

                    if self._thermal_monitor.should_pause():
                        if not thermal_paused:
                            logger.warning(
                                f"[THERMAL] Pausing - GPU at {thermal_status.temperature:.1f}°C"
                            )
                            thermal_paused = True
                            if self._output_callback:
                                self._output_callback(
                                    f"[System] Thermal pause - GPU temperature: {thermal_status.temperature:.1f}°C. "
                                    "Waiting for cooldown..."
                                )
                        await asyncio.sleep(5.0)  # Wait longer when thermally paused
                        continue
                    elif thermal_paused:
                        # Resume from thermal pause
                        logger.info(
                            f"[THERMAL] Resuming - GPU at {thermal_status.temperature:.1f}°C"
                        )
                        thermal_paused = False
                        if self._output_callback:
                            self._output_callback(
                                f"[System] Thermal recovery - GPU temperature: {thermal_status.temperature:.1f}°C. "
                                "Resuming normal operation."
                            )

                    # Check power limit (15% max for RTX A2000)
                    if self._thermal_monitor.is_power_exceeded():
                        logger.warning(
                            f"[THERMAL] Power limit exceeded: {thermal_status.power_percent:.1f}% > "
                            f"{self.config.max_gpu_power_percent}%"
                        )

                # ========== HEALTH CHECK (Self-Monitoring) ==========
                if (
                    self.config.self_health_monitoring
                    and current_time - self._last_health_check >= self.config.health_check_interval
                ):

                    self._last_health_check = current_time

                    # Run self-monitoring via Agency
                    obs = Observation(text="", surprise_signal=0.0)
                    await self._agency.execute_policy(PolicyType.SELF_MONITOR, obs)
                    logger.debug("[HEALTH] Periodic health check completed")

                # ========== USER INPUT PROCESSING ==========
                user_input = None
                if self._input_callback:
                    user_input = self._input_callback()

                if user_input:
                    # Check for emergency shutdown
                    if self.config.emergency_shutdown_phrase in user_input.lower():
                        logger.info("Emergency shutdown phrase detected")
                        break

                    # Process input (Search-First workflow)
                    response = await self.process_input(user_input)

                    # Output response
                    if self._output_callback:
                        self._output_callback(response)

                    last_input_time = time.time()

                    # Short sleep for responsive interaction
                    await asyncio.sleep(self.config.main_loop_interval)
                else:
                    # ========== IDLE STATE - PROACTIVE BEHAVIOR ==========
                    time_since_input = time.time() - last_input_time

                    # Create observation from current state
                    observation = Observation(
                        silence_duration=time_since_input,
                        surprise_signal=0.0,
                    )

                    # Let Agency decide on proactive behavior
                    policy, result = await self._agency.process_observation(observation)

                    # Handle proactive actions
                    if policy == PolicyType.ASK_CLARIFICATION and time_since_input > 300:
                        proactive_msg = "Is there anything I can help you with?"
                        if self._output_callback:
                            self._output_callback(f"[Proactive] {proactive_msg}")
                        last_input_time = time.time()  # Reset timer
                    elif policy == PolicyType.SELF_MONITOR:
                        # Proactive self-monitoring during idle
                        logger.debug("[IDLE] Running proactive self-monitoring")

                    # Longer sleep when idle
                    await asyncio.sleep(self.config.idle_loop_interval)

            except asyncio.CancelledError:
                logger.info("Main loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1.0)

        await self.shutdown()

    async def _autosave(self) -> None:
        """Autosave system state including episodic memory."""
        logger.debug("Autosaving state...")
        try:
            self._medulla._save_state()
            self._bridge.save_state()
            self._agency.save_state()

            # Save episodic memory
            if self._episodic_store and hasattr(self._episodic_store, "save"):
                self._episodic_store.save()

        except Exception as e:
            logger.error(f"Autosave failed: {e}")

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the system.

        Saves all state and releases resources.
        """
        logger.info("Initiating AVA Core System shutdown...")
        self.state = SystemState.SHUTTING_DOWN
        self.is_running = False

        try:
            # Save state
            logger.info("Saving system state...")
            await self._autosave()

            # Shutdown components
            if self._medulla:
                await self._medulla.shutdown()

            if self._cortex:
                await self._cortex.shutdown()

            # Save episodic memory one final time
            if self._episodic_store and hasattr(self._episodic_store, "save"):
                logger.info("Saving episodic memory...")
                self._episodic_store.save()

            # Clean up thermal monitor
            if self._thermal_monitor:
                logger.info("Shutting down thermal monitor...")
                # ThermalMonitor cleanup handled by pynvml

            logger.info("AVA Core System shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        self.state = SystemState.INITIALIZING

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive system statistics including Search-First metrics."""
        uptime = (datetime.now() - self.session_start).total_seconds()

        stats = {
            "system": {
                "state": self.state.value,
                "uptime_seconds": uptime,
                "total_interactions": self.total_interactions,
                "session_start": self.session_start.isoformat(),
                "search_first_enabled": self.config.search_first_enabled,
            },
            "medulla": self._medulla.get_stats() if self._medulla else {},
            "cortex": self._cortex.get_stats() if self._cortex else {},
            "bridge": self._bridge.get_stats() if self._bridge else {},
            "agency": self._agency.get_stats() if self._agency else {},
        }

        # Add thermal monitoring stats
        if self._thermal_monitor:
            thermal_status = self._thermal_monitor.get_status()
            stats["thermal"] = {
                "enabled": True,
                "temperature": thermal_status.temperature,
                "power_percent": thermal_status.power_percent,
                "power_watts": thermal_status.power_draw_watts,
                "is_throttled": thermal_status.is_throttled,
                "max_power_limit": self.config.max_gpu_power_percent,
            }
        else:
            stats["thermal"] = {"enabled": False}

        # Add episodic memory stats
        if self._episodic_store:
            stats["episodic_memory"] = {
                "enabled": True,
                "total_memories": len(self._episodic_store.memories),
                "max_entries": self.config.episodic_max_entries,
            }
        else:
            stats["episodic_memory"] = {"enabled": False}

        # Add Search-First usage stats from interaction records
        search_count = sum(1 for r in self.interaction_records if "SEARCH" in r.selected_policy)
        cortex_count = sum(1 for r in self.interaction_records if r.used_cortex)

        stats["search_first"] = {
            "enabled": self.config.search_first_enabled,
            "search_queries": search_count,
            "cortex_queries": cortex_count,
            "search_ratio": search_count / max(self.total_interactions, 1),
        }

        return stats

    def get_interaction_log(self, n: int = 10) -> list[dict]:
        """Get the last N interaction records."""
        return [r.to_dict() for r in self.interaction_records[-n:]]


# Convenience function for quick start
async def create_and_run_ava(config: CoreConfig | None = None) -> AVACoreSystem:
    """
    Create and initialize an AVA Core System instance.

    Args:
        config: Optional configuration

    Returns:
        Initialized AVA Core System
    """
    ava = AVACoreSystem(config)
    await ava.initialize()
    return ava
