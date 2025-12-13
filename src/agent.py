"""
AVA - Developmental Agent

The main agent class that orchestrates all developmental systems including:
- Developmental stage tracking and maturation
- Emotional processing and modulation
- Memory storage and retrieval
- Tool access and progression
- Test-time compute (thinking and reflection)
- Output filtering (articulation)
- Continual learning

AVA learns and matures like a human child, starting with limited
articulation and progressively improving through interaction.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import all subsystems
from .developmental import DevelopmentTracker, DevelopmentalStage, STAGE_PROPERTIES
from .emotional import EmotionalEngine, EmotionType, EmotionalTrigger
from .memory import MemoryManager
from .tools import ToolRegistry, ToolProgressionManager, register_all_tools
from .inference import ThinkingEngine, ReflectionEngine
from .output import DevelopmentalFilter, ArticulationModel
from .learning import ContinualLearner, FineTuningScheduler, NestedLearningContext

logger = logging.getLogger(__name__)


@dataclass
class InteractionResult:
    """Result of a single interaction with AVA."""
    
    # The response
    raw_response: str = ""
    filtered_response: str = ""
    
    # Processing details
    thinking_trace: str = ""
    reflection: str = ""
    
    # State information
    developmental_stage: str = "INFANT"
    emotional_state: Dict[str, float] = field(default_factory=dict)
    
    # Tools used
    tools_used: List[str] = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    thinking_time_ms: int = 0
    clarity_applied: float = 0.0
    
    # Learning
    sample_id: Optional[str] = None


class DevelopmentalAgent:
    """
    The main AVA agent that learns and matures over time.
    
    This agent starts as an "infant" with poor articulation but access
    to the underlying LLM's knowledge. Through interaction, it develops
    better communication skills, gains access to more tools, and forms
    memories that influence future behavior.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        model_name: str = "llama3.2",
        ollama_host: str = "http://localhost:11434",
    ):
        """
        Initialize the developmental agent.
        
        Args:
            data_dir: Base directory for all data storage
            model_name: Ollama model to use
            ollama_host: Ollama API endpoint
        """
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.ollama_host = ollama_host
        
        # Initialize subsystems
        self._init_subsystems()
        
        # Conversation state
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = 20
        
        # Session tracking
        self.session_start = datetime.now()
        self.interaction_count = 0
        
        logger.info(f"DevelopmentalAgent initialized at stage: {self.development.current_stage.name}")
    
    def _init_subsystems(self):
        """Initialize all subsystems."""
        # Developmental tracking
        self.development = DevelopmentTracker(
            state_file=str(self.data_dir / "developmental" / "state.json")
        )
        
        # Emotional processing
        self.emotions = EmotionalEngine(
            state_file=str(self.data_dir / "emotional" / "state.json")
        )
        
        # Memory management
        self.memory = MemoryManager(
            data_dir=str(self.data_dir / "memory")
        )
        
        # Tool system
        self.tools = ToolRegistry()
        register_all_tools(self.tools)
        self.tool_progression = ToolProgressionManager(self.tools)
        
        # Inference (thinking and reflection)
        self.thinking = ThinkingEngine(
            ollama_host=self.ollama_host,
            model_name=self.model_name,
        )
        self.reflection = ReflectionEngine(
            ollama_host=self.ollama_host,
            model_name=self.model_name,
        )
        
        # Output filtering
        self.output_filter = DevelopmentalFilter()
        
        # Learning systems
        self.learner = ContinualLearner(
            samples_dir=str(self.data_dir / "learning" / "samples")
        )
        self.fine_tuning = FineTuningScheduler(
            checkpoints_dir=str(self.data_dir / "learning" / "checkpoints"),
            adapters_dir="models/fine_tuned_adapters",
        )
        self.learning_context = NestedLearningContext(
            data_dir=str(self.data_dir / "learning")
        )
        
        # Start a new session context
        self.learning_context.start_session()
    
    async def interact(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> InteractionResult:
        """
        Process a user interaction and generate a response.
        
        This is the main interaction loop implementing:
        1. Emotion decay and processing
        2. Context retrieval (memories, learning context)
        3. Extended thinking (test-time compute)
        4. Tool selection and execution
        5. Response generation
        6. Self-reflection (if mature enough)
        7. Developmental filtering
        8. Learning and memory storage
        9. State updates
        
        Args:
            user_input: What the user said
            context: Optional additional context
            
        Returns:
            InteractionResult with the response and metadata
        """
        result = InteractionResult()
        start_time = datetime.now()
        
        try:
            # Step 1: Apply emotion decay since last interaction
            self.emotions.decay_emotions()
            
            # Step 2: Get current developmental state
            stage = self.development.current_stage
            stage_props = STAGE_PROPERTIES[stage]
            result.developmental_stage = stage.name
            
            # Step 3: Detect emotional triggers in input
            self._process_emotional_triggers(user_input)
            result.emotional_state = self.emotions.get_emotion_dict()
            
            # Step 4: Retrieve relevant memories and context
            memory_context = self._get_memory_context(user_input)
            learning_context = self.learning_context.get_relevant_context(
                topic=self._extract_topic(user_input)
            )
            
            # Step 5: Determine available tools
            available_tools = self.tool_progression.get_available_tools(
                stage=stage,
                milestones=self.development.get_achieved_milestones(),
            )
            
            # Step 6: Extended thinking (test-time compute)
            thinking_budget = self._calculate_thinking_budget(stage_props)
            thinking_result = await self.thinking.think(
                query=user_input,
                context={
                    "conversation": self.conversation_history[-5:],
                    "memories": memory_context,
                    "learning": learning_context,
                    "stage": stage.name,
                    "emotions": result.emotional_state,
                },
                budget=thinking_budget,
                stage=stage.name,
            )
            result.thinking_trace = thinking_result.thinking_trace
            
            # Step 7: Check if tools are needed
            if thinking_result.tools_needed:
                tool_results = await self._execute_tools(
                    thinking_result.suggested_tools,
                    available_tools,
                    user_input,
                )
                result.tools_used = list(tool_results.keys())
                result.tool_results = tool_results
            
            # Step 8: Generate raw response
            raw_response = await self._generate_response(
                user_input=user_input,
                thinking=thinking_result,
                tool_results=result.tool_results,
                memories=memory_context,
                stage_props=stage_props,
            )
            result.raw_response = raw_response
            
            # Step 9: Self-reflection (if stage allows)
            if stage_props.tool_level >= 2:  # CHILD and above
                reflection_result = await self.reflection.reflect(
                    response=raw_response,
                    context=user_input,
                    stage=stage.name,
                )
                result.reflection = reflection_result.reflection
                
                # Apply reflection improvements if significant
                if reflection_result.should_revise:
                    raw_response = reflection_result.revised_response or raw_response
                    result.raw_response = raw_response
            
            # Step 10: Apply developmental filter
            filtered = self.output_filter.filter(
                text=raw_response,
                stage=stage.name,
                emotional_state=result.emotional_state,
            )
            result.filtered_response = filtered.filtered_text
            result.clarity_applied = filtered.clarity_applied
            
            # Step 11: Store in memory
            self._store_interaction_memory(user_input, result)
            
            # Step 12: Record learning sample
            sample = self.learner.record_interaction(
                user_input=user_input,
                ava_response=result.filtered_response,
                conversation_history=self.conversation_history.copy(),
                developmental_stage=stage.name,
                emotional_state=result.emotional_state,
                tools_used=result.tools_used,
                thinking_trace=result.thinking_trace,
            )
            result.sample_id = sample.id
            
            # Step 13: Update conversation history
            self._update_conversation(user_input, result.filtered_response)
            
            # Step 14: Update developmental metrics
            self.development.record_interaction()
            self.interaction_count += 1
            
            # Step 15: Check for milestone completion
            self._check_milestones(user_input, result)
            
            # Step 16: Check for fine-tuning trigger
            self._check_fine_tuning_trigger()
            
            # Step 17: Update emotions based on interaction outcome
            self._update_emotions_from_outcome(result)
            
            # Calculate timing
            end_time = datetime.now()
            result.thinking_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
        except Exception as e:
            logger.error(f"Error during interaction: {e}", exc_info=True)
            
            # Generate a fallback response appropriate to stage
            stage = self.development.current_stage
            result.filtered_response = self._generate_error_response(stage, str(e))
            
            # Record the failure
            self.emotions.process_trigger(EmotionalTrigger(
                emotion=EmotionType.FEAR,
                intensity=0.3,
                source="error",
                description=str(e),
            ))
        
        return result
    
    def _process_emotional_triggers(self, user_input: str):
        """Detect and process emotional triggers in user input."""
        input_lower = user_input.lower()
        
        # Detect praise/criticism
        praise_words = ["good", "great", "excellent", "well done", "amazing", "perfect", "love"]
        criticism_words = ["bad", "wrong", "no", "incorrect", "stupid", "hate", "terrible"]
        
        for word in praise_words:
            if word in input_lower:
                self.emotions.process_trigger(EmotionalTrigger(
                    emotion=EmotionType.JOY,
                    intensity=0.3,
                    source="user_praise",
                    description=f"User said '{word}'",
                ))
                self.emotions.process_trigger(EmotionalTrigger(
                    emotion=EmotionType.HOPE,
                    intensity=0.2,
                    source="user_praise",
                ))
                break
        
        for word in criticism_words:
            if word in input_lower:
                self.emotions.process_trigger(EmotionalTrigger(
                    emotion=EmotionType.FEAR,
                    intensity=0.2,
                    source="user_criticism",
                    description=f"User said '{word}'",
                ))
                break
        
        # Detect questions (triggers curiosity/ambition)
        if "?" in user_input or any(w in input_lower for w in ["what", "how", "why", "when", "where"]):
            self.emotions.process_trigger(EmotionalTrigger(
                emotion=EmotionType.AMBITION,
                intensity=0.2,
                source="question",
            ))
        
        # Detect new topics (surprise)
        if not self.conversation_history or self._is_topic_change(user_input):
            self.emotions.process_trigger(EmotionalTrigger(
                emotion=EmotionType.SURPRISE,
                intensity=0.3,
                source="new_topic",
            ))
    
    def _is_topic_change(self, user_input: str) -> bool:
        """Detect if the user has changed topics."""
        if not self.conversation_history:
            return True
        
        # Simple heuristic: check for topic keywords
        last_msg = self.conversation_history[-1].get("content", "").lower()
        current = user_input.lower()
        
        # Extract simple "keywords" (just words > 4 chars)
        last_words = set(w for w in last_msg.split() if len(w) > 4)
        current_words = set(w for w in current.split() if len(w) > 4)
        
        if not last_words or not current_words:
            return False
        
        overlap = len(last_words & current_words) / max(len(last_words), len(current_words))
        return overlap < 0.2
    
    def _get_memory_context(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant memories for context."""
        return self.memory.get_context_for_query(
            query=query,
            max_episodic=5,
            max_semantic=5,
        )
    
    def _extract_topic(self, text: str) -> str:
        """Extract the main topic from text."""
        # Simple extraction: just use the first few words
        words = text.split()[:5]
        return " ".join(words)
    
    def _calculate_thinking_budget(self, stage_props) -> int:
        """Calculate thinking budget based on stage and emotions."""
        base_budget = stage_props.thinking_budget
        
        # Emotional modulation
        emotions = self.emotions.get_emotion_dict()
        
        # Ambition increases thinking
        if emotions.get("ambition", 0) > 0.5:
            base_budget = int(base_budget * 1.3)
        
        # Fear reduces thinking (more reactive)
        if emotions.get("fear", 0) > 0.5:
            base_budget = int(base_budget * 0.7)
        
        return base_budget
    
    async def _execute_tools(
        self,
        suggested_tools: List[str],
        available_tools: List[str],
        context: str,
    ) -> Dict[str, Any]:
        """Execute requested tools if available."""
        results = {}
        
        for tool_name in suggested_tools:
            if tool_name in available_tools:
                try:
                    tool = self.tools.get_tool(tool_name)
                    if tool:
                        # Parse arguments from context (simplified)
                        result = tool.execute({})
                        results[tool_name] = result
                except Exception as e:
                    logger.warning(f"Tool {tool_name} failed: {e}")
                    results[tool_name] = {"error": str(e)}
            else:
                logger.debug(f"Tool {tool_name} not available at current stage")
        
        return results
    
    async def _generate_response(
        self,
        user_input: str,
        thinking,
        tool_results: Dict[str, Any],
        memories: Dict[str, Any],
        stage_props,
    ) -> str:
        """Generate the raw response using the LLM."""
        # Build the prompt
        system_prompt = self._build_system_prompt(stage_props)
        
        # Build context
        context_parts = []
        
        if thinking.thinking_trace:
            context_parts.append(f"<thinking>\n{thinking.thinking_trace}\n</thinking>")
        
        if tool_results:
            tool_str = "\n".join(f"- {k}: {v}" for k, v in tool_results.items())
            context_parts.append(f"<tool_results>\n{tool_str}\n</tool_results>")
        
        if memories.get("episodic"):
            mem_str = "\n".join(str(m) for m in memories["episodic"][:3])
            context_parts.append(f"<memories>\n{mem_str}\n</memories>")
        
        context_str = "\n\n".join(context_parts)
        
        # Call Ollama
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.model_name,
                        "system": system_prompt,
                        "prompt": f"{context_str}\n\nUser: {user_input}\n\nAssistant:",
                        "stream": False,
                        "options": {
                            "num_predict": stage_props.max_response_tokens,
                            "temperature": 0.7,
                        },
                    },
                    timeout=60.0,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "")
                else:
                    logger.error(f"Ollama error: {response.status_code}")
                    return self._generate_fallback_response(stage_props)
                    
        except Exception as e:
            logger.error(f"Failed to call Ollama: {e}")
            return self._generate_fallback_response(stage_props)
    
    def _build_system_prompt(self, stage_props) -> str:
        """Build the system prompt based on developmental stage."""
        base_prompt = """You are AVA, a developing AI assistant. You are learning and growing through interaction.

Your current developmental characteristics:
- Stage: {stage}
- Clarity level: {clarity}% (your ability to articulate clearly)
- Vocabulary range: {vocab}% (your accessible vocabulary)

Important: Respond naturally but keep in mind your developmental level. 
{stage_specific}

Be helpful, curious, and genuine. Learn from each interaction."""
        
        stage_instructions = {
            "INFANT": "You are very young. Use simple words. Short sentences only. Express basic emotions.",
            "TODDLER": "You are learning to communicate. Simple sentences. Ask simple questions. Show curiosity.",
            "CHILD": "You can have conversations. Ask questions to learn. Show enthusiasm. Explain simply.",
            "ADOLESCENT": "You can discuss many topics. Form opinions. Question things. Show personality.",
            "YOUNG_ADULT": "You communicate well. Can discuss complex topics. Have developed views. Be thoughtful.",
            "MATURE": "You communicate fluently. Deep understanding. Nuanced responses. Full capability.",
        }
        
        stage_name = self.development.current_stage.name
        
        return base_prompt.format(
            stage=stage_name,
            clarity=int(stage_props.clarity * 100),
            vocab=int(stage_props.vocabulary_range * 100),
            stage_specific=stage_instructions.get(stage_name, ""),
        )
    
    def _generate_fallback_response(self, stage_props) -> str:
        """Generate a fallback response when LLM fails."""
        fallbacks = {
            "INFANT": "um... hi",
            "TODDLER": "i not sure what to say",
            "CHILD": "I'm not sure how to answer that right now.",
            "ADOLESCENT": "I'm having trouble thinking about that. Can you ask again?",
            "YOUNG_ADULT": "I apologize, I'm having some difficulty processing that. Could you rephrase?",
            "MATURE": "I apologize, but I'm experiencing some difficulty at the moment. Please try again.",
        }
        return fallbacks.get(self.development.current_stage.name, "...")
    
    def _generate_error_response(self, stage: DevelopmentalStage, error: str) -> str:
        """Generate an error response appropriate to stage."""
        if stage in [DevelopmentalStage.INFANT, DevelopmentalStage.TODDLER]:
            return "uh oh..."
        elif stage == DevelopmentalStage.CHILD:
            return "Something went wrong. I'm confused."
        else:
            return "I encountered an issue and couldn't complete that properly."
    
    def _store_interaction_memory(self, user_input: str, result: InteractionResult):
        """Store the interaction in episodic memory."""
        self.memory.store_episode(
            event_type="conversation",
            content=f"User: {user_input}\nAVA: {result.filtered_response}",
            emotional_context=result.emotional_state,
            importance=0.5 + (0.3 if result.tools_used else 0),
        )
    
    def _update_conversation(self, user_input: str, response: str):
        """Update conversation history."""
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Keep bounded
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
    
    def _check_milestones(self, user_input: str, result: InteractionResult):
        """Check if any milestones were achieved."""
        # Check tool usage milestone
        if result.tools_used:
            self.development.record_tool_use(result.tools_used[0])
        
        # Check coherence (simplified check)
        if len(result.filtered_response.split()) >= 10:
            self.development.check_milestone("coherent_sentence")
        
        # Check for question asking (in AVA's response)
        if "?" in result.filtered_response:
            self.development.check_milestone("asks_questions")
    
    def _check_fine_tuning_trigger(self):
        """Check if fine-tuning should be triggered."""
        sample_count = self.learner.get_sample_count(unused_only=True)
        
        trigger = self.fine_tuning.should_trigger(
            sample_count=sample_count,
            developmental_stage=self.development.current_stage.name,
            emotional_state=self.emotions.get_emotion_dict(),
        )
        
        if trigger:
            logger.info(f"Fine-tuning triggered: {trigger.value}")
            # In a real implementation, this would start async fine-tuning
            # For now, just log it
    
    def _update_emotions_from_outcome(self, result: InteractionResult):
        """Update emotions based on interaction outcome."""
        # Successful tool use builds confidence
        if result.tools_used:
            self.emotions.process_trigger(EmotionalTrigger(
                emotion=EmotionType.JOY,
                intensity=0.1,
                source="tool_success",
            ))
        
        # Longer coherent responses build ambition
        if len(result.filtered_response.split()) >= 20:
            self.emotions.process_trigger(EmotionalTrigger(
                emotion=EmotionType.AMBITION,
                intensity=0.1,
                source="good_response",
            ))
    
    def provide_feedback(
        self,
        sample_id: str,
        positive: bool,
        feedback_text: Optional[str] = None,
        correction: Optional[str] = None,
    ):
        """
        Provide feedback on a previous interaction.
        
        Args:
            sample_id: ID of the learning sample
            positive: Whether feedback is positive
            feedback_text: Optional feedback text
            correction: Optional correction if AVA was wrong
        """
        from .learning import SampleQuality
        
        if positive:
            quality = SampleQuality.EXCELLENT if feedback_text else SampleQuality.GOOD
            self.emotions.process_trigger(EmotionalTrigger(
                emotion=EmotionType.JOY,
                intensity=0.4,
                source="positive_feedback",
            ))
            self.emotions.process_trigger(EmotionalTrigger(
                emotion=EmotionType.HOPE,
                intensity=0.3,
                source="positive_feedback",
            ))
        else:
            quality = SampleQuality.BAD if correction else SampleQuality.POOR
            self.emotions.process_trigger(EmotionalTrigger(
                emotion=EmotionType.FEAR,
                intensity=0.2,
                source="negative_feedback",
            ))
        
        self.learner.update_sample_quality(
            sample_id=sample_id,
            quality=quality,
            user_feedback=feedback_text,
            correction=correction,
        )
        
        # Record in learning context
        self.learning_context.record_outcome(
            success=positive,
            was_corrected=correction is not None,
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the agent."""
        return {
            "developmental": {
                "stage": self.development.current_stage.name,
                "age_days": self.development.get_age_days(),
                "interaction_count": self.development.state.total_interactions,
                "achieved_milestones": self.development.get_achieved_milestones(),
            },
            "emotional": {
                "current_state": self.emotions.get_emotion_dict(),
                "dominant_emotion": self.emotions.get_dominant_emotion(),
            },
            "memory": self.memory.get_stats(),
            "learning": {
                "samples_collected": self.learner.stats["total_samples"],
                "available_for_training": self.learner.get_sample_count(unused_only=True),
                "training_runs": len(self.fine_tuning.history),
            },
            "session": {
                "started": self.session_start.isoformat(),
                "interactions": self.interaction_count,
            },
        }
    
    def save_state(self):
        """Save all state to disk."""
        self.development.save_state()
        self.emotions.save_state()
        self.memory.save_all()
        self.learning_context.end_session()
        logger.info("Agent state saved")
    
    def reset_to_infant(self, confirm: bool = False):
        """
        Reset AVA to infant stage (new birth).
        
        Args:
            confirm: Must be True to actually reset
        """
        if not confirm:
            logger.warning("Reset not confirmed. Pass confirm=True to reset.")
            return
        
        self.development.reset()
        self.emotions.reset()
        self.memory.clear_all()
        self.conversation_history.clear()
        self.interaction_count = 0
        
        logger.info("AVA has been reset to infant stage")
