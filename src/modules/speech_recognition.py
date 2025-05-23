#!/usr/bin/env python3
"""
Enhanced Speech Recognition Module for AVA
Production-Ready ASR with Multiple Backends and Real-time Capabilities
Optimized for RTX A2000 4GB VRAM constraints
"""

import asyncio
import io
import json
import logging
import os
import tempfile
import time
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator
import warnings

# Core imports
import numpy as np

# Speech recognition imports with fallbacks
try:
    import speech_recognition as sr
    HAS_SPEECH_RECOGNITION = True
except ImportError:
    HAS_SPEECH_RECOGNITION = False
    warnings.warn("SpeechRecognition not available - basic ASR unavailable")

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    warnings.warn("OpenAI Whisper not available - advanced ASR unavailable")

try:
    import torch
    import torchaudio
    HAS_TORCH_AUDIO = True
except ImportError:
    HAS_TORCH_AUDIO = False
    warnings.warn("torchaudio not available - GPU-accelerated ASR unavailable")

try:
    import webrtcvad
    HAS_WEBRTC_VAD = True
except ImportError:
    HAS_WEBRTC_VAD = False
    warnings.warn("webrtcvad not available - advanced VAD unavailable")

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    warnings.warn("PyAudio not available - real-time ASR unavailable")

# Language detection
try:
    import langdetect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    warnings.warn("langdetect not available - language detection limited")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ASRBackend(Enum):
    """Available ASR backends."""
    WHISPER = "whisper"
    GOOGLE = "google"
    SPHINX = "sphinx"
    AZURE = "azure"
    AWS = "aws"
    IBM = "ibm"
    AUTO = "auto"


class WhisperModel(Enum):
    """Whisper model sizes."""
    TINY = "tiny"           # ~39 MB
    BASE = "base"           # ~74 MB
    SMALL = "small"         # ~244 MB
    MEDIUM = "medium"       # ~769 MB
    LARGE = "large"         # ~1550 MB


class LanguageCode(Enum):
    """Supported language codes."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    JAPANESE = "ja"
    CHINESE = "zh"
    KOREAN = "ko"
    AUTO = "auto"


class VADMode(Enum):
    """Voice Activity Detection modes."""
    DISABLED = "disabled"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"


@dataclass
class ASRConfig:
    """Speech recognition configuration."""
    backend: ASRBackend = ASRBackend.AUTO
    language: LanguageCode = LanguageCode.ENGLISH
    whisper_model: WhisperModel = WhisperModel.BASE
    sample_rate: int = 16000
    chunk_duration_ms: int = 30
    vad_mode: VADMode = VADMode.BASIC
    confidence_threshold: float = 0.5
    enable_punctuation: bool = True
    enable_profanity_filter: bool = False
    max_alternatives: int = 1
    timeout_seconds: float = 5.0
    phrase_time_limit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptionSegment:
    """Individual transcription segment."""
    text: str
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 1.0
    language: Optional[str] = None
    speaker_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    success: bool
    text: str = ""
    segments: List[TranscriptionSegment] = field(default_factory=list)
    language_detected: Optional[str] = None
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    backend_used: Optional[ASRBackend] = None
    model_used: str = ""
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VADResult:
    """Voice Activity Detection result."""
    has_speech: bool
    speech_segments: List[Tuple[float, float]] = field(default_factory=list)
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    method_used: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpeechRecognizer:
    """Advanced speech recognition engine for AVA."""
    
    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self.recognizer = None
        self.microphone = None
        self.whisper_model = None
        self.vad = None
        self.is_listening = False
        
        # Initialize components
        self._initialize_recognizer()
        self._initialize_whisper()
        self._initialize_vad()
        self._initialize_microphone()
    
    def _initialize_recognizer(self):
        """Initialize speech recognition engine."""
        if HAS_SPEECH_RECOGNITION:
            try:
                self.recognizer = sr.Recognizer()
                # Adjust for ambient noise
                self.recognizer.energy_threshold = 4000
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = 0.8
                self.recognizer.phrase_threshold = 0.3
                logger.info("SpeechRecognition engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize SpeechRecognition: {e}")
    
    def _initialize_whisper(self):
        """Initialize Whisper model."""
        if HAS_WHISPER and self.config.backend in [ASRBackend.WHISPER, ASRBackend.AUTO]:
            try:
                model_name = self.config.whisper_model.value
                self.whisper_model = whisper.load_model(model_name)
                logger.info(f"Whisper model '{model_name}' loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Whisper model: {e}")
                self.whisper_model = None
    
    def _initialize_vad(self):
        """Initialize Voice Activity Detection."""
        if HAS_WEBRTC_VAD and self.config.vad_mode != VADMode.DISABLED:
            try:
                # Map VAD modes to WebRTC VAD aggressiveness levels
                vad_level_map = {
                    VADMode.BASIC: 0,
                    VADMode.AGGRESSIVE: 2,
                    VADMode.VERY_AGGRESSIVE: 3
                }
                aggressiveness = vad_level_map.get(self.config.vad_mode, 1)
                self.vad = webrtcvad.Vad(aggressiveness)
                logger.info(f"WebRTC VAD initialized with aggressiveness {aggressiveness}")
            except Exception as e:
                logger.warning(f"Failed to initialize WebRTC VAD: {e}")
                self.vad = None
    
    def _initialize_microphone(self):
        """Initialize microphone for real-time recognition."""
        if HAS_PYAUDIO and HAS_SPEECH_RECOGNITION:
            try:
                self.microphone = sr.Microphone(sample_rate=self.config.sample_rate)
                # Calibrate microphone for ambient noise
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.info("Microphone initialized and calibrated")
            except Exception as e:
                logger.warning(f"Failed to initialize microphone: {e}")
                self.microphone = None
    
    async def transcribe_file(self, file_path: Union[str, Path]) -> TranscriptionResult:
        """Transcribe audio from file."""
        start_time = time.time()
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return TranscriptionResult(
                    success=False,
                    error=f"Audio file not found: {file_path}",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Choose backend based on configuration
            if self.config.backend == ASRBackend.WHISPER or (
                self.config.backend == ASRBackend.AUTO and self.whisper_model
            ):
                return await self._transcribe_with_whisper(file_path, start_time)
            elif self.config.backend == ASRBackend.GOOGLE or (
                self.config.backend == ASRBackend.AUTO and self.recognizer
            ):
                return await self._transcribe_with_google(file_path, start_time)
            else:
                return TranscriptionResult(
                    success=False,
                    error="No suitable ASR backend available",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
                
        except Exception as e:
            logger.error(f"Error transcribing file: {e}")
            return TranscriptionResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _transcribe_with_whisper(self, file_path: Path, start_time: float) -> TranscriptionResult:
        """Transcribe using Whisper model."""
        try:
            if not self.whisper_model:
                return TranscriptionResult(
                    success=False,
                    error="Whisper model not available",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Load audio file
            audio = whisper.load_audio(str(file_path))
            
            # Prepare options
            decode_options = {
                "language": self.config.language.value if self.config.language != LanguageCode.AUTO else None,
                "fp16": torch.cuda.is_available() if HAS_TORCH_AUDIO else False,
            }
            
            # Transcribe
            result = self.whisper_model.transcribe(audio, **decode_options)
            
            # Process segments
            segments = []
            for segment in result.get("segments", []):
                segments.append(TranscriptionSegment(
                    text=segment["text"].strip(),
                    start_time=segment["start"],
                    end_time=segment["end"],
                    confidence=segment.get("confidence", 1.0),
                    language=result.get("language")
                ))
            
            # Detect language if auto-detection was used
            detected_language = result.get("language")
            
            return TranscriptionResult(
                success=True,
                text=result["text"].strip(),
                segments=segments,
                language_detected=detected_language,
                confidence=np.mean([s.confidence for s in segments]) if segments else 1.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                backend_used=ASRBackend.WHISPER,
                model_used=self.config.whisper_model.value,
                metadata={
                    "whisper_info": {
                        "model": self.config.whisper_model.value,
                        "language": detected_language,
                        "num_segments": len(segments)
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return TranscriptionResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _transcribe_with_google(self, file_path: Path, start_time: float) -> TranscriptionResult:
        """Transcribe using Google Speech Recognition."""
        try:
            if not self.recognizer:
                return TranscriptionResult(
                    success=False,
                    error="SpeechRecognition not available",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Load audio file
            with sr.AudioFile(str(file_path)) as source:
                audio = self.recognizer.record(source)
            
            # Transcribe
            language = self.config.language.value if self.config.language != LanguageCode.AUTO else None
            
            try:
                text = self.recognizer.recognize_google(
                    audio,
                    language=language,
                    show_all=False
                )
                
                # Detect language if possible
                detected_language = None
                if HAS_LANGDETECT and self.config.language == LanguageCode.AUTO:
                    try:
                        detected_language = langdetect.detect(text)
                    except:
                        pass
                
                return TranscriptionResult(
                    success=True,
                    text=text,
                    segments=[TranscriptionSegment(
                        text=text,
                        confidence=0.8,  # Google doesn't provide confidence scores
                        language=detected_language
                    )],
                    language_detected=detected_language,
                    confidence=0.8,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    backend_used=ASRBackend.GOOGLE,
                    model_used="google_cloud"
                )
                
            except sr.UnknownValueError:
                return TranscriptionResult(
                    success=False,
                    error="Could not understand audio",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            except sr.RequestError as e:
                return TranscriptionResult(
                    success=False,
                    error=f"Google API error: {e}",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
                
        except Exception as e:
            logger.error(f"Google transcription failed: {e}")
            return TranscriptionResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def detect_voice_activity(self, audio_data: np.ndarray, sample_rate: int = 16000) -> VADResult:
        """Detect voice activity in audio data."""
        start_time = time.time()
        
        try:
            if self.config.vad_mode == VADMode.DISABLED:
                return VADResult(
                    has_speech=True,  # Assume speech when VAD is disabled
                    confidence=1.0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    method_used="disabled"
                )
            
            if self.vad and HAS_WEBRTC_VAD:
                return await self._detect_vad_webrtc(audio_data, sample_rate, start_time)
            else:
                return await self._detect_vad_energy(audio_data, sample_rate, start_time)
                
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return VADResult(
                has_speech=True,  # Default to assuming speech on error
                confidence=0.5,
                processing_time_ms=(time.time() - start_time) * 1000,
                method_used="error_fallback"
            )
    
    async def _detect_vad_webrtc(self, audio_data: np.ndarray, sample_rate: int, start_time: float) -> VADResult:
        """Use WebRTC VAD for voice activity detection."""
        try:
            # Convert to required format (16-bit PCM)
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
            
            # WebRTC VAD requires specific sample rates and frame sizes
            if sample_rate not in [8000, 16000, 32000, 48000]:
                sample_rate = 16000  # Default fallback
            
            frame_duration = 30  # ms
            frame_size = int(sample_rate * frame_duration / 1000)
            
            speech_frames = 0
            total_frames = 0
            speech_segments = []
            current_speech_start = None
            
            # Process audio in frames
            for i in range(0, len(audio_data) - frame_size, frame_size):
                frame = audio_data[i:i + frame_size].tobytes()
                
                try:
                    is_speech = self.vad.is_speech(frame, sample_rate)
                    
                    if is_speech:
                        speech_frames += 1
                        if current_speech_start is None:
                            current_speech_start = i / sample_rate
                    else:
                        if current_speech_start is not None:
                            speech_segments.append((current_speech_start, i / sample_rate))
                            current_speech_start = None
                    
                    total_frames += 1
                    
                except Exception as e:
                    logger.debug(f"VAD frame processing error: {e}")
                    continue
            
            # Close final speech segment if needed
            if current_speech_start is not None:
                speech_segments.append((current_speech_start, len(audio_data) / sample_rate))
            
            has_speech = speech_frames > 0
            confidence = speech_frames / total_frames if total_frames > 0 else 0.0
            
            return VADResult(
                has_speech=has_speech,
                speech_segments=speech_segments,
                confidence=confidence,
                processing_time_ms=(time.time() - start_time) * 1000,
                method_used="webrtc_vad",
                metadata={
                    "speech_frames": speech_frames,
                    "total_frames": total_frames,
                    "frame_duration_ms": frame_duration
                }
            )
            
        except Exception as e:
            logger.warning(f"WebRTC VAD failed: {e}")
            return await self._detect_vad_energy(audio_data, sample_rate, start_time)
    
    async def _detect_vad_energy(self, audio_data: np.ndarray, sample_rate: int, start_time: float) -> VADResult:
        """Simple energy-based voice activity detection."""
        try:
            # Convert to float if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Calculate RMS energy
            frame_size = int(sample_rate * 0.03)  # 30ms frames
            energies = []
            
            for i in range(0, len(audio_data) - frame_size, frame_size):
                frame = audio_data[i:i + frame_size]
                energy = np.sqrt(np.mean(frame ** 2))
                energies.append(energy)
            
            if not energies:
                return VADResult(
                    has_speech=False,
                    confidence=0.0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    method_used="energy_based"
                )
            
            # Determine threshold (adaptive)
            max_energy = max(energies)
            threshold = max_energy * 0.1  # 10% of max energy
            
            # Find speech segments
            speech_segments = []
            current_speech_start = None
            
            for i, energy in enumerate(energies):
                time_position = i * 0.03  # 30ms frames
                
                if energy > threshold:
                    if current_speech_start is None:
                        current_speech_start = time_position
                else:
                    if current_speech_start is not None:
                        speech_segments.append((current_speech_start, time_position))
                        current_speech_start = None
            
            # Close final segment if needed
            if current_speech_start is not None:
                speech_segments.append((current_speech_start, len(energies) * 0.03))
            
            has_speech = len(speech_segments) > 0
            confidence = len([e for e in energies if e > threshold]) / len(energies)
            
            return VADResult(
                has_speech=has_speech,
                speech_segments=speech_segments,
                confidence=confidence,
                processing_time_ms=(time.time() - start_time) * 1000,
                method_used="energy_based",
                metadata={
                    "threshold": threshold,
                    "max_energy": max_energy,
                    "num_frames": len(energies)
                }
            )
            
        except Exception as e:
            logger.error(f"Energy-based VAD failed: {e}")
            return VADResult(
                has_speech=True,
                confidence=0.5,
                processing_time_ms=(time.time() - start_time) * 1000,
                method_used="fallback"
            )
    
    async def listen_continuously(self) -> AsyncGenerator[TranscriptionResult, None]:
        """Continuously listen and transcribe speech."""
        if not self.microphone or not self.recognizer:
            yield TranscriptionResult(
                success=False,
                error="Microphone or recognizer not available"
            )
            return
        
        self.is_listening = True
        logger.info("Starting continuous listening...")
        
        try:
            while self.is_listening:
                try:
                    # Listen for audio
                    with self.microphone as source:
                        # Listen for phrase with timeout
                        audio = self.recognizer.listen(
                            source,
                            timeout=self.config.timeout_seconds,
                            phrase_time_limit=self.config.phrase_time_limit
                        )
                    
                    # Transcribe the audio
                    start_time = time.time()
                    
                    try:
                        # Choose backend for real-time transcription
                        if self.config.backend == ASRBackend.GOOGLE or (
                            self.config.backend == ASRBackend.AUTO and not self.whisper_model
                        ):
                            language = self.config.language.value if self.config.language != LanguageCode.AUTO else None
                            text = self.recognizer.recognize_google(audio, language=language)
                            
                            result = TranscriptionResult(
                                success=True,
                                text=text,
                                segments=[TranscriptionSegment(text=text, confidence=0.8)],
                                confidence=0.8,
                                processing_time_ms=(time.time() - start_time) * 1000,
                                backend_used=ASRBackend.GOOGLE,
                                model_used="google_cloud"
                            )
                        else:
                            # For Whisper, save to temp file and transcribe
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                                temp_path = temp_file.name
                                
                                # Save audio to temp file
                                with wave.open(temp_path, 'wb') as wav_file:
                                    wav_file.setnchannels(1)
                                    wav_file.setsampwidth(2)
                                    wav_file.setframerate(self.config.sample_rate)
                                    wav_file.writeframes(audio.get_wav_data())
                                
                                # Transcribe with Whisper
                                result = await self.transcribe_file(temp_path)
                                
                                # Clean up temp file
                                os.unlink(temp_path)
                        
                        yield result
                        
                    except sr.UnknownValueError:
                        yield TranscriptionResult(
                            success=False,
                            error="Could not understand audio",
                            processing_time_ms=(time.time() - start_time) * 1000
                        )
                    except sr.RequestError as e:
                        yield TranscriptionResult(
                            success=False,
                            error=f"API error: {e}",
                            processing_time_ms=(time.time() - start_time) * 1000
                        )
                        
                except sr.WaitTimeoutError:
                    # Timeout is normal, continue listening
                    continue
                except Exception as e:
                    logger.error(f"Listening error: {e}")
                    yield TranscriptionResult(
                        success=False,
                        error=str(e)
                    )
                    break
                    
        finally:
            self.is_listening = False
            logger.info("Stopped continuous listening")
    
    def stop_listening(self):
        """Stop continuous listening."""
        self.is_listening = False
        logger.info("Stopping continuous listening...")
    
    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Function calling interface for AVA agent compatibility.
        
        Args:
            operation: The operation to perform
            **kwargs: Operation parameters
            
        Returns:
            Dictionary with operation results
        """
        try:
            if operation == "transcribe_file":
                file_path = kwargs.get("file_path")
                if not file_path:
                    return {"error": "file_path parameter required"}
                
                result = asyncio.run(self.transcribe_file(file_path))
                return {
                    "success": result.success,
                    "text": result.text,
                    "language": result.language_detected,
                    "confidence": result.confidence,
                    "segments": len(result.segments),
                    "processing_time_ms": result.processing_time_ms,
                    "backend": result.backend_used.value if result.backend_used else None,
                    "error": result.error
                }
            
            elif operation == "transcribe_and_detect_language":
                file_path = kwargs.get("file_path")
                if not file_path:
                    return {"error": "file_path parameter required"}
                
                # Temporarily set to auto-detect language
                original_language = self.config.language
                self.config.language = LanguageCode.AUTO
                
                try:
                    result = asyncio.run(self.transcribe_file(file_path))
                    return {
                        "success": result.success,
                        "text": result.text,
                        "language_detected": result.language_detected,
                        "confidence": result.confidence,
                        "processing_time_ms": result.processing_time_ms,
                        "error": result.error
                    }
                finally:
                    self.config.language = original_language
            
            elif operation == "detect_voice_activity":
                file_path = kwargs.get("file_path")
                if not file_path:
                    return {"error": "file_path parameter required"}
                
                # Load audio file (simplified)
                try:
                    import wave
                    with wave.open(file_path, 'rb') as wav_file:
                        sample_rate = wav_file.getframerate()
                        frames = wav_file.readframes(-1)
                        audio_data = np.frombuffer(frames, dtype=np.int16)
                    
                    vad_result = asyncio.run(self.detect_voice_activity(audio_data, sample_rate))
                    
                    return {
                        "success": True,
                        "has_speech": vad_result.has_speech,
                        "confidence": vad_result.confidence,
                        "speech_segments": len(vad_result.speech_segments),
                        "processing_time_ms": vad_result.processing_time_ms,
                        "method": vad_result.method_used
                    }
                except Exception as e:
                    return {"error": f"Failed to process audio file: {e}"}
            
            else:
                return {"error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            logger.error(f"Error in speech recognition operation '{operation}': {e}")
            return {"error": str(e)}
    
    def __del__(self):
        """Cleanup resources."""
        self.stop_listening()


def test_speech_recognition():
    """Test speech recognition functionality."""
    print("Testing Speech Recognition Module...")
    
    # Test basic initialization
    recognizer = SpeechRecognizer()
    print(f"✓ Speech recognizer initialized")
    print(f"  - Whisper available: {recognizer.whisper_model is not None}")
    print(f"  - SpeechRecognition available: {recognizer.recognizer is not None}")
    print(f"  - Microphone available: {recognizer.microphone is not None}")
    print(f"  - VAD available: {recognizer.vad is not None}")
    
    # Test function calling interface
    result = recognizer.run("transcribe_file", file_path="nonexistent.wav")
    print(f"✓ Function calling interface tested (expected error): {result.get('error', 'No error')}")
    
    # Test configuration
    config = ASRConfig(
        backend=ASRBackend.WHISPER,
        language=LanguageCode.ENGLISH,
        whisper_model=WhisperModel.BASE
    )
    recognizer_configured = SpeechRecognizer(config)
    print(f"✓ Speech recognizer with custom config initialized")
    
    print("Speech recognition module tests completed!")


async def main():
    """Async main function for testing."""
    print("AVA Speech Recognition Module")
    print("=" * 50)
    
    # Test speech recognition
    test_speech_recognition()
    
    # Create recognizer instance
    recognizer = SpeechRecognizer()
    
    print("\nTesting speech recognition capabilities...")
    print(f"Available backends: {[backend.value for backend in ASRBackend]}")
    print(f"Available models: {[model.value for model in WhisperModel]}")
    print(f"Supported languages: {[lang.value for lang in LanguageCode]}")
    
    # Cleanup
    del recognizer
    print("Speech recognition testing completed!")


if __name__ == "__main__":
    asyncio.run(main())
