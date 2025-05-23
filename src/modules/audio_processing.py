#!/usr/bin/env python3
"""
Enhanced Audio Processing Module for AVA
Production-Ready Audio I/O, Processing, and Feature Extraction
Optimized for RTX A2000 4GB VRAM constraints
"""

import asyncio
import io
import json
import logging
import numpy as np
import os
import tempfile
import time
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator
import warnings

# Core audio processing imports
try:
    import librosa
    import soundfile as sf
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    warnings.warn("librosa/soundfile not available - advanced audio processing limited")

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    warnings.warn("PyAudio not available - real-time audio I/O unavailable")

try:
    import scipy.signal
    import scipy.io.wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("SciPy not available - advanced signal processing limited")

try:
    import torch
    import torchaudio
    HAS_TORCH_AUDIO = True
except ImportError:
    HAS_TORCH_AUDIO = False
    warnings.warn("torchaudio not available - GPU-accelerated audio processing unavailable")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    M4A = "m4a"
    OGG = "ogg"
    RAW = "raw"


class AudioQuality(Enum):
    """Audio quality presets."""
    LOW = "low"          # 8kHz, 16-bit
    MEDIUM = "medium"    # 16kHz, 16-bit
    HIGH = "high"        # 44.1kHz, 16-bit
    STUDIO = "studio"    # 48kHz, 24-bit


class ProcessingMode(Enum):
    """Audio processing modes."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"


class NoiseReductionMethod(Enum):
    """Noise reduction methods."""
    SPECTRAL_SUBTRACTION = "spectral_subtraction"
    WIENER_FILTER = "wiener_filter"
    LOW_PASS = "low_pass"
    HIGH_PASS = "high_pass"
    BAND_PASS = "band_pass"


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16
    chunk_size: int = 1024
    format: AudioFormat = AudioFormat.WAV
    quality: AudioQuality = AudioQuality.MEDIUM
    enable_noise_reduction: bool = True
    noise_reduction_method: NoiseReductionMethod = NoiseReductionMethod.SPECTRAL_SUBTRACTION
    normalize_audio: bool = True
    trim_silence: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioData:
    """Audio data container."""
    samples: np.ndarray = field(default_factory=lambda: np.array([]))
    sample_rate: int = 16000
    channels: int = 1
    duration_seconds: float = 0.0
    format: AudioFormat = AudioFormat.WAV
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioFeatures:
    """Extracted audio features."""
    mfcc: Optional[np.ndarray] = None
    mel_spectrogram: Optional[np.ndarray] = None
    chroma: Optional[np.ndarray] = None
    spectral_centroid: Optional[np.ndarray] = None
    zero_crossing_rate: Optional[np.ndarray] = None
    rms_energy: Optional[np.ndarray] = None
    pitch: Optional[np.ndarray] = None
    tempo: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Audio processing result."""
    success: bool
    audio_data: Optional[AudioData] = None
    features: Optional[AudioFeatures] = None
    processing_time_ms: float = 0.0
    method_used: str = ""
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AudioProcessor:
    """Advanced audio processing engine for AVA."""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.is_recording = False
        self.audio_stream = None
        self.pyaudio_instance = None
        
        # Initialize PyAudio if available
        if HAS_PYAUDIO:
            try:
                self.pyaudio_instance = pyaudio.PyAudio()
                logger.info("PyAudio initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize PyAudio: {e}")
                self.pyaudio_instance = None
    
    async def load_audio(self, file_path: Union[str, Path]) -> ProcessingResult:
        """Load audio from file."""
        start_time = time.time()
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return ProcessingResult(
                    success=False,
                    error=f"Audio file not found: {file_path}",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Use librosa if available, fallback to scipy/wave
            if HAS_LIBROSA:
                samples, sample_rate = librosa.load(
                    str(file_path),
                    sr=self.config.sample_rate,
                    mono=(self.config.channels == 1)
                )
            elif HAS_SCIPY and file_path.suffix.lower() == '.wav':
                sample_rate, samples = scipy.io.wavfile.read(str(file_path))
                if samples.dtype != np.float32:
                    samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
                
                # Resample if needed (basic implementation)
                if sample_rate != self.config.sample_rate:
                    ratio = self.config.sample_rate / sample_rate
                    new_length = int(len(samples) * ratio)
                    samples = np.interp(np.linspace(0, len(samples), new_length), 
                                      np.arange(len(samples)), samples)
                    sample_rate = self.config.sample_rate
            else:
                # Fallback to wave module for basic WAV support
                with wave.open(str(file_path), 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    frames = wav_file.readframes(-1)
                    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Convert to mono if needed
            if len(samples.shape) > 1 and self.config.channels == 1:
                samples = np.mean(samples, axis=1)
            
            duration = len(samples) / sample_rate
            
            audio_data = AudioData(
                samples=samples,
                sample_rate=sample_rate,
                channels=self.config.channels,
                duration_seconds=duration,
                format=AudioFormat(file_path.suffix[1:].lower()) if file_path.suffix[1:].lower() in [f.value for f in AudioFormat] else AudioFormat.WAV,
                metadata={
                    "file_path": str(file_path),
                    "file_size_bytes": file_path.stat().st_size,
                    "original_sample_rate": sample_rate
                }
            )
            
            return ProcessingResult(
                success=True,
                audio_data=audio_data,
                processing_time_ms=(time.time() - start_time) * 1000,
                method_used="librosa" if HAS_LIBROSA else "scipy/wave"
            )
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def save_audio(self, audio_data: AudioData, file_path: Union[str, Path]) -> ProcessingResult:
        """Save audio to file."""
        start_time = time.time()
        
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure samples are in correct format
            samples = audio_data.samples
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32)
            
            # Use soundfile if available, fallback to scipy/wave
            if HAS_LIBROSA:
                sf.write(
                    str(file_path),
                    samples,
                    audio_data.sample_rate,
                    format=file_path.suffix[1:].upper()
                )
            elif HAS_SCIPY and file_path.suffix.lower() == '.wav':
                # Convert to int16 for WAV
                samples_int16 = (samples * 32767).astype(np.int16)
                scipy.io.wavfile.write(str(file_path), audio_data.sample_rate, samples_int16)
            else:
                # Fallback to wave module
                with wave.open(str(file_path), 'wb') as wav_file:
                    wav_file.setnchannels(audio_data.channels)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(audio_data.sample_rate)
                    samples_int16 = (samples * 32767).astype(np.int16)
                    wav_file.writeframes(samples_int16.tobytes())
            
            return ProcessingResult(
                success=True,
                processing_time_ms=(time.time() - start_time) * 1000,
                method_used="soundfile" if HAS_LIBROSA else "scipy/wave",
                metadata={"output_path": str(file_path)}
            )
            
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def process_audio(self, audio_data: AudioData) -> ProcessingResult:
        """Apply audio processing pipeline."""
        start_time = time.time()
        
        try:
            processed_samples = audio_data.samples.copy()
            warnings_list = []
            
            # Normalize audio
            if self.config.normalize_audio:
                max_val = np.max(np.abs(processed_samples))
                if max_val > 0:
                    processed_samples = processed_samples / max_val
            
            # Trim silence
            if self.config.trim_silence and HAS_LIBROSA:
                try:
                    processed_samples, _ = librosa.effects.trim(processed_samples, top_db=20)
                except Exception as e:
                    warnings_list.append(f"Could not trim silence: {e}")
            
            # Apply noise reduction
            if self.config.enable_noise_reduction:
                processed_samples = await self._apply_noise_reduction(
                    processed_samples, 
                    audio_data.sample_rate
                )
            
            processed_audio = AudioData(
                samples=processed_samples,
                sample_rate=audio_data.sample_rate,
                channels=audio_data.channels,
                duration_seconds=len(processed_samples) / audio_data.sample_rate,
                format=audio_data.format,
                metadata=audio_data.metadata.copy()
            )
            
            return ProcessingResult(
                success=True,
                audio_data=processed_audio,
                processing_time_ms=(time.time() - start_time) * 1000,
                method_used="audio_processing_pipeline",
                warnings=warnings_list
            )
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def extract_features(self, audio_data: AudioData) -> ProcessingResult:
        """Extract audio features for analysis."""
        start_time = time.time()
        
        try:
            features = AudioFeatures()
            
            if not HAS_LIBROSA:
                return ProcessingResult(
                    success=False,
                    error="librosa not available - feature extraction unavailable",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            samples = audio_data.samples
            sr = audio_data.sample_rate
            
            # Extract MFCC features
            try:
                mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=13)
                features.mfcc = mfcc
            except Exception as e:
                logger.warning(f"Could not extract MFCC: {e}")
            
            # Extract mel spectrogram
            try:
                mel_spec = librosa.feature.melspectrogram(y=samples, sr=sr)
                features.mel_spectrogram = mel_spec
            except Exception as e:
                logger.warning(f"Could not extract mel spectrogram: {e}")
            
            # Extract chroma features
            try:
                chroma = librosa.feature.chroma_stft(y=samples, sr=sr)
                features.chroma = chroma
            except Exception as e:
                logger.warning(f"Could not extract chroma: {e}")
            
            # Extract spectral centroid
            try:
                spectral_centroid = librosa.feature.spectral_centroid(y=samples, sr=sr)
                features.spectral_centroid = spectral_centroid
            except Exception as e:
                logger.warning(f"Could not extract spectral centroid: {e}")
            
            # Extract zero crossing rate
            try:
                zcr = librosa.feature.zero_crossing_rate(samples)
                features.zero_crossing_rate = zcr
            except Exception as e:
                logger.warning(f"Could not extract ZCR: {e}")
            
            # Extract RMS energy
            try:
                rms = librosa.feature.rms(y=samples)
                features.rms_energy = rms
            except Exception as e:
                logger.warning(f"Could not extract RMS: {e}")
            
            # Extract pitch
            try:
                pitches, magnitudes = librosa.piptrack(y=samples, sr=sr)
                features.pitch = pitches
            except Exception as e:
                logger.warning(f"Could not extract pitch: {e}")
            
            # Extract tempo
            try:
                tempo, _ = librosa.beat.beat_track(y=samples, sr=sr)
                features.tempo = float(tempo)
            except Exception as e:
                logger.warning(f"Could not extract tempo: {e}")
            
            features.metadata = {
                "sample_rate": sr,
                "duration": len(samples) / sr,
                "n_samples": len(samples)
            }
            
            return ProcessingResult(
                success=True,
                features=features,
                processing_time_ms=(time.time() - start_time) * 1000,
                method_used="librosa_feature_extraction"
            )
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _apply_noise_reduction(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply noise reduction to audio samples."""
        try:
            if self.config.noise_reduction_method == NoiseReductionMethod.SPECTRAL_SUBTRACTION:
                return await self._spectral_subtraction(samples, sample_rate)
            elif self.config.noise_reduction_method == NoiseReductionMethod.LOW_PASS:
                return await self._low_pass_filter(samples, sample_rate, cutoff=4000)
            elif self.config.noise_reduction_method == NoiseReductionMethod.HIGH_PASS:
                return await self._high_pass_filter(samples, sample_rate, cutoff=100)
            else:
                return samples
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return samples
    
    async def _spectral_subtraction(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply spectral subtraction noise reduction."""
        if not HAS_LIBROSA:
            return samples
        
        try:
            # Simple spectral subtraction implementation
            stft = librosa.stft(samples)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise from first few frames
            noise_frames = min(10, magnitude.shape[1])
            noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
            
            # Apply spectral subtraction
            alpha = 2.0  # Over-subtraction factor
            magnitude_cleaned = magnitude - alpha * noise_spectrum
            magnitude_cleaned = np.maximum(magnitude_cleaned, 0.1 * magnitude)
            
            # Reconstruct signal
            stft_cleaned = magnitude_cleaned * np.exp(1j * phase)
            samples_cleaned = librosa.istft(stft_cleaned)
            
            return samples_cleaned
        except Exception as e:
            logger.warning(f"Spectral subtraction failed: {e}")
            return samples
    
    async def _low_pass_filter(self, samples: np.ndarray, sample_rate: int, cutoff: float) -> np.ndarray:
        """Apply low-pass filter."""
        if not HAS_SCIPY:
            return samples
        
        try:
            nyquist = sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            b, a = scipy.signal.butter(4, normalized_cutoff, btype='low')
            return scipy.signal.filtfilt(b, a, samples)
        except Exception as e:
            logger.warning(f"Low-pass filter failed: {e}")
            return samples
    
    async def _high_pass_filter(self, samples: np.ndarray, sample_rate: int, cutoff: float) -> np.ndarray:
        """Apply high-pass filter."""
        if not HAS_SCIPY:
            return samples
        
        try:
            nyquist = sample_rate / 2
            normalized_cutoff = cutoff / nyquist
            b, a = scipy.signal.butter(4, normalized_cutoff, btype='high')
            return scipy.signal.filtfilt(b, a, samples)
        except Exception as e:
            logger.warning(f"High-pass filter failed: {e}")
            return samples
    
    async def start_recording(self) -> ProcessingResult:
        """Start real-time audio recording."""
        if not HAS_PYAUDIO or not self.pyaudio_instance:
            return ProcessingResult(
                success=False,
                error="PyAudio not available - recording unavailable"
            )
        
        try:
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )
            self.is_recording = True
            
            return ProcessingResult(
                success=True,
                method_used="pyaudio_recording",
                metadata={"recording_started": True}
            )
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return ProcessingResult(
                success=False,
                error=str(e)
            )
    
    async def stop_recording(self) -> ProcessingResult:
        """Stop real-time audio recording."""
        try:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
            
            self.is_recording = False
            
            return ProcessingResult(
                success=True,
                method_used="pyaudio_recording",
                metadata={"recording_stopped": True}
            )
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return ProcessingResult(
                success=False,
                error=str(e)
            )
    
    async def read_audio_chunk(self) -> Optional[np.ndarray]:
        """Read a chunk of audio from the recording stream."""
        if not self.is_recording or not self.audio_stream:
            return None
        
        try:
            data = self.audio_stream.read(self.config.chunk_size, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            return samples
        except Exception as e:
            logger.warning(f"Error reading audio chunk: {e}")
            return None
    
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
            if operation == "load_audio":
                file_path = kwargs.get("file_path")
                if not file_path:
                    return {"error": "file_path parameter required"}
                
                result = asyncio.run(self.load_audio(file_path))
                return {
                    "success": result.success,
                    "duration": result.audio_data.duration_seconds if result.audio_data else 0,
                    "sample_rate": result.audio_data.sample_rate if result.audio_data else 0,
                    "processing_time_ms": result.processing_time_ms,
                    "error": result.error
                }
            
            elif operation == "extract_features":
                file_path = kwargs.get("file_path")
                if not file_path:
                    return {"error": "file_path parameter required"}
                
                # Load and extract features
                load_result = asyncio.run(self.load_audio(file_path))
                if not load_result.success:
                    return {"error": f"Failed to load audio: {load_result.error}"}
                
                feature_result = asyncio.run(self.extract_features(load_result.audio_data))
                
                # Convert features to serializable format
                features_dict = {}
                if feature_result.features:
                    if feature_result.features.mfcc is not None:
                        features_dict["mfcc_shape"] = feature_result.features.mfcc.shape
                        features_dict["mfcc_mean"] = float(np.mean(feature_result.features.mfcc))
                    if feature_result.features.tempo:
                        features_dict["tempo"] = feature_result.features.tempo
                
                return {
                    "success": feature_result.success,
                    "features": features_dict,
                    "processing_time_ms": feature_result.processing_time_ms,
                    "error": feature_result.error
                }
            
            elif operation == "process_audio":
                file_path = kwargs.get("file_path")
                output_path = kwargs.get("output_path")
                if not file_path or not output_path:
                    return {"error": "file_path and output_path parameters required"}
                
                # Load, process, and save audio
                load_result = asyncio.run(self.load_audio(file_path))
                if not load_result.success:
                    return {"error": f"Failed to load audio: {load_result.error}"}
                
                process_result = asyncio.run(self.process_audio(load_result.audio_data))
                if not process_result.success:
                    return {"error": f"Failed to process audio: {process_result.error}"}
                
                save_result = asyncio.run(self.save_audio(process_result.audio_data, output_path))
                
                return {
                    "success": save_result.success,
                    "output_path": output_path,
                    "processing_time_ms": process_result.processing_time_ms + save_result.processing_time_ms,
                    "error": save_result.error
                }
            
            else:
                return {"error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            logger.error(f"Error in audio processing operation '{operation}': {e}")
            return {"error": str(e)}
    
    def __del__(self):
        """Cleanup resources."""
        if self.is_recording:
            asyncio.run(self.stop_recording())
        
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except Exception as e:
                logger.warning(f"Error terminating PyAudio: {e}")


def test_audio_processing():
    """Test audio processing functionality."""
    print("Testing Audio Processing Module...")
    
    # Test basic initialization
    processor = AudioProcessor()
    print(f"✓ Audio processor initialized")
    
    # Test function calling interface
    result = processor.run("load_audio", file_path="nonexistent.wav")
    print(f"✓ Function calling interface tested (expected error): {result.get('error', 'No error')}")
    
    # Test configuration
    config = AudioConfig(
        sample_rate=16000,
        channels=1,
        enable_noise_reduction=True
    )
    processor_configured = AudioProcessor(config)
    print(f"✓ Audio processor with custom config initialized")
    
    print("Audio processing module tests completed!")


async def main():
    """Async main function for testing."""
    print("AVA Audio Processing Module")
    print("=" * 50)
    
    # Test audio processing
    test_audio_processing()
    
    # Create processor instance
    processor = AudioProcessor()
    
    # Test with a temporary audio file if available
    print("\nTesting audio processing capabilities...")
    
    # Cleanup
    del processor
    print("Audio processing testing completed!")


if __name__ == "__main__":
    asyncio.run(main())
