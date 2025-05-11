from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union, Literal
import uuid
from datetime import datetime


# Import shared schemas
# from .schemas import (
#     TranscriptionRequest, 
#     TranscriptionResult,
#     AudioBlobSource,
#     AudioUploadSource,
#     ModelConfig,
#     ProcessingConfig
# )

# Add this line
from storage.azure_storage import AzureBlobStorage

from fastapi import APIRouter, HTTPException, BackgroundTasks, Response, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any, BinaryIO
import uuid
import os
import torch
import logging
from datetime import datetime
import pathlib
import json
import base64
import time
import tempfile
import io
import math
from pydub import AudioSegment



import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from multiprocessing import Process, Queue
import queue
import asyncio

router = APIRouter()
logger = logging.getLogger(__name__)


# ==============================================
# ORIGINAL SCHEMA CLASSES (DO NOT REMOVE THESE)
# ==============================================

# Base models
class AudioBase(BaseModel):
    """Base class for audio sources"""
    source_type: str = Field(..., description="Type of audio source")

class ModelOptions(BaseModel):
    """Options for transcription model behavior"""
    diarize: bool = Field(True, description="Enable speaker diarization")
    word_timestamps: bool = Field(True, description="Include word-level timestamps")

class ModelConfig(BaseModel):
    """Configuration for the transcription model"""
    name: str = Field("tiny", description="Model name (tiny, base, small, medium, large)")
    language: Optional[str] = Field("auto", description="Language code or 'auto' for detection")
    options: ModelOptions = Field(default_factory=ModelOptions, description="Model options")

class ProcessingConfig(BaseModel):
    """Configuration for audio processing"""
    batch_size: int = Field(8, description="Batch size for model inference")
    compute_type: str = Field("float16", description="Compute type (float16, float32, etc.)")

# Request models
class AudioBlobSource(AudioBase):
    """Audio source from a blob storage"""
    source_type: Literal["azure_blob"] = "azure_blob"
    path: str = Field(..., description="Path in the blob storage (container/filename)")

class AudioUploadSource(AudioBase):
    """Audio source from a direct upload"""
    source_type: Literal["upload"] = "upload"
    file_name: str = Field(..., description="Original filename")
    file_size_mb: float = Field(..., description="File size in MB")

class AudioStreamSource(AudioBase):
    """Audio source from a WebSocket stream"""
    source_type: Literal["stream"] = "stream"
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")

class TranscriptionRequest(BaseModel):
    """Request for audio transcription (either batch or streaming)"""
    mode: Literal["batch", "realtime"] = Field("batch", description="Processing mode: batch or realtime")
    audio: Union[AudioBlobSource, AudioUploadSource] = Field(..., description="Audio source")
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")
    config: ProcessingConfig = Field(default_factory=ProcessingConfig, description="Processing configuration")

# WebSocket models
class WebSocketConfig(BaseModel):
    """Configuration for WebSocket-based streaming transcription"""
    # Basic configuration
    model: str = Field("tiny", description="Model name (tiny, base, small, medium, large)")
    language: str = Field("auto", description="Language code or 'auto' for detection")
    compute_type: str = Field("float16", description="Compute type for model inference")
    device: str = Field("cuda", description="Device to use (cuda, cpu)")
    
    # Realtime transcription parameters
    enable_realtime_transcription: bool = Field(True, description="Enable realtime transcription")
    use_main_model_for_realtime: bool = Field(False, description="Use main model for realtime transcription")
    realtime_model_type: str = Field("tiny", description="Model type for realtime transcription")
    realtime_processing_pause: float = Field(0.2, description="Pause between realtime processing in seconds")
    init_realtime_after_seconds: float = Field(0.2, description="Start realtime transcription after this many seconds")
    realtime_batch_size: int = Field(16, description="Batch size for realtime model")
    
    # Formatting options
    ensure_sentence_starting_uppercase: bool = Field(True, description="Ensure sentences start with uppercase")
    ensure_sentence_ends_with_period: bool = Field(True, description="Ensure sentences end with a period")
    
    # Voice activation parameters
    silero_sensitivity: float = Field(0.4, description="Silero VAD sensitivity")
    silero_use_onnx: bool = Field(False, description="Use ONNX runtime for Silero VAD")
    silero_deactivity_detection: bool = Field(False, description="Enable deactivity detection")
    webrtc_sensitivity: int = Field(3, description="WebRTC VAD sensitivity")
    post_speech_silence_duration: float = Field(0.6, description="Post-speech silence duration in seconds")
    min_length_of_recording: float = Field(0.5, description="Minimum length of recording in seconds")
    min_gap_between_recordings: float = Field(0.0, description="Minimum gap between recordings in seconds")
    pre_recording_buffer_duration: float = Field(1.0, description="Pre-recording buffer duration in seconds")

class WebSocketCommand(BaseModel):
    """Base class for WebSocket commands"""
    command: str = Field(..., description="Command name")
    session_id: Optional[str] = Field(None, description="Session ID for existing connections")

class WebSocketConnect(WebSocketCommand):
    """Command to establish a WebSocket connection"""
    command: Literal["connect"] = "connect"
    config: WebSocketConfig = Field(default_factory=WebSocketConfig, description="Connection configuration")

class WebSocketAudioData(WebSocketCommand):
    """Command to send audio data over WebSocket"""
    command: Literal["send_audio"] = "send_audio"
    audio: str = Field(..., description="Base64-encoded audio data")
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")

class WebSocketSimpleCommand(WebSocketCommand):
    """Simple WebSocket command without additional parameters"""
    command: Literal["start_listening", "stop_listening", "get_transcript", "wakeup", "abort", "disconnect"]

# Response models
class TranscriptionSegment(BaseModel):
    """Segment of transcribed speech"""
    id: int = Field(..., description="Segment ID")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text")
    speaker: Optional[str] = Field(None, description="Speaker ID if diarization is enabled")
    words: Optional[List[Dict[str, Any]]] = Field(None, description="Word-level timestamps if enabled")

class TranscriptionMetadata(BaseModel):
    """Metadata about the transcription process"""
    job_id: str = Field(..., description="Unique job ID")
    status: str = Field(..., description="Job status (processing, completed, error)")
    created_at: datetime = Field(..., description="Creation timestamp")
    merged_at: Optional[datetime] = Field(None, description="Timestamp when results were merged")
    total_processing_time_seconds: Optional[float] = Field(None, description="Total processing time in seconds")
    workers_used: Optional[int] = Field(None, description="Number of worker processes used")
    error: Optional[str] = Field(None, description="Error message if job failed")

class TranscriptionResult(BaseModel):
    """Complete transcription result"""
    metadata: TranscriptionMetadata = Field(..., description="Job metadata")
    model_info: Dict[str, Any] = Field(..., description="Information about the model used")
    total_chunks: int = Field(..., description="Total number of chunks processed")
    total_duration: float = Field(..., description="Total audio duration in seconds")
    num_speakers_detected: int = Field(..., description="Number of speakers detected")
    transcript: str = Field(..., description="Complete transcript text")
    segments: List[TranscriptionSegment] = Field(..., description="Detailed segments")

class WebSocketEvent(BaseModel):
    """Base class for WebSocket events"""
    event: str = Field(..., description="Event type")
    session_id: str = Field(..., description="Session ID")
    data: Optional[Dict[str, Any]] = Field(None, description="Event data")

class ErrorResponse(BaseModel):
    """Error response"""
    detail: str = Field(..., description="Error detail message")

# ==============================================
# NEW UNIFIED SCHEMA (Deepgram-style API)
# ==============================================

# Base shared models
class BaseModelConfig(BaseModel):
    """Base configuration for transcription models - shared across all modes"""
    name: str = Field("tiny", description="Model name (tiny, base, small, medium, large)")
    language: str = Field("auto", description="Language code or 'auto' for detection")
    compute_type: str = Field("float16", description="Compute type (float16, float32)")
    device: str = Field("cuda", description="Device to use (cuda, cpu)")

# Batch-specific models
class BatchDiarizationOptions(BaseModel):
    """Options specific to batch diarization"""
    diarize: bool = Field(True, description="Enable speaker diarization")
    word_timestamps: bool = Field(True, description="Include word-level timestamps") 
    num_speakers: Optional[int] = Field(None, description="Expected number of speakers (None for auto)")

class BatchProcessingConfig(BaseModel):
    """Configuration specific to batch processing"""
    batch_size: int = Field(8, description="Batch size for model inference")
    beam_size: int = Field(5, description="Beam size for decoding")
    initial_prompt: Optional[str] = Field(None, description="Initial prompt for transcription")

# Real-time specific models
class RealtimeVadConfig(BaseModel):
    """Voice Activity Detection settings for real-time processing"""
    silero_sensitivity: float = Field(0.4, description="Silero VAD sensitivity")
    silero_use_onnx: bool = Field(False, description="Use ONNX runtime for Silero VAD")
    silero_deactivity_detection: bool = Field(False, description="Enable deactivity detection")
    webrtc_sensitivity: int = Field(3, description="WebRTC VAD sensitivity")
    post_speech_silence_duration: float = Field(0.6, description="Post-speech silence duration in seconds")
    min_length_of_recording: float = Field(0.5, description="Minimum length of recording in seconds")
    min_gap_between_recordings: float = Field(0.0, description="Minimum gap between recordings in seconds")
    pre_recording_buffer_duration: float = Field(1.0, description="Pre-recording buffer duration in seconds")

class RealtimeTranscriptionConfig(BaseModel):
    """Real-time transcription configuration"""
    enable_realtime_transcription: bool = Field(True, description="Enable realtime transcription")
    use_main_model_for_realtime: bool = Field(False, description="Use main model for realtime transcription")
    realtime_model_type: str = Field("tiny", description="Model type for realtime transcription")
    realtime_processing_pause: float = Field(0.2, description="Pause between realtime processing in seconds")
    init_realtime_after_seconds: float = Field(0.2, description="Start realtime transcription after this many seconds")
    realtime_batch_size: int = Field(16, description="Batch size for realtime model")
    beam_size_realtime: int = Field(3, description="Beam size for realtime decoding")
    initial_prompt_realtime: Optional[str] = Field(None, description="Initial prompt for realtime transcription")

class RealtimeWakeWordConfig(BaseModel):
    """Wake word configuration for real-time processing"""
    wakeword_backend: str = Field("pvporcupine", description="Wake word detection backend")
    wake_words: str = Field("", description="Wake words to listen for (comma-separated)")
    wake_words_sensitivity: float = Field(0.6, description="Wake word detection sensitivity")
    wake_word_activation_delay: float = Field(0.0, description="Delay after wake word detection")
    wake_word_timeout: float = Field(5.0, description="Timeout after wake word detection")
    wake_word_buffer_duration: float = Field(0.1, description="Wake word buffer duration")

class RealtimeProcessingConfig(BaseModel):
    """Configuration specific to real-time processing"""
    vad: RealtimeVadConfig = Field(default_factory=RealtimeVadConfig, description="Voice activity detection settings")
    transcription: RealtimeTranscriptionConfig = Field(default_factory=RealtimeTranscriptionConfig, description="Real-time transcription settings")
    wake_word: Optional[RealtimeWakeWordConfig] = Field(None, description="Wake word detection settings (optional)")
    ensure_sentence_starting_uppercase: bool = Field(True, description="Ensure sentences start with uppercase")
    ensure_sentence_ends_with_period: bool = Field(True, description="Ensure sentences end with a period")

# Unified request models
class AudioSource(BaseModel):
    """Audio source information"""
    type: Literal["upload", "azure_blob", "stream"] = Field(..., description="Type of audio source")
    # For upload
    file_name: Optional[str] = Field(None, description="Original filename for uploads")
    file_size_mb: Optional[float] = Field(None, description="File size in MB for uploads")
    # For blob
    path: Optional[str] = Field(None, description="Path in blob storage (container/filename)")
    # For stream
    sample_rate: Optional[int] = Field(16000, description="Audio sample rate for streaming")

class UnifiedTranscriptionRequest(BaseModel):
    """Unified request for both batch and real-time transcription"""
    # Common fields
    model: BaseModelConfig = Field(default_factory=BaseModelConfig, description="Base model configuration")
    
    # Mode-specific configuration - only one will be used based on mode
    mode: Literal["batch", "realtime"] = Field("batch", description="Processing mode: batch or realtime")
    batch_config: Optional[BatchProcessingConfig] = Field(None, description="Batch processing configuration")
    batch_options: Optional[BatchDiarizationOptions] = Field(None, description="Batch diarization options")
    realtime_config: Optional[RealtimeProcessingConfig] = Field(None, description="Real-time processing configuration")
    
    # Audio source - required for batch, optional for real-time (which uses WebSocket)
    audio: Optional[AudioSource] = Field(None, description="Audio source information")
    
    # Custom metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Custom metadata")

# New WebSocket commands that match the unified schema
class UnifiedWebSocketConnect(WebSocketCommand):
    """Command to establish a WebSocket connection with unified config"""
    command: Literal["connect"] = "connect"
    config: UnifiedTranscriptionRequest = Field(..., description="Unified transcription configuration")