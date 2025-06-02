"""
Enhanced Realtime Storage Service for CloudSTT
Handles comprehensive storage of realtime sessions including:
- Streaming audio chunks
- Realtime model transcriptions (live updates)
- Main model final transcription
- Session metadata and configuration
- Timing and performance metrics
"""

import os
import json
import logging
import time
import uuid
import numpy as np
import soundfile as sf
from datetime import datetime
from typing import Dict, List, Optional, Any, BinaryIO
from dataclasses import dataclass, asdict
from pathlib import Path

from storage.azure_storage import AzureFilesStorage

logger = logging.getLogger(__name__)

@dataclass
class RealtimeTranscript:
    """Individual realtime transcript entry"""
    timestamp: float
    text: str
    is_stabilized: bool
    confidence: Optional[float] = None
    speaker: Optional[str] = None

@dataclass
class SessionConfiguration:
    """Session configuration details"""
    session_id: str
    created_at: str
    
    # Model configurations
    realtime_model: str
    main_model: str
    language: str
    compute_type: str
    device: str
    
    # Realtime settings
    enable_realtime_transcription: bool
    use_main_model_for_realtime: bool
    realtime_processing_pause: float
    
    # VAD settings
    silero_sensitivity: float
    webrtc_sensitivity: int
    post_speech_silence_duration: float
    min_length_of_recording: float
    
    # Additional settings
    ensure_sentence_starting_uppercase: bool
    ensure_sentence_ends_with_period: bool

@dataclass
class SessionMetrics:
    """Session performance metrics"""
    session_start_time: float
    session_end_time: Optional[float] = None
    total_audio_duration: float = 0.0
    total_chunks_received: int = 0
    realtime_transcripts_count: int = 0
    main_model_processing_time: Optional[float] = None
    average_chunk_processing_time: float = 0.0

@dataclass
class SessionResult:
    """Complete session result"""
    session_id: str
    configuration: SessionConfiguration
    metrics: SessionMetrics
    
    # Audio data
    audio_url: Optional[str] = None
    total_audio_duration: float = 0.0
    
    # Transcription results
    realtime_transcripts: List[RealtimeTranscript] = None
    main_model_transcript: Optional[str] = None
    main_model_segments: Optional[List[Dict]] = None
    
    # Language detection
    detected_language: Optional[str] = None
    language_probability: Optional[float] = None
    
    # Speaker information
    num_speakers_detected: int = 0
    
    # Status
    status: str = "active"  # active, completed, error
    error_message: Optional[str] = None

class RealtimeStorageService:
    """Enhanced storage service for realtime transcription sessions"""
    
    def __init__(self, storage_client: AzureFilesStorage):
        self.storage_client = storage_client
        self.active_sessions: Dict[str, SessionResult] = {}
        self.audio_buffers: Dict[str, List[np.ndarray]] = {}
        
    def create_session(self, session_id: str, config: Dict[str, Any]) -> SessionResult:
        """Create a new realtime session with configuration"""
        try:
            # Extract configuration
            session_config = SessionConfiguration(
                session_id=session_id,
                created_at=datetime.now().isoformat(),
                
                # Model configurations
                realtime_model=config.get("realtime_model_type", "tiny"),
                main_model=config.get("model", "tiny"),
                language=config.get("language", "auto"),
                compute_type=config.get("compute_type", "float16"),
                device=config.get("device", "cuda"),
                
                # Realtime settings
                enable_realtime_transcription=config.get("enable_realtime_transcription", True),
                use_main_model_for_realtime=config.get("use_main_model_for_realtime", False),
                realtime_processing_pause=config.get("realtime_processing_pause", 0.2),
                
                # VAD settings
                silero_sensitivity=config.get("silero_sensitivity", 0.4),
                webrtc_sensitivity=config.get("webrtc_sensitivity", 3),
                post_speech_silence_duration=config.get("post_speech_silence_duration", 0.6),
                min_length_of_recording=config.get("min_length_of_recording", 0.5),
                
                # Additional settings
                ensure_sentence_starting_uppercase=config.get("ensure_sentence_starting_uppercase", True),
                ensure_sentence_ends_with_period=config.get("ensure_sentence_ends_with_period", True),
            )
            
            # Create session metrics
            session_metrics = SessionMetrics(
                session_start_time=time.time()
            )
            
            # Create session result
            session_result = SessionResult(
                session_id=session_id,
                configuration=session_config,
                metrics=session_metrics,
                realtime_transcripts=[],
                status="active"
            )
            
            # Store in active sessions
            self.active_sessions[session_id] = session_result
            self.audio_buffers[session_id] = []
            
            # Save initial session metadata
            self._save_session_metadata(session_id)
            
            logger.info(f"Created realtime session {session_id} with models: realtime={session_config.realtime_model}, main={session_config.main_model}")
            return session_result
            
        except Exception as e:
            logger.error(f"Error creating session {session_id}: {str(e)}")
            raise
    
    def add_audio_chunk(self, session_id: str, audio_data: bytes, sample_rate: int = 16000) -> bool:
        """Add audio chunk to session buffer"""
        try:
            if session_id not in self.active_sessions:
                logger.error(f"Session {session_id} not found")
                return False
            
            # Convert audio data to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Add to buffer
            self.audio_buffers[session_id].append(audio_array)
            
            # Update metrics
            session = self.active_sessions[session_id]
            session.metrics.total_chunks_received += 1
            session.metrics.total_audio_duration += len(audio_array) / sample_rate
            
            logger.debug(f"Added audio chunk to session {session_id}: {len(audio_array)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error adding audio chunk to session {session_id}: {str(e)}")
            return False
    
    def add_realtime_transcript(self, session_id: str, text: str, is_stabilized: bool = False, 
                              confidence: Optional[float] = None, speaker: Optional[str] = None) -> bool:
        """Add realtime transcript entry"""
        try:
            if session_id not in self.active_sessions:
                logger.error(f"Session {session_id} not found")
                return False
            
            # Create transcript entry
            transcript_entry = RealtimeTranscript(
                timestamp=time.time(),
                text=text,
                is_stabilized=is_stabilized,
                confidence=confidence,
                speaker=speaker
            )
            
            # Add to session
            session = self.active_sessions[session_id]
            session.realtime_transcripts.append(transcript_entry)
            session.metrics.realtime_transcripts_count += 1
            
            # Save realtime transcript incrementally
            self._save_realtime_transcript(session_id, transcript_entry)
            
            logger.debug(f"Added realtime transcript to session {session_id}: {text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error adding realtime transcript to session {session_id}: {str(e)}")
            return False
    
    def finalize_session(self, session_id: str, final_transcript: Optional[str] = None, 
                        segments: Optional[List[Dict]] = None, detected_language: Optional[str] = None,
                        language_probability: Optional[float] = None) -> Optional[SessionResult]:
        """Finalize session with main model results and save everything"""
        try:
            logger.info(f"🔄 Starting finalize_session for {session_id}")
            logger.info(f"📝 Final transcript: '{final_transcript}' ({len(final_transcript or '')} chars)")
            logger.info(f"🌍 Language: {detected_language} ({language_probability})")
            
            if session_id not in self.active_sessions:
                logger.error(f"❌ Session {session_id} not found in active_sessions")
                logger.info(f"🔍 Available sessions: {list(self.active_sessions.keys())}")
                return None
            
            session = self.active_sessions[session_id]
            logger.info(f"✅ Found session {session_id} in active_sessions")
            
            # Update session end time
            session.metrics.session_end_time = time.time()
            
            # Save accumulated audio
            logger.info(f"💾 Saving accumulated audio for session {session_id}")
            audio_url = self._save_session_audio(session_id)
            session.audio_url = audio_url
            logger.info(f"🎵 Audio saved with URL: {audio_url}")
            
            # Add main model results
            if final_transcript:
                session.main_model_transcript = final_transcript
            if segments:
                session.main_model_segments = segments
            if detected_language:
                session.detected_language = detected_language
            if language_probability:
                session.language_probability = language_probability
            
            session.status = "completed"
            
            # Calculate final metrics
            session_duration = session.metrics.session_end_time - session.metrics.session_start_time
            if session.metrics.total_chunks_received > 0:
                session.metrics.average_chunk_processing_time = session_duration / session.metrics.total_chunks_received
            
            # Save complete session result
            logger.info(f"💾 Saving complete session result for {session_id}")
            save_success = self._save_complete_session_result(session_id)
            logger.info(f"📄 Complete session result saved: {save_success}")
            
            logger.info(f"🎉 Finalized session {session_id}: {session.metrics.total_audio_duration:.2f}s audio, "
                       f"{session.metrics.realtime_transcripts_count} realtime transcripts, "
                       f"{len(session.main_model_transcript or '')} chars final transcript")
            
            return session
            
        except Exception as e:
            logger.error(f"Error finalizing session {session_id}: {str(e)}")
            if session_id in self.active_sessions:
                self.active_sessions[session_id].status = "error"
                self.active_sessions[session_id].error_message = str(e)
            return None
    
    def get_session(self, session_id: str) -> Optional[SessionResult]:
        """Get session result"""
        return self.active_sessions.get(session_id)
    
    def cleanup_session(self, session_id: str, preserve_audio: bool = True, preserve_transcripts: bool = True) -> bool:
        """Clean up session with preservation options"""
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.audio_buffers:
                del self.audio_buffers[session_id]
            
            # Use storage client's cleanup method
            self.storage_client.cleanup_realtime_session(session_id, preserve_output=preserve_transcripts)
            
            logger.info(f"Cleaned up session {session_id} (preserved: audio={preserve_audio}, transcripts={preserve_transcripts})")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {str(e)}")
            return False
    
    def _save_session_metadata(self, session_id: str) -> bool:
        """Save session metadata to Azure Files"""
        try:
            session = self.active_sessions[session_id]
            metadata = {
                "session_id": session_id,
                "configuration": asdict(session.configuration),
                "metrics": asdict(session.metrics),
                "status": session.status,
                "created_at": session.configuration.created_at
            }
            
            return self.storage_client.save_session_metadata(session_id, metadata)
            
        except Exception as e:
            logger.error(f"Error saving session metadata for {session_id}: {str(e)}")
            return False
    
    def _save_realtime_transcript(self, session_id: str, transcript: RealtimeTranscript) -> bool:
        """Save individual realtime transcript entry"""
        try:
            # Create filename with timestamp for ordering
            timestamp_str = str(int(transcript.timestamp * 1000))  # milliseconds
            filename = f"realtime_transcript_{timestamp_str}.json"
            
            transcript_data = asdict(transcript)
            
            return self.storage_client.save_json(
                container_name=self.storage_client.REALTIME_OUTPUT,
                blob_name=f"session-{session_id}/realtime_transcripts/{filename}",
                data=transcript_data
            )
            
        except Exception as e:
            logger.error(f"Error saving realtime transcript for {session_id}: {str(e)}")
            return False
    
    def _save_session_audio(self, session_id: str) -> Optional[str]:
        """Save accumulated session audio to Azure Files"""
        try:
            logger.info(f"🎵 Starting _save_session_audio for {session_id}")
            
            if session_id not in self.audio_buffers:
                logger.error(f"❌ Session {session_id} not found in audio_buffers")
                logger.info(f"🔍 Available audio sessions: {list(self.audio_buffers.keys())}")
                return None
                
            if not self.audio_buffers[session_id]:
                logger.warning(f"⚠️ No audio data for session {session_id} (empty buffer)")
                return None
                
            logger.info(f"✅ Found {len(self.audio_buffers[session_id])} audio chunks for session {session_id}")
            
            # Concatenate all audio chunks
            full_audio = np.concatenate(self.audio_buffers[session_id])
            
            # Create temporary file
            temp_path = f"temp_audio/session_{session_id}_full.wav"
            os.makedirs("temp_audio", exist_ok=True)
            
            # Save to temporary WAV file
            sf.write(temp_path, full_audio, 16000)
            
            # Upload to Azure Files
            with open(temp_path, 'rb') as f:
                success = self.storage_client.upload_file(
                    container_name=self.storage_client.REALTIME_SESSIONS,
                    blob_name=f"session-{session_id}/full_audio.wav",
                    data=f,
                    content_type="audio/wav",
                    metadata={
                        "session_id": session_id,
                        "sample_rate": "16000",
                        "channels": "1",
                        "duration_seconds": str(len(full_audio) / 16000),
                        "total_chunks": str(len(self.audio_buffers[session_id]))
                    }
                )
            
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
            
            if success:
                # Generate URL
                audio_url = self.storage_client.get_sas_url(
                    container_name=self.storage_client.REALTIME_SESSIONS,
                    blob_name=f"session-{session_id}/full_audio.wav",
                    duration_hours=24
                )
                logger.info(f"Saved session audio for {session_id}: {len(full_audio)/16000:.2f}s")
                return audio_url
            else:
                logger.error(f"Failed to save session audio for {session_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving session audio for {session_id}: {str(e)}")
            return None
    
    def _save_complete_session_result(self, session_id: str) -> bool:
        """Save complete session result with all data"""
        try:
            session = self.active_sessions[session_id]
            
            # Create comprehensive result
            complete_result = {
                "session_id": session_id,
                "status": session.status,
                "error_message": session.error_message,
                
                # Configuration
                "configuration": asdict(session.configuration),
                
                # Metrics
                "metrics": asdict(session.metrics),
                
                # Audio information
                "audio": {
                    "url": session.audio_url,
                    "total_duration_seconds": session.metrics.total_audio_duration,
                    "total_chunks": session.metrics.total_chunks_received
                },
                
                # Realtime transcription results
                "realtime_transcription": {
                    "total_entries": len(session.realtime_transcripts),
                    "transcripts": [asdict(t) for t in session.realtime_transcripts],
                    "combined_text": " ".join([t.text for t in session.realtime_transcripts if t.is_stabilized])
                },
                
                # Main model results
                "main_model_transcription": {
                    "transcript": session.main_model_transcript,
                    "segments": session.main_model_segments,
                    "detected_language": session.detected_language,
                    "language_probability": session.language_probability
                },
                
                # Timestamps
                "created_at": session.configuration.created_at,
                "completed_at": datetime.now().isoformat() if session.status == "completed" else None
            }
            
            # Save complete result
            success = self.storage_client.save_json(
                container_name=self.storage_client.REALTIME_OUTPUT,
                blob_name=f"session-{session_id}/complete_session_result.json",
                data=complete_result
            )
            
            if success:
                logger.info(f"Saved complete session result for {session_id}")
            else:
                logger.error(f"Failed to save complete session result for {session_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error saving complete session result for {session_id}: {str(e)}")
            return False 