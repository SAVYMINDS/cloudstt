from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.routing import APIRouter
import asyncio
import uuid
import base64
import logging
import json
import os
import queue
import traceback
import numpy as np
import multiprocessing

# Import shared schemas
from .schemas import (
    WebSocketConfig,
    WebSocketConnect,
    WebSocketAudioData,
    WebSocketSimpleCommand,
    WebSocketEvent
)

# Import unified service from main
from .main import transcription_service

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Create a queue for async messages
message_queue = queue.Queue()

# Create a function to process messages from the queue
async def process_message_queue(websocket):
    """Process messages from the queue and send them to the websocket."""
    message_processor_task = None
    
    async def message_processor():
        while True:
            try:
                # Check if there are messages in the queue
                while not message_queue.empty():
                    # Get message from queue
                    message = message_queue.get_nowait()
                    # Send to websocket
                    await websocket.send_json(message)
                    message_queue.task_done()
                
                # Wait a bit before checking again
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error processing message queue: {str(e)}")
                await asyncio.sleep(1)  # Wait a bit longer if there's an error
    
    # Start the message processor
    message_processor_task = asyncio.create_task(message_processor())
    
    # Return the task so it can be cancelled later
    return message_processor_task

# Create a new callback wrapper that uses the queue
def callback_wrapper(event_type, session_id, additional_data=None):
    """Create a callback that puts messages on the queue instead of trying to use asyncio directly."""
    def wrapper(*args, **kwargs):
        # Create the message
        message = {
            "event": event_type,
            "session_id": session_id
        }
        
        # Add additional data if provided (e.g. text from transcription)
        if additional_data and args and len(args) > 0:
            data = {additional_data: args[0]}
            message["data"] = data
        
        # Put the message on the queue
        message_queue.put(message)
    
    return wrapper

# Using the save_audio_to_blob method from the unified service

@router.websocket("/transcribe")
async def transcribe_stream(websocket: WebSocket):
    """Real-time transcription via WebSocket."""
    await websocket.accept()
    
    # Track active sessions using session_id
    session_id = str(uuid.uuid4())
    session = None
    message_processor_task = None
    audio_url = None
    
    try:
        # Wait for connection command with configuration
        raw_message = await websocket.receive_text()
        try:
            init_message = json.loads(raw_message)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {str(e)}")
            await websocket.send_json({
                "event": "error",
                "data": {"message": f"Invalid JSON: {str(e)}"}
            })
            return
        
        if init_message.get("command") != "connect":
            await websocket.send_json({
                "event": "error",
                "data": {"message": "First message must be a connection request with 'command': 'connect'"}
            })
            return
        
        logger.info(f"New WebSocket connection established: {session_id}")
        
        # Start the message queue processor
        message_processor_task = await process_message_queue(websocket)
        
        # Extract configuration from the message using the unified schema
        # Check if config uses the new unified format or the old format
        config = init_message.get("config", {})
        
        # If using the new unified format, extract the relevant parts
        if "mode" in config and config.get("mode") == "realtime":
            model_config = config.get("model", {})
            realtime_config = config.get("realtime_config", {})
            
            # Combine appropriate sections for the recorder
            recorder_config = {
                # Base model settings
                "model": model_config.get("name", "tiny"),
                "language": model_config.get("language", "auto"),
                "compute_type": model_config.get("compute_type", "float16"),
                "device": model_config.get("device", "cuda"),
                "use_microphone": True,
                "debug_mode": True,
                # "input_device_index": model_config.get("input_device_index", 0) # No longer needed here for this mode
            }
            
            # Add VAD settings if present
            if realtime_config and "vad" in realtime_config:
                vad_config = realtime_config.get("vad", {})
                recorder_config.update({
                    "silero_sensitivity": vad_config.get("silero_sensitivity", 0.4),
                    "silero_use_onnx": vad_config.get("silero_use_onnx", False),
                    "silero_deactivity_detection": vad_config.get("silero_deactivity_detection", False),
                    "webrtc_sensitivity": vad_config.get("webrtc_sensitivity", 3),
                    "post_speech_silence_duration": vad_config.get("post_speech_silence_duration", 0.6),
                    "min_length_of_recording": vad_config.get("min_length_of_recording", 0.5),
                    "min_gap_between_recordings": vad_config.get("min_gap_between_recordings", 0),
                    "pre_recording_buffer_duration": vad_config.get("pre_recording_buffer_duration", 1.0),
                })
            
            # Add transcription settings if present
            if realtime_config and "transcription" in realtime_config:
                trans_config = realtime_config.get("transcription", {})
                recorder_config.update({
                    "enable_realtime_transcription": trans_config.get("enable_realtime_transcription", True),
                    "use_main_model_for_realtime": trans_config.get("use_main_model_for_realtime", False),
                    "realtime_model_type": trans_config.get("realtime_model_type", "tiny"),
                    "realtime_processing_pause": trans_config.get("realtime_processing_pause", 0.2),
                    "init_realtime_after_seconds": trans_config.get("init_realtime_after_seconds", 0.2),
                    "realtime_batch_size": trans_config.get("realtime_batch_size", 16),
                    "beam_size_realtime": trans_config.get("beam_size_realtime", 3),
                    "initial_prompt_realtime": trans_config.get("initial_prompt_realtime"),
                })
                
            # Add wake word settings if present
            if realtime_config and "wake_word" in realtime_config:
                ww_config = realtime_config.get("wake_word", {})
                recorder_config.update({
                    "wakeword_backend": ww_config.get("wakeword_backend", "pvporcupine"),
                    "wake_words": ww_config.get("wake_words", ""),
                    "wake_words_sensitivity": ww_config.get("wake_words_sensitivity", 0.6),
                    "wake_word_activation_delay": ww_config.get("wake_word_activation_delay", 0.0),
                    "wake_word_timeout": ww_config.get("wake_word_timeout", 5.0),
                    "wake_word_buffer_duration": ww_config.get("wake_word_buffer_duration", 0.1),
                })
                
            # Add formatting settings
            if realtime_config:
                recorder_config.update({
                    "ensure_sentence_starting_uppercase": realtime_config.get("ensure_sentence_starting_uppercase", True),
                    "ensure_sentence_ends_with_period": realtime_config.get("ensure_sentence_ends_with_period", True),
                })
        else:
            # Use the config as is (legacy format)
            recorder_config = config
        
        # Create callback functions for this session
        callback_functions = {
            "on_recording_start": callback_wrapper("recording_start", session_id),
            "on_recording_stop": callback_wrapper("recording_stop", session_id),
            "on_transcription_start": callback_wrapper("transcription_start", session_id),
            "on_realtime_transcription_update": callback_wrapper("realtime_update", session_id, "text"),
            "on_realtime_transcription_stabilized": callback_wrapper("realtime_stabilized", session_id, "text"),
            "on_vad_detect_start": callback_wrapper("voice_activity_start", session_id),
            "on_vad_detect_stop": callback_wrapper("voice_activity_stop", session_id),
            "on_wakeword_detected": callback_wrapper("wakeword_detected", session_id)
        }
        
        # Create enhanced storage session
        logger.info(f"🗂️ Creating Azure Files storage session for {session_id}")
        storage_session = transcription_service.realtime_storage.create_session(session_id, recorder_config)
        logger.info(f"✅ Azure Files storage session created successfully for {session_id}")
        
        # Create enhanced callback functions that also save to storage
        enhanced_callbacks = {
            "on_recording_start": callback_wrapper("recording_start", session_id),
            "on_recording_stop": callback_wrapper("recording_stop", session_id),
            "on_transcription_start": callback_wrapper("transcription_start", session_id),
            "on_realtime_transcription_update": lambda text: [
                callback_wrapper("realtime_update", session_id, "text")(text),
                logger.debug(f"⏳ Realtime update for {session_id}: '{text[:50]}...'") if text else None,
                transcription_service.realtime_storage.add_realtime_transcript(session_id, text, is_stabilized=False) if text else None
            ],
            "on_realtime_transcription_stabilized": lambda text: [
                callback_wrapper("realtime_stabilized", session_id, "text")(text),
                logger.info(f"🔒 Realtime stabilized for {session_id}: '{text}'") if text else logger.info(f"🔒 Empty stabilized transcript for {session_id}"),
                logger.info(f"💾 Saving stabilized transcript to Azure Files for {session_id}") if text else None,
                transcription_service.realtime_storage.add_realtime_transcript(session_id, text, is_stabilized=True) if text else None
            ],
            "on_vad_detect_start": callback_wrapper("voice_activity_start", session_id),
            "on_vad_detect_stop": callback_wrapper("voice_activity_stop", session_id),
            "on_wakeword_detected": callback_wrapper("wakeword_detected", session_id)
        }
        
        # Create recorder using the unified service
        session = transcription_service.create_audio_recorder(recorder_config, session_id, enhanced_callbacks)
        
        # Store session
        transcription_service.active_sessions[session_id] = {
            "recorder": session,
            "audio_url": None,
            "storage_session": storage_session
        }
        
        # Send successful connection response
        await websocket.send_json({
            "event": "connected",
            "session_id": session_id,
            "data": {
                "model": recorder_config.get("model", "tiny"),
                "message": "Connection established and STT session initialized"
            }
        })
        
        # Process WebSocket messages
        while True:
            raw_message = await websocket.receive_text()
            try:
                message = json.loads(raw_message)
                command = message.get("command")
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {str(e)}")
                await websocket.send_json({
                    "event": "error",
                    "data": {"message": f"Invalid JSON: {str(e)}"}
                })
                continue
            
            if command == "start_listening":
                # Try session.start() as per RealtimeSTT documentation for manual mic recording
                # Even with use_microphone=False, this might initialize internal states for feed_audio processing.
                if session:
                    session.start()
                await websocket.send_json({
                    "event": "listening_started",
                    "session_id": session_id
                })
                
            elif command == "stop_listening":
                # Stop current listening/recording session
                audio_url = None
                
                # if session.is_recording: # is_recording might not be True in feed_audio mode
                if session: # Call stop() regardless to finalize any processing
                    session.stop()
                    logger.info(f"Called session.stop() for session {session_id}")
                
                # Try to get audio URL if we haven't set it yet
                if audio_url is None and session_id in transcription_service.active_sessions:
                    audio_url = transcription_service.active_sessions[session_id].get("audio_url")
                
                await websocket.send_json({
                    "event": "listening_stopped",
                    "session_id": session_id,
                    "data": {"audio_url": audio_url} if audio_url else {}
                })
                
            elif command == "get_transcript":
                # Get final transcript from current audio (main model processing)
                logger.info(f"🎯 GET_TRANSCRIPT command received for session {session_id}")
                
                try:
                    result = session.text()
                    logger.info(f"📝 Got transcript from session: '{result}' ({len(result)} chars)")
                    
                    # Get additional information
                    detected_language = getattr(session, 'detected_language', None)
                    language_probability = getattr(session, 'detected_language_probability', None)
                    logger.info(f"🌍 Language detection: {detected_language} ({language_probability})")
                    
                    # Check if session exists in realtime storage
                    if session_id in transcription_service.realtime_storage.active_sessions:
                        logger.info(f"✅ Session {session_id} found in realtime storage")
                        storage_session = transcription_service.realtime_storage.active_sessions[session_id]
                        logger.info(f"📊 Current session metrics: {storage_session.metrics.total_chunks_received} chunks, "
                                   f"{storage_session.metrics.realtime_transcripts_count} realtime transcripts")
                    else:
                        logger.error(f"❌ Session {session_id} NOT found in realtime storage!")
                        logger.info(f"🔍 Active sessions: {list(transcription_service.realtime_storage.active_sessions.keys())}")
                    
                    # Finalize the storage session with main model results
                    logger.info(f"🔄 Finalizing Azure Files storage session for {session_id}")
                    finalized_session = transcription_service.realtime_storage.finalize_session(
                        session_id=session_id,
                        final_transcript=result,
                        segments=None,  # Could be enhanced to get segments if available
                        detected_language=detected_language,
                        language_probability=language_probability
                    )
                    
                    if finalized_session:
                        logger.info(f"✅ Azure Files storage session finalized successfully for {session_id}")
                        logger.info(f"📊 Final session metrics: {finalized_session.metrics.total_audio_duration:.2f}s audio, "
                                   f"{finalized_session.metrics.realtime_transcripts_count} realtime transcripts")
                        logger.info(f"🎵 Audio URL: {finalized_session.audio_url}")
                    else:
                        logger.error(f"❌ Failed to finalize Azure Files storage session for {session_id}")
                    
                    # Get audio URL from finalized session
                    audio_url = finalized_session.audio_url if finalized_session else None
                    
                    # Prepare response data
                    response_data = {
                        "text": result,
                        "language": detected_language,
                        "language_probability": language_probability
                    }
                
                    # Add comprehensive session data if available
                    if finalized_session:
                        session_summary = {
                            "total_duration": finalized_session.metrics.total_audio_duration,
                            "realtime_transcripts_count": finalized_session.metrics.realtime_transcripts_count,
                            "total_chunks": finalized_session.metrics.total_chunks_received,
                            "session_id": session_id
                        }
                        response_data.update({
                            "audio_url": audio_url,
                            "session_summary": session_summary
                        })
                        logger.info(f"📤 Sending session summary: {session_summary}")
                    else:
                        logger.warning(f"⚠️ No session summary available - finalization failed")
                    
                    logger.info(f"📤 Sending transcript response with {len(response_data)} fields")
                    await websocket.send_json({
                        "event": "transcript",
                        "session_id": session_id,
                        "data": response_data
                    })
                    
                except Exception as e:
                    logger.error(f"❌ Error in get_transcript for {session_id}: {str(e)}")
                    logger.error(traceback.format_exc())
                    await websocket.send_json({
                        "event": "error",
                        "session_id": session_id,
                        "data": {"message": f"Error getting transcript: {str(e)}"}
                    })
                
            elif command == "send_audio":
                # Process audio data sent by the client
                audio_data = base64.b64decode(message["audio"])
                sample_rate = message.get("sample_rate", 16000)
                
                # Feed audio to the recorder
                session.feed_audio(audio_data, sample_rate)
                
                # Also save audio chunk to storage
                chunk_success = transcription_service.realtime_storage.add_audio_chunk(session_id, audio_data, sample_rate)
                if chunk_success:
                    logger.debug(f"🎤 Audio chunk saved to Azure Files for session {session_id}: {len(audio_data)} bytes")
                else:
                    logger.error(f"❌ Failed to save audio chunk to Azure Files for session {session_id}")
                
            elif command == "wakeup":
                # Manually trigger wake word detection
                session.wakeup()
                await websocket.send_json({
                    "event": "wakeup_triggered",
                    "session_id": session_id
                })
                
            elif command == "abort":
                # Abort current transcription/recording
                session.abort()
                await websocket.send_json({
                    "event": "aborted",
                    "session_id": session_id
                })
                
            elif command == "disconnect":
                # Disconnect the session
                break
                
            else:
                # Handle unknown command
                await websocket.send_json({
                    "event": "error",
                    "session_id": session_id,
                    "data": {"message": f"Unknown command: {command}"}
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            await websocket.send_json({
                "event": "error",
                "data": {"message": f"Server error: {str(e)}"}
            })
        except:
            pass
    finally:
        # Clean up resources
        if session:
            session.shutdown()
        
        # Clean up storage session (preserve transcripts and audio by default)
        transcription_service.realtime_storage.cleanup_session(
            session_id, 
            preserve_audio=True, 
            preserve_transcripts=True
        )
        
        # Remove session from active sessions
        if session_id in transcription_service.active_sessions:
            del transcription_service.active_sessions[session_id]
        
        # Cancel message processor task
        if message_processor_task:
            message_processor_task.cancel()
            
        logger.info(f"Session cleaned up: {session_id}")

print(f"Available CPU cores: {multiprocessing.cpu_count()}")

