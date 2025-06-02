#!/usr/bin/env python3
"""
Debug script to test the realtime storage service and identify the session finalization issue.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storage.azure_storage import AzureFilesStorage
from api.realtime_storage_service import RealtimeStorageService

def debug_session_workflow():
    """Debug the session workflow to identify the issue."""
    print("🔍 Debugging Realtime Storage Session Workflow...")
    
    # Initialize storage
    storage_client = AzureFilesStorage()
    realtime_storage = RealtimeStorageService(storage_client)
    
    # Test configuration
    session_id = "debug-session-123"
    test_config = {
        "realtime_model_type": "tiny",
        "model": "base",
        "language": "en",
        "compute_type": "float16",
        "device": "cuda",
        "enable_realtime_transcription": True,
        "use_main_model_for_realtime": False,
        "realtime_processing_pause": 0.2,
        "silero_sensitivity": 0.4,
        "webrtc_sensitivity": 3,
        "post_speech_silence_duration": 0.6,
        "min_length_of_recording": 0.5,
        "ensure_sentence_starting_uppercase": True,
        "ensure_sentence_ends_with_period": True
    }
    
    print(f"📋 Test Configuration:")
    print(f"  Session ID: {session_id}")
    print(f"  Realtime Model: {test_config['realtime_model_type']}")
    print(f"  Main Model: {test_config['model']}")
    print("")
    
    # Step 1: Create session
    print("1️⃣ Creating realtime session...")
    session_result = realtime_storage.create_session(session_id, test_config)
    
    if session_result:
        print(f"   ✅ Session created successfully")
        print(f"   📊 Session ID: {session_result.session_id}")
        print(f"   📊 Status: {session_result.status}")
    else:
        print(f"   ❌ Failed to create session")
        return False
    
    # Check active sessions
    print(f"🔍 Active sessions: {list(realtime_storage.active_sessions.keys())}")
    print(f"🔍 Audio buffers: {list(realtime_storage.audio_buffers.keys())}")
    
    # Step 2: Add some audio chunks
    print("2️⃣ Adding audio chunks...")
    sample_rate = 16000
    chunk_duration = 1.0
    num_chunks = 3
    
    for i in range(num_chunks):
        # Generate fake audio data
        samples = int(sample_rate * chunk_duration)
        frequency = 440 + (i * 100)
        t = np.linspace(0, chunk_duration, samples, False)
        audio_data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        
        success = realtime_storage.add_audio_chunk(session_id, audio_data.tobytes(), sample_rate)
        if success:
            print(f"   ✅ Added audio chunk {i+1}/{num_chunks}")
        else:
            print(f"   ❌ Failed to add audio chunk {i+1}")
    
    # Step 3: Add some realtime transcripts
    print("3️⃣ Adding realtime transcripts...")
    realtime_transcripts = [
        ("Hello", False),
        ("Hello there", True),
        ("How are", False),
        ("How are you?", True),
    ]
    
    for i, (text, is_stabilized) in enumerate(realtime_transcripts):
        success = realtime_storage.add_realtime_transcript(session_id, text, is_stabilized)
        if success:
            status = "🔒 stabilized" if is_stabilized else "⏳ updating"
            print(f"   ✅ Added transcript {i+1}: '{text}' ({status})")
        else:
            print(f"   ❌ Failed to add transcript {i+1}")
    
    # Check session state before finalization
    print("4️⃣ Checking session state before finalization...")
    if session_id in realtime_storage.active_sessions:
        session = realtime_storage.active_sessions[session_id]
        print(f"   ✅ Session found in active_sessions")
        print(f"   📊 Chunks: {session.metrics.total_chunks_received}")
        print(f"   📊 Realtime transcripts: {session.metrics.realtime_transcripts_count}")
        print(f"   📊 Audio duration: {session.metrics.total_audio_duration:.2f}s")
    else:
        print(f"   ❌ Session NOT found in active_sessions")
        return False
    
    if session_id in realtime_storage.audio_buffers:
        audio_chunks = len(realtime_storage.audio_buffers[session_id])
        print(f"   ✅ Audio buffer found with {audio_chunks} chunks")
    else:
        print(f"   ❌ Audio buffer NOT found")
        return False
    
    # Step 4: Finalize session
    print("5️⃣ Finalizing session...")
    final_transcript = "Hello there, how are you?"
    
    finalized_session = realtime_storage.finalize_session(
        session_id=session_id,
        final_transcript=final_transcript,
        segments=None,
        detected_language="en",
        language_probability=0.95
    )
    
    if finalized_session:
        print(f"   ✅ Session finalized successfully")
        print(f"   📊 Final metrics:")
        print(f"      - Status: {finalized_session.status}")
        print(f"      - Total duration: {finalized_session.metrics.total_audio_duration:.2f}s")
        print(f"      - Audio chunks: {finalized_session.metrics.total_chunks_received}")
        print(f"      - Realtime transcripts: {finalized_session.metrics.realtime_transcripts_count}")
        print(f"      - Main transcript: '{finalized_session.main_model_transcript}'")
        print(f"      - Audio URL: {finalized_session.audio_url}")
        
        # Test session summary creation
        session_summary = {
            "total_duration": finalized_session.metrics.total_audio_duration,
            "realtime_transcripts_count": finalized_session.metrics.realtime_transcripts_count,
            "total_chunks": finalized_session.metrics.total_chunks_received,
            "session_id": session_id
        }
        print(f"   📊 Session summary: {session_summary}")
        
        return True
    else:
        print(f"   ❌ Failed to finalize session")
        return False

if __name__ == "__main__":
    print("🚀 Starting Debug Session Test...")
    print("=" * 60)
    
    success = debug_session_workflow()
    
    print("=" * 60)
    if success:
        print("✅ Debug test completed successfully!")
        print("🔍 The realtime storage service appears to be working correctly.")
        print("🤔 The issue might be in the WebSocket callback functions or session management.")
    else:
        print("❌ Debug test failed!")
        print("🔍 There's an issue with the realtime storage service.")
    
    sys.exit(0 if success else 1) 