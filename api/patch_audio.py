from RealtimeSTT.audio_recorder import AudioToTextRecorder
import logging
import time

# Get the logger
logger = logging.getLogger(__name__)

# Store the original _audio_data_worker method
original_audio_data_worker = AudioToTextRecorder._audio_data_worker

# Create a patched version of _audio_data_worker
def patched_audio_data_worker(audio_queue, target_sample_rate, buffer_size, input_device_index, 
                             shutdown_event, interrupt_stop_event, use_microphone):
    """Patched version of _audio_data_worker that properly handles use_microphone=False"""
    try:
        import pyaudio
        import numpy as np
        from scipy import signal
        
        # Check if microphone is enabled before initializing
        if not use_microphone.value:
            logger.info("Microphone disabled for this session. Running in stream-only mode.")
            
            # Just wait for shutdown instead of trying to access hardware
            while not shutdown_event.is_set():
                time.sleep(0.1)
            
            return
        
        # Original behavior for microphone mode
        logger.info(f"Initializing microphone recording (sample rate: {target_sample_rate} Hz, buffer size: {buffer_size})")
        audio_interface = pyaudio.PyAudio()
        
        # Initialize audio stream
        stream = audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=target_sample_rate,
            input=True,
            frames_per_buffer=buffer_size,
            input_device_index=input_device_index
        )
        
        logger.info(f"Microphone recording initialized at {target_sample_rate} Hz")
        
        # Main recording loop
        while not shutdown_event.is_set():
            try:
                data = stream.read(buffer_size, exception_on_overflow=False)
                audio_queue.put(data)
            except Exception as e:
                logger.error(f"Error reading audio: {e}")
                
    except KeyboardInterrupt:
        interrupt_stop_event.set()
        logger.debug("Audio worker interrupted")
    finally:
        if 'stream' in locals() and stream:
            stream.stop_stream()
            stream.close()
        if 'audio_interface' in locals() and audio_interface:
            audio_interface.terminate()

# Apply the monkey patch
def apply_patch():
    """Apply the monkey patch to RealtimeSTT"""
    AudioToTextRecorder._audio_data_worker = staticmethod(patched_audio_data_worker)
    logger.info("Applied patch to RealtimeSTT _audio_data_worker to fix microphone initialization") 