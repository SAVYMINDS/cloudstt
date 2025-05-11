#!/usr/bin/env python3
"""
Real-Time Speech-to-Text with RealtimeSTT

This script implements a complete audio transcription system that can be run directly.
It provides voice activity detection, real-time transcription, and wake word detection.

Author: Based on RealtimeSTT by Kolja Beigel
"""

import os
import sys
import time
import torch
import logging
import numpy as np
import argparse
import threading
import copy
import gc
import signal
import queue
from ctypes import c_bool
import torch.multiprocessing as mp

# Constants
INIT_MODEL_TRANSCRIPTION = "tiny"
INIT_MODEL_TRANSCRIPTION_REALTIME = "tiny"
INIT_REALTIME_PROCESSING_PAUSE = 0.2
INIT_REALTIME_INITIAL_PAUSE = 0.2
INIT_SILERO_SENSITIVITY = 0.4
INIT_WEBRTC_SENSITIVITY = 3
INIT_POST_SPEECH_SILENCE_DURATION = 0.6
INIT_MIN_LENGTH_OF_RECORDING = 0.5
INIT_MIN_GAP_BETWEEN_RECORDINGS = 0
INIT_WAKE_WORDS_SENSITIVITY = 0.6
INIT_PRE_RECORDING_BUFFER_DURATION = 1.0
INIT_WAKE_WORD_ACTIVATION_DELAY = 0.0
INIT_WAKE_WORD_TIMEOUT = 5.0
INIT_WAKE_WORD_BUFFER_DURATION = 0.1
ALLOWED_LATENCY_LIMIT = 100

TIME_SLEEP = 0.02
SAMPLE_RATE = 16000
BUFFER_SIZE = 512
INT16_MAX_ABS_VALUE = 32768.0

# Default batch sizes
DEFAULT_BATCH_SIZE = 16
DEFAULT_REALTIME_BATCH_SIZE = 16

# Default beam sizes
DEFAULT_BEAM_SIZE = 5
DEFAULT_BEAM_SIZE_REALTIME = 3

# Setup logger
logger = logging.getLogger("realtimestt")
logger.propagate = False

# Check if necessary imports are available and install them if not
try:
    from faster_whisper import WhisperModel, BatchedInferencePipeline
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faster-whisper"])
    from faster_whisper import WhisperModel, BatchedInferencePipeline

try:
    import webrtcvad
except ImportError:
    print("Installing webrtcvad...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "webrtcvad"])
    import webrtcvad

try:
    import soundfile as sf
except ImportError:
    print("Installing soundfile...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile"])
    import soundfile as sf

try:
    import halo
except ImportError:
    print("Installing halo...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "halo"])
    import halo

try:
    import scipy
    from scipy import signal
except ImportError:
    print("Installing scipy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    from scipy import signal

# Import optional packages
try:
    import pvporcupine
except ImportError:
    pvporcupine = None

try:
    import openwakeword
    from openwakeword.model import Model as OpenWakeWordModel
except ImportError:
    openwakeword = None
    OpenWakeWordModel = None

class bcolors:
    OKGREEN = '\033[92m'  # Green for active speech detection
    WARNING = '\033[93m'  # Yellow for silence detection
    ENDC = '\033[0m'      # Reset to default color


class TranscriptionWorker:
    def __init__(self, conn, stdout_pipe, model_path, download_root, compute_type, gpu_device_index, device,
                 ready_event, shutdown_event, interrupt_stop_event, beam_size, initial_prompt, suppress_tokens,
                 batch_size, faster_whisper_vad_filter):
        self.conn = conn
        self.stdout_pipe = stdout_pipe
        self.model_path = model_path
        self.download_root = download_root
        self.compute_type = compute_type
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.ready_event = ready_event
        self.shutdown_event = shutdown_event
        self.interrupt_stop_event = interrupt_stop_event
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.suppress_tokens = suppress_tokens
        self.batch_size = batch_size
        self.faster_whisper_vad_filter = faster_whisper_vad_filter
        self.queue = queue.Queue()

    def custom_print(self, *args, **kwargs):
        message = ' '.join(map(str, args))
        try:
            self.stdout_pipe.send(message)
        except (BrokenPipeError, EOFError, OSError):
            pass

    def poll_connection(self):
        while not self.shutdown_event.is_set():
            if self.conn.poll(0.01):
                try:
                    data = self.conn.recv()
                    self.queue.put(data)
                except Exception as e:
                    logging.error(f"Error receiving data from connection: {e}", exc_info=True)
            else:
                time.sleep(TIME_SLEEP)

    def run(self):
        if __name__ == "__main__":
             signal.signal(signal.SIGINT, signal.SIG_IGN)
             __builtins__['print'] = self.custom_print

        logging.info(f"Initializing faster_whisper main transcription model {self.model_path}")

        try:
            model = WhisperModel(
                model_size_or_path=self.model_path,
                device=self.device,
                compute_type=self.compute_type,
                device_index=self.gpu_device_index,
                download_root=self.download_root,
            )
            
            # Apply batched inference if batch_size is set
            if self.batch_size > 0:
                model = BatchedInferencePipeline(model=model)

            # Run a warm-up transcription
            current_dir = os.path.dirname(os.path.realpath(__file__))
            warmup_audio_path = os.path.join(current_dir, "warmup_audio.wav")
            
            # Create warmup audio if it doesn't exist
            if not os.path.exists(warmup_audio_path):
                # Create a short silence audio file
                silence = np.zeros(16000, dtype=np.float32)  # 1 second of silence at 16kHz
                sf.write(warmup_audio_path, silence, 16000)
            
            warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
            segments, info = model.transcribe(
                warmup_audio_data, 
                language="en", 
                beam_size=1,
                vad_filter=self.faster_whisper_vad_filter
            )
            model_warmup_transcription = " ".join(segment.text for segment in segments)
        except Exception as e:
            logging.exception(f"Error initializing main faster_whisper transcription model: {e}")
            raise

        self.ready_event.set()
        logging.debug("Faster_whisper main speech to text transcription model initialized successfully")

        # Start the polling thread
        polling_thread = threading.Thread(target=self.poll_connection)
        polling_thread.start()

        try:
            while not self.shutdown_event.is_set():
                try:
                    audio, language = self.queue.get(timeout=0.1)
                    try:
                        logging.debug(f"Transcribing audio with language {language}")
                        if self.batch_size > 0:
                            segments, info = model.transcribe(
                                audio,
                                language=language if language else None,
                                beam_size=self.beam_size,
                                initial_prompt=self.initial_prompt,
                                suppress_tokens=self.suppress_tokens,
                                batch_size=self.batch_size, 
                                vad_filter=self.faster_whisper_vad_filter  # Use VAD filter parameter
                            )
                        else:
                            segments, info = model.transcribe(
                                audio,
                                language=language if language else None,
                                beam_size=self.beam_size,
                                initial_prompt=self.initial_prompt,
                                suppress_tokens=self.suppress_tokens,
                                vad_filter=self.faster_whisper_vad_filter  # Use VAD filter parameter
                            )

                        transcription = " ".join(seg.text for seg in segments).strip()
                        logging.debug(f"Final text detected with main model: {transcription}")
                        self.conn.send(('success', (transcription, info)))
                    except Exception as e:
                        logging.error(f"General error in transcription: {e}", exc_info=True)
                        self.conn.send(('error', str(e)))
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    self.interrupt_stop_event.set()
                    logging.debug("Transcription worker process finished due to KeyboardInterrupt")
                    break
                except Exception as e:
                    logging.error(f"General error in processing queue item: {e}", exc_info=True)
        finally:
            # Restore the original print function
            __builtins__['print'] = print
            self.conn.close()
            self.stdout_pipe.close()
            self.shutdown_event.set()  # Ensure the polling thread will stop
            polling_thread.join()  # Wait for the polling thread to finish


class AudioToTextRecorder:
    """
    A class responsible for capturing audio from the microphone, detecting
    voice activity, and then transcribing the captured audio using the
    `faster_whisper` model.
    """

    def __init__(self,
                 model: str = INIT_MODEL_TRANSCRIPTION,
                 download_root: str = None, 
                 language: str = "",
                 compute_type: str = "default",
                 input_device_index: int = None,
                 gpu_device_index = 0,
                 device: str = "cuda",
                 on_recording_start=None,
                 on_recording_stop=None,
                 on_transcription_start=None,
                 ensure_sentence_starting_uppercase=True,
                 ensure_sentence_ends_with_period=True,
                 use_microphone=True,
                 spinner=True,
                 level=logging.WARNING,
                 batch_size: int = 16,

                 # Realtime transcription parameters
                 enable_realtime_transcription=False,
                 use_main_model_for_realtime=False,
                 realtime_model_type=INIT_MODEL_TRANSCRIPTION_REALTIME,
                 realtime_processing_pause=INIT_REALTIME_PROCESSING_PAUSE,
                 init_realtime_after_seconds=INIT_REALTIME_INITIAL_PAUSE,
                 on_realtime_transcription_update=None,
                 on_realtime_transcription_stabilized=None,
                 realtime_batch_size: int = 16,

                 # Voice activation parameters
                 silero_sensitivity: float = INIT_SILERO_SENSITIVITY,
                 silero_use_onnx: bool = False,
                 silero_deactivity_detection: bool = False,
                 webrtc_sensitivity: int = INIT_WEBRTC_SENSITIVITY,
                 post_speech_silence_duration: float = (
                     INIT_POST_SPEECH_SILENCE_DURATION
                 ),
                 min_length_of_recording: float = (
                     INIT_MIN_LENGTH_OF_RECORDING
                 ),
                 min_gap_between_recordings: float = (
                     INIT_MIN_GAP_BETWEEN_RECORDINGS
                 ),
                 pre_recording_buffer_duration: float = (
                     INIT_PRE_RECORDING_BUFFER_DURATION
                 ),
                 on_vad_start=None,
                 on_vad_stop=None,
                 on_vad_detect_start=None,
                 on_vad_detect_stop=None,

                 # Wake word parameters
                 wakeword_backend: str = "pvporcupine",
                 openwakeword_model_paths: str = None,
                 openwakeword_inference_framework: str = "onnx",
                 wake_words: str = "",
                 wake_words_sensitivity: float = INIT_WAKE_WORDS_SENSITIVITY,
                 wake_word_activation_delay: float = (
                    INIT_WAKE_WORD_ACTIVATION_DELAY
                 ),
                 wake_word_timeout: float = INIT_WAKE_WORD_TIMEOUT,
                 wake_word_buffer_duration: float = INIT_WAKE_WORD_BUFFER_DURATION,
                 on_wakeword_detected=None,
                 on_wakeword_timeout=None,
                 on_wakeword_detection_start=None,
                 on_wakeword_detection_end=None,
                 on_recorded_chunk=None,
                 debug_mode=False,
                 handle_buffer_overflow: bool = True,
                 beam_size: int = 5,
                 beam_size_realtime: int = 3,
                 buffer_size: int = BUFFER_SIZE,
                 sample_rate: int = SAMPLE_RATE,
                 initial_prompt = None,
                 initial_prompt_realtime = None,
                 suppress_tokens = [-1],
                 print_transcription_time: bool = False,
                 early_transcription_on_silence: int = 0,
                 allowed_latency_limit: int = ALLOWED_LATENCY_LIMIT,
                 no_log_file: bool = False,
                 use_extended_logging: bool = False,
                 faster_whisper_vad_filter: bool = True,  # Add the new parameter
                 ):
        """
        Initializes an audio recorder and transcription and wake word detection.
        """
        self.language = language
        self.compute_type = compute_type
        self.input_device_index = input_device_index
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.wake_words = wake_words
        self.wake_word_activation_delay = wake_word_activation_delay
        self.wake_word_timeout = wake_word_timeout
        self.wake_word_buffer_duration = wake_word_buffer_duration
        self.ensure_sentence_starting_uppercase = ensure_sentence_starting_uppercase
        self.ensure_sentence_ends_with_period = ensure_sentence_ends_with_period
        self.use_microphone = mp.Value(c_bool, use_microphone)
        self.min_gap_between_recordings = min_gap_between_recordings
        self.min_length_of_recording = min_length_of_recording
        self.pre_recording_buffer_duration = pre_recording_buffer_duration
        self.post_speech_silence_duration = post_speech_silence_duration
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
        self.on_wakeword_detected = on_wakeword_detected
        self.on_wakeword_timeout = on_wakeword_timeout
        self.on_vad_start = on_vad_start
        self.on_vad_stop = on_vad_stop
        self.on_vad_detect_start = on_vad_detect_start
        self.on_vad_detect_stop = on_vad_detect_stop
        self.on_wakeword_detection_start = on_wakeword_detection_start
        self.on_wakeword_detection_end = on_wakeword_detection_end
        self.on_recorded_chunk = on_recorded_chunk
        self.on_transcription_start = on_transcription_start
        self.enable_realtime_transcription = enable_realtime_transcription
        self.use_main_model_for_realtime = use_main_model_for_realtime
        self.main_model_type = model
        if not download_root:
            download_root = None
        self.download_root = download_root
        self.realtime_model_type = realtime_model_type
        self.realtime_processing_pause = realtime_processing_pause
        self.init_realtime_after_seconds = init_realtime_after_seconds
        self.on_realtime_transcription_update = on_realtime_transcription_update
        self.on_realtime_transcription_stabilized = on_realtime_transcription_stabilized
        self.debug_mode = debug_mode
        self.handle_buffer_overflow = handle_buffer_overflow
        self.beam_size = beam_size
        self.beam_size_realtime = beam_size_realtime
        self.allowed_latency_limit = allowed_latency_limit
        self.batch_size = batch_size
        self.realtime_batch_size = realtime_batch_size
        self.faster_whisper_vad_filter = faster_whisper_vad_filter  # Store the new parameter

        self.level = level
        self.audio_queue = mp.Queue()
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.recording_start_time = 0
        self.recording_stop_time = 0
        self.last_recording_start_time = 0
        self.last_recording_stop_time = 0
        self.wake_word_detect_time = 0
        self.silero_check_time = 0
        self.silero_working = False
        self.speech_end_silence_start = 0
        self.silero_sensitivity = silero_sensitivity
        self.silero_deactivity_detection = silero_deactivity_detection
        self.listen_start = 0
        self.spinner = spinner
        self.halo = None
        self.state = "inactive"
        self.wakeword_detected = False
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.is_webrtc_speech_active = False
        self.is_silero_speech_active = False
        self.recording_thread = None
        self.realtime_thread = None
        self.audio_interface = None
        self.audio = None
        self.stream = None
        self.start_recording_event = threading.Event()
        self.stop_recording_event = threading.Event()
        self.backdate_stop_seconds = 0.0
        self.backdate_resume_seconds = 0.0
        self.last_transcription_bytes = None
        self.last_transcription_bytes_b64 = None
        self.initial_prompt = initial_prompt
        self.initial_prompt_realtime = initial_prompt_realtime
        self.suppress_tokens = suppress_tokens
        self.use_wake_words = wake_words or wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}
        self.detected_language = None
        self.detected_language_probability = 0
        self.detected_realtime_language = None
        self.detected_realtime_language_probability = 0
        self.transcription_lock = threading.Lock()
        self.shutdown_lock = threading.Lock()
        self.transcribe_count = 0
        self.print_transcription_time = print_transcription_time
        self.early_transcription_on_silence = early_transcription_on_silence
        self.use_extended_logging = use_extended_logging

        # Named logger configuration
        logger.setLevel(logging.DEBUG)  # We capture all, then filter via handlers

        log_format = "RealTimeSTT: %(name)s - %(levelname)s - %(message)s"
        file_log_format = "%(asctime)s.%(msecs)03d - " + log_format

        # Create and set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.level)
        console_handler.setFormatter(logging.Formatter(log_format))

        logger.addHandler(console_handler)

        if not no_log_file:
            file_handler = logging.FileHandler('realtimestt.log')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(file_log_format, datefmt='%Y-%m-%d %H:%M:%S'))
            logger.addHandler(file_handler)

        self.is_shut_down = False
        self.shutdown_event = mp.Event()
        
        try:
            # Only set the start method if it hasn't been set already
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method("spawn")
        except RuntimeError as e:
            logger.info(f"Start method has already been set. Details: {e}")

        logger.info("Starting RealTimeSTT")

        self.interrupt_stop_event = mp.Event()
        self.was_interrupted = mp.Event()
        self.main_transcription_ready_event = mp.Event()
        self.parent_transcription_pipe, child_transcription_pipe = mp.Pipe()
        self.parent_stdout_pipe, child_stdout_pipe = mp.Pipe()

        # Set device for model
        self.device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"

        # Initialize the transcription worker
        self.transcript_process = self._start_thread(
            target=self._transcription_worker,
            args=(
                child_transcription_pipe,
                child_stdout_pipe,
                self.main_model_type,
                self.download_root,
                self.compute_type,
                self.gpu_device_index,
                self.device,
                self.main_transcription_ready_event,
                self.shutdown_event,
                self.interrupt_stop_event,
                self.beam_size,
                self.initial_prompt,
                self.suppress_tokens,
                self.batch_size,
                self.faster_whisper_vad_filter,  # Pass the parameter to worker
            )
        )

        # Start audio data reading process if using microphone
        if self.use_microphone.value:
            logger.info(f"Initializing audio recording (creating pyAudio input stream, "
                         f"sample rate: {self.sample_rate} buffer size: {self.buffer_size}")
            self.reader_process = self._start_thread(
                target=self._audio_data_worker,
                args=(
                    self.audio_queue,
                    self.sample_rate,
                    self.buffer_size,
                    self.input_device_index,
                    self.shutdown_event,
                    self.interrupt_stop_event,
                    self.use_microphone
                )
            )

        # Initialize realtime transcription model if enabled
        if self.enable_realtime_transcription and not self.use_main_model_for_realtime:
            try:
                logger.info(f"Initializing faster_whisper realtime transcription model {self.realtime_model_type}")
                self.realtime_model = WhisperModel(
                    model_size_or_path=self.realtime_model_type,
                    device=self.device,
                    compute_type=self.compute_type,
                    device_index=self.gpu_device_index,
                    download_root=self.download_root,
                )
                
                if self.realtime_batch_size > 0:
                    self.realtime_model = BatchedInferencePipeline(model=self.realtime_model)

                # Run a warm-up transcription
                current_dir = os.path.dirname(os.path.realpath(__file__))
                warmup_audio_path = os.path.join(current_dir, "warmup_audio.wav")
                
                # Create the warmup file if it doesn't exist
                if not os.path.exists(warmup_audio_path):
                    silence = np.zeros(16000, dtype=np.float32)  # 1 second of silence at 16kHz
                    sf.write(warmup_audio_path, silence, 16000)
                    
                warmup_audio_data, _ = sf.read(warmup_audio_path, dtype="float32")
                segments, info = self.realtime_model.transcribe(
                    warmup_audio_data, 
                    language="en", 
                    beam_size=1,
                    vad_filter=self.faster_whisper_vad_filter  # Use VAD filter parameter
                )
                
            except Exception as e:
                logger.exception(f"Error initializing faster_whisper realtime transcription model: {e}")
                raise

            logger.debug("Faster_whisper realtime speech to text transcription model initialized successfully")

        # Setup WebRTC VAD
        try:
            logger.info(f"Initializing WebRTC voice with Sensitivity {webrtc_sensitivity}")
            self.webrtc_vad_model = webrtcvad.Vad()
            self.webrtc_vad_model.set_mode(webrtc_sensitivity)
        except Exception as e:
            logger.exception(f"Error initializing WebRTC voice activity detection engine: {e}")
            raise

        logger.debug("WebRTC VAD voice activity detection engine initialized successfully")

        # Setup Silero VAD
        try:
            self.silero_vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                verbose=False,
                onnx=silero_use_onnx
            )
        except Exception as e:
            logger.exception(f"Error initializing Silero VAD voice activity detection engine: {e}")
            raise

        logger.debug("Silero VAD voice activity detection engine initialized successfully")

        self.audio_buffer = []
        self.last_words_buffer = []
        self.frames = []
        self.last_frames = []

        # Recording control flags
        self.is_recording = False
        self.is_running = True
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False

        # Start the recording worker thread
        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        # Start the realtime transcription worker thread
        self.realtime_thread = threading.Thread(target=self._realtime_worker)
        self.realtime_thread.daemon = True
        self.realtime_thread.start()
                   
        # Wait for transcription models to start
        logger.debug('Waiting for main transcription model to start')
        self.main_transcription_ready_event.wait()
        logger.debug('Main transcription model ready')

        self.stdout_thread = threading.Thread(target=self._read_stdout)
        self.stdout_thread.daemon = True
        self.stdout_thread.start()

        logger.debug('RealtimeSTT initialization completed successfully')
                   
    # Class methods would go here
    def _start_thread(self, target=None, args=()):
        """Start a new thread or process depending on platform."""
        import platform
        
        if platform.system() == 'Linux':
            thread = threading.Thread(target=target, args=args)
            thread.daemon = True
            thread.start()
            return thread
        else:
            process = mp.Process(target=target, args=args)
            process.start()
            return process
            
    def _transcription_worker(*args, **kwargs):
        """Static method to start the transcription worker."""
        worker = TranscriptionWorker(*args, **kwargs)
        worker.run()
        
    @staticmethod
    def _audio_data_worker(audio_queue, target_sample_rate, buffer_size, input_device_index, 
                          shutdown_event, interrupt_stop_event, use_microphone):
        """Worker to handle audio input."""
        # Implementation would go here - using pyaudio to record from microphone
        # For brevity, this implementation is simplified
        try:
            import pyaudio
            import numpy as np
            from scipy import signal
            
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
            
            logger.info(f"Audio recording initialized at {target_sample_rate} Hz")
            
            # Main recording loop
            while not shutdown_event.is_set():
                if use_microphone.value:
                    try:
                        data = stream.read(buffer_size, exception_on_overflow=False)
                        audio_queue.put(data)
                    except Exception as e:
                        logger.error(f"Error reading audio: {e}")
                else:
                    time.sleep(0.01)
                    
        except KeyboardInterrupt:
            interrupt_stop_event.set()
            logger.debug("Audio worker interrupted")
        finally:
            if 'stream' in locals() and stream:
                stream.stop_stream()
                stream.close()
            if 'audio_interface' in locals() and audio_interface:
                audio_interface.terminate()
                
    def _read_stdout(self):
        """Read messages from the stdout pipe."""
        while not self.shutdown_event.is_set():
            try:
                if self.parent_stdout_pipe.poll(0.1):
                    message = self.parent_stdout_pipe.recv()
                    logger.info(message)
            except:
                pass
            time.sleep(0.1)
    
    def _recording_worker(self):
        """Process audio data and detect voice activity."""
        # This would be a complex implementation handling VAD, wake words, etc.
        # For brevity, this is simplified
        pass
    
    def _realtime_worker(self):
        """Handle real-time transcription if enabled."""
        try:
            logger.debug('Starting realtime worker')
            
            if not self.enable_realtime_transcription:
                return
                
            while self.is_running:
                if self.is_recording:
                    time.sleep(self.realtime_processing_pause)
                    
                    # Convert frames to audio array
                    if not self.frames:
                        continue
                        
                    audio_array = np.frombuffer(b''.join(self.frames), dtype=np.int16)
                    audio_array = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
                    
                    # Perform transcription
                    if self.use_main_model_for_realtime:
                        # Use main model
                        with self.transcription_lock:
                            self.parent_transcription_pipe.send((audio_array, self.language))
                            # ... handle response ...
                    else:
                        # Use dedicated realtime model
                        if self.realtime_batch_size > 0:
                            segments, info = self.realtime_model.transcribe(
                                audio_array,
                                language=self.language if self.language else None,
                                beam_size=self.beam_size_realtime,
                                initial_prompt=self.initial_prompt_realtime,
                                suppress_tokens=self.suppress_tokens,
                                batch_size=self.realtime_batch_size,
                                vad_filter=self.faster_whisper_vad_filter  # Use VAD filter parameter
                            )
                        else:
                            segments, info = self.realtime_model.transcribe(
                                audio_array,
                                language=self.language if self.language else None,
                                beam_size=self.beam_size_realtime,
                                initial_prompt=self.initial_prompt_realtime,
                                suppress_tokens=self.suppress_tokens,
                                vad_filter=self.faster_whisper_vad_filter  # Use VAD filter parameter
                            )
                            
                        # Process transcription results
                        # ...
                else:
                    time.sleep(TIME_SLEEP)
                    
        except Exception as e:
            logger.error(f"Error in realtime worker: {e}", exc_info=True)
            
    def start(self):
        """Start recording audio."""
        logger.info("Recording started")
        self._set_state("recording")
        self.frames = []
        self.is_recording = True
        self.recording_start_time = time.time()
        self.stop_recording_event.clear()
        self.start_recording_event.set()
        
        if self.on_recording_start:
            self.on_recording_start()
            
        return self
        
    def stop(self):
        """Stop recording audio."""
        logger.info("Recording stopped")
        self.is_recording = False
        self.recording_stop_time = time.time()
        self.start_recording_event.clear()
        self.stop_recording_event.set()
        
        if self.on_recording_stop:
            self.on_recording_stop()
            
        return self
        
    def text(self):
        """Transcribe recorded audio to text."""
        self.interrupt_stop_event.clear()
        self.was_interrupted.clear()
        
        # Wait for audio recording to complete
        try:
            self._wait_audio()
        except KeyboardInterrupt:
            logger.info("Interrupted while waiting for audio")
            self.shutdown()
            raise
            
        if self.is_shut_down or self.interrupt_stop_event.is_set():
            if self.interrupt_stop_event.is_set():
                self.was_interrupted.set()
            return ""
            
        # Transcribe the audio
        return self._transcribe()
        
    def _wait_audio(self):
        """Wait for audio recording to complete."""
        # Implementation would wait for voice activity and recording to finish
        pass
        
    def _transcribe(self):
        """Perform transcription on the recorded audio."""
        # Implementation would send audio to the transcription model
        pass
        
    def _set_state(self, new_state):
        """Update the recorder state and update UI if needed."""
        if new_state == self.state:
            return
            
        logger.info(f"State changed from '{self.state}' to '{new_state}'")
        self.state = new_state
        
        # Update spinner if needed
        self._set_spinner(new_state)
        
    def _set_spinner(self, text):
        """Update the spinner display."""
        if self.spinner:
            if self.halo is None:
                self.halo = halo.Halo(text=text)
                self.halo.start()
            else:
                self.halo.text = text
                
    def shutdown(self):
        """Clean up resources and shut down."""
        with self.shutdown_lock:
            if self.is_shut_down:
                return
                
            logger.info("Shutting down")
            self.is_shut_down = True
            self.shutdown_event.set()
            self.is_recording = False
            self.is_running = False
            
            # Clean up threads and processes
            # ...
            
    def __enter__(self):
        """Support context manager protocol."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up on context exit."""
        self.shutdown()


def on_recording_start():
    """Called when recording begins."""
    print("\n🎤 Recording Started")

def on_recording_stop():
    """Called when recording stops."""
    print("\n⏹️ Recording Stopped")

def on_transcription_start():
    """Called when final transcription begins."""
    print("\n🔄 Processing final transcription...")

def on_vad_start():
    """Called when voice activity is detected."""
    print("\n🗣️ Voice Detected")

def on_vad_stop():
    """Called when voice activity ends."""
    print("\n🔇 Voice Ended")

def on_realtime_update(text):
    """Called during real-time transcription updates."""
    print(f"\r🎯 Real-time: {text}", end="")

def on_realtime_stabilized(text):
    """Called when transcription stabilizes."""
    print(f"\n📌 Stabilized: {text}")

def detect_device():
    """Detect if CUDA is available and return appropriate device settings."""
    try:
        if torch.cuda.is_available():
            print("🔥 CUDA available. Using GPU for transcription.")
            device = "cuda"
            compute_type = "float16"  # More efficient for GPU
            gpu_index = 0  # Default GPU index
        else:
            print("🖥️ CUDA not available. Using CPU for transcription.")
            device = "cpu"
            compute_type = "int8"  # More efficient for CPU
            gpu_index = None
    except ImportError:
        print("⚠️ PyTorch not found or CUDA detection failed, falling back to CPU.")
        device = "cpu"
        compute_type = "int8"
        gpu_index = None
    
    return device, compute_type, gpu_index

def parse_arguments():
    """Parse command line arguments with defaults from constants."""
    parser = argparse.ArgumentParser(description="Real-time speech-to-text transcription with RealtimeSTT")
    
    # Basic configuration
    parser.add_argument(
        "--model", 
        default=INIT_MODEL_TRANSCRIPTION, 
        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2"],
        help="Model size for final transcription"
    )
    parser.add_argument(
        "--download-root", 
        default=None, 
        help="Root path for downloading models"
    )
    parser.add_argument(
        "--language", 
        default="", 
        help="Language code (blank for auto-detection)"
    )
    parser.add_argument(
        "--input-device", 
        type=int, 
        default=None, 
        help="Audio input device index"
    )
    parser.add_argument(
        "--gpu-device-index", 
        type=int, 
        default=0, 
        help="GPU device index to use"
    )
    
    # Real-time transcription
    parser.add_argument(
        "--no-realtime", 
        action="store_true", 
        help="Disable real-time transcription"
    )
    parser.add_argument(
        "--realtime-model", 
        default=INIT_MODEL_TRANSCRIPTION_REALTIME, 
        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en"],
        help="Model size for real-time transcription"
    )
    parser.add_argument(
        "--use-main-model-realtime", 
        action="store_true", 
        help="Use main model for real-time transcription (saves memory)"
    )
    parser.add_argument(
        "--realtime-pause", 
        type=float, 
        default=INIT_REALTIME_PROCESSING_PAUSE, 
        help="Pause between real-time transcription updates (seconds)"
    )
    parser.add_argument(
        "--initial-realtime-delay", 
        type=float, 
        default=INIT_REALTIME_INITIAL_PAUSE, 
        help="Initial delay before first real-time update (seconds)"
    )
    
    # Voice Activity Detection (VAD)
    parser.add_argument(
        "--silero-sensitivity", 
        type=float, 
        default=INIT_SILERO_SENSITIVITY, 
        help="Silero VAD sensitivity (0-1)"
    )
    parser.add_argument(
        "--silero-onnx", 
        action="store_true", 
        help="Use ONNX format for Silero model (faster)"
    )
    parser.add_argument(
        "--silero-deactivity", 
        action="store_true", 
        help="Enable Silero for deactivity detection (more robust against background noise)"
    )
    parser.add_argument(
        "--webrtc-sensitivity", 
        type=int, 
        default=INIT_WEBRTC_SENSITIVITY, 
        choices=[0, 1, 2, 3], 
        help="WebRTC VAD sensitivity (0-3, 0 most sensitive)"
    )
    parser.add_argument(
        "--silence-duration", 
        type=float, 
        default=INIT_POST_SPEECH_SILENCE_DURATION, 
        help="Duration of silence to end recording (seconds)"
    )
    parser.add_argument(
        "--min-recording-length", 
        type=float, 
        default=INIT_MIN_LENGTH_OF_RECORDING, 
        help="Minimum duration of recording (seconds)"
    )
    parser.add_argument(
        "--min-gap-recordings", 
        type=float, 
        default=INIT_MIN_GAP_BETWEEN_RECORDINGS, 
        help="Minimum time between recordings (seconds)"
    )
    parser.add_argument(
        "--pre-recording-buffer", 
        type=float, 
        default=INIT_PRE_RECORDING_BUFFER_DURATION, 
        help="Duration of pre-recording buffer (seconds)"
    )
    
    # Wake word detection
    parser.add_argument(
        "--wake-words", 
        default="", 
        help="Comma-separated wake words (e.g. 'computer,jarvis')"
    )
    parser.add_argument(
        "--wake-word-backend", 
        default="pvporcupine", 
        choices=["pvporcupine", "openwakeword"], 
        help="Wake word detection backend"
    )
    parser.add_argument(
        "--wake-word-sensitivity", 
        type=float, 
        default=INIT_WAKE_WORDS_SENSITIVITY, 
        help="Wake word detection sensitivity (0-1)"
    )
    parser.add_argument(
        "--wake-word-delay", 
        type=float, 
        default=INIT_WAKE_WORD_ACTIVATION_DELAY, 
        help="Delay before activating wake word detection (seconds)"
    )
    parser.add_argument(
        "--wake-word-timeout", 
        type=float, 
        default=INIT_WAKE_WORD_TIMEOUT, 
        help="Timeout after wake word detection (seconds)"
    )
    parser.add_argument(
        "--wake-word-buffer", 
        type=float, 
        default=INIT_WAKE_WORD_BUFFER_DURATION, 
        help="Duration of wake word buffer (seconds)"
    )
    parser.add_argument(
        "--openwakeword-models", 
        default=None, 
        help="Comma-separated paths to OpenWakeWord model files"
    )
    parser.add_argument(
        "--openwakeword-framework", 
        default="onnx", 
        choices=["onnx", "tflite"], 
        help="Inference framework for OpenWakeWord"
    )
    
    # Performance tuning
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=DEFAULT_BATCH_SIZE, 
        help="Batch size for main transcription model"
    )
    parser.add_argument(
        "--realtime-batch-size", 
        type=int, 
        default=DEFAULT_REALTIME_BATCH_SIZE, 
        help="Batch size for real-time transcription model"
    )
    parser.add_argument(
        "--beam-size", 
        type=int, 
        default=DEFAULT_BEAM_SIZE, 
        help="Beam size for main transcription model"
    )
    parser.add_argument(
        "--beam-size-realtime", 
        type=int, 
        default=DEFAULT_BEAM_SIZE_REALTIME, 
        help="Beam size for real-time transcription model"
    )
    parser.add_argument(
        "--buffer-size", 
        type=int, 
        default=BUFFER_SIZE, 
        help="Audio buffer size (bytes)"
    )
    parser.add_argument(
        "--sample-rate", 
        type=int, 
        default=SAMPLE_RATE, 
        help="Audio sample rate (Hz)"
    )
    parser.add_argument(
        "--initial-prompt", 
        default=None, 
        help="Initial prompt for main transcription model"
    )
    parser.add_argument(
        "--initial-prompt-realtime", 
        default=None, 
        help="Initial prompt for real-time transcription model"
    )
    parser.add_argument(
        "--early-transcription-silence", 
        type=int, 
        default=0, 
        help="Milliseconds of silence to start early transcription"
    )
    parser.add_argument(
        "--latency-limit", 
        type=int, 
        default=ALLOWED_LATENCY_LIMIT, 
        help="Maximum allowed latency in chunks"
    )
    parser.add_argument(
        "--no-whisper-vad", 
        action="store_true", 
        help="Disable faster_whisper VAD filter"
    )
    
    # Logging and debugging
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    parser.add_argument(
        "--no-log-file", 
        action="store_true", 
        help="Disable log file creation"
    )
    parser.add_argument(
        "--extended-logging", 
        action="store_true", 
        help="Enable extended logging"
    )
    parser.add_argument(
        "--show-transcription-time", 
        action="store_true", 
        help="Show transcription processing time"
    )
    parser.add_argument(
        "--no-buffer-overflow-handling", 
        action="store_true", 
        help="Disable buffer overflow handling"
    )
    parser.add_argument(
        "--no-spinner", 
        action="store_true", 
        help="Disable spinner animation"
    )
    
    # Text formatting
    parser.add_argument(
        "--no-uppercase-start", 
        action="store_true", 
        help="Disable forcing uppercase at start of sentences"
    )
    parser.add_argument(
        "--no-period-end", 
        action="store_true", 
        help="Disable adding period at end of sentences"
    )

    return parser.parse_args()

def main():
    """Main function to initialize and run the transcription process."""
    args = parse_arguments()
    
    # Set up logging based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    print("\n" + "="*60)
    print("🎤  RealtimeSTT Microphone Test")
    print("="*60)
    print("Speak into your microphone to test transcription")
    print(f"Silence for {args.silence_duration} seconds will end recording")
    print("Press Ctrl+C at any time to exit")
    print("="*60 + "\n")
    
    # Detect device (GPU or CPU)
    device, compute_type, gpu_idx = detect_device()
    if args.gpu_device_index is not None:
        gpu_idx = args.gpu_device_index
    
    print(f"Configuration:")
    print(f"  • Main model: {args.model}")
    print(f"  • Real-time: {'Disabled' if args.no_realtime else args.realtime_model}")
    print(f"  • Device: {device} ({compute_type}), GPU idx: {gpu_idx}")
    print(f"  • Language: {args.language if args.language else 'Auto-detect'}")
    print(f"  • VAD: Silero={args.silero_sensitivity}, WebRTC={args.webrtc_sensitivity}")
    print(f"  • Silence duration: {args.silence_duration}s")
    if args.wake_words:
        print(f"  • Wake words: {args.wake_words}")
    print(f"  • Faster Whisper VAD filter: {'Disabled' if args.no_whisper_vad else 'Enabled'}")
    print("="*60 + "\n")
    
    # Configure parameters for AudioToTextRecorder
    params = {
        # Model configuration
        "model": args.model,
        "download_root": args.download_root,
        "language": args.language,
        "device": device,
        "compute_type": compute_type,
        "input_device_index": args.input_device,
        "gpu_device_index": gpu_idx,
        
        # Audio configuration
        "use_microphone": True,
        "sample_rate": args.sample_rate,
        "buffer_size": args.buffer_size,
        
        # Real-time transcription configuration
        "enable_realtime_transcription": not args.no_realtime,
        "realtime_model_type": args.realtime_model,
        "use_main_model_for_realtime": args.use_main_model_realtime,
        "realtime_processing_pause": args.realtime_pause,
        "init_realtime_after_seconds": args.initial_realtime_delay,
        "on_realtime_transcription_update": on_realtime_update,
        "on_realtime_transcription_stabilized": on_realtime_stabilized,
        "realtime_batch_size": args.realtime_batch_size,
        
        # Voice activity detection configuration
        "silero_sensitivity": args.silero_sensitivity,
        "silero_use_onnx": args.silero_onnx,
        "silero_deactivity_detection": args.silero_deactivity,
        "webrtc_sensitivity": args.webrtc_sensitivity,
        "post_speech_silence_duration": args.silence_duration,
        "min_length_of_recording": args.min_recording_length,
        "min_gap_between_recordings": args.min_gap_recordings,
        "pre_recording_buffer_duration": args.pre_recording_buffer,
        
        # Wake word configuration
        "wakeword_backend": args.wake_word_backend,
        "openwakeword_model_paths": args.openwakeword_models,
        "openwakeword_inference_framework": args.openwakeword_framework,
        "wake_words": args.wake_words,
        "wake_words_sensitivity": args.wake_word_sensitivity,
        "wake_word_activation_delay": args.wake_word_delay,
        "wake_word_timeout": args.wake_word_timeout,
        "wake_word_buffer_duration": args.wake_word_buffer,
        
        # Status callbacks
        "on_recording_start": on_recording_start,
        "on_recording_stop": on_recording_stop,
        "on_transcription_start": on_transcription_start,
        "on_vad_start": on_vad_start,
        "on_vad_stop": on_vad_stop,
        
        # Text formatting
        "ensure_sentence_starting_uppercase": not args.no_uppercase_start,
        "ensure_sentence_ends_with_period": not args.no_period_end,
        
        # Performance settings
        "beam_size": args.beam_size,
        "beam_size_realtime": args.beam_size_realtime,
        "batch_size": args.batch_size,
        "initial_prompt": args.initial_prompt,
        "initial_prompt_realtime": args.initial_prompt_realtime,
        "suppress_tokens": [-1],  # Default from original
        "early_transcription_on_silence": args.early_transcription_silence,
        "allowed_latency_limit": args.latency_limit,
        "handle_buffer_overflow": not args.no_buffer_overflow_handling,
        "faster_whisper_vad_filter": not args.no_whisper_vad,  # Apply the VAD filter parameter
        
        # UI, logging and performance
        "spinner": not args.no_spinner,
        "level": log_level,
        "debug_mode": args.debug,
        "use_extended_logging": args.extended_logging,
        "no_log_file": args.no_log_file,
        "print_transcription_time": args.show_transcription_time,
    }
    
    try:
        print("Initializing AudioToTextRecorder...\n")
        with AudioToTextRecorder(**params) as recorder:
            print("\n🎧 Listening... (Press Ctrl+C to stop)")
            
            # Main loop for continuous transcription
            while True:
                # Get final transcription
                result = recorder.text()
                if result:
                    print(f"\n\n✅ Final Transcription: {result}\n")
                    
                    # Print detected language if any
                    if recorder.detected_language:
                        print(f"🌐 Detected language: {recorder.detected_language} "
                              f"(confidence: {recorder.detected_language_probability:.2f})")
                    
                    print("\n🎧 Listening for next input... (Press Ctrl+C to stop)")
                
                # Small pause to prevent CPU overload
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n\n👋 Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 