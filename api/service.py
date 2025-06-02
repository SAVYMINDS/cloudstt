import os
import logging
import uuid
import time
import tempfile
import json
import torch
import numpy as np
import asyncio
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, BinaryIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from multiprocessing import Process, Queue
import queue
import io
from pydantic import BaseModel
import math

# Instead of importing schemas directly, import specific classes only when needed
# Remove the circular import
# from .schemas import (
#     TranscriptionRequest, 
#     TranscriptionResult, 
#     TranscriptionMetadata,
#     AudioBlobSource,
#     AudioUploadSource,
#     WebSocketConfig
# )

# Import existing services
from core.diarization_processor import DiarizationService
from storage.azure_storage import AzureFilesStorage
from RealtimeSTT import AudioToTextRecorder
from .realtime_storage_service import RealtimeStorageService

# Setup logging
logger = logging.getLogger(__name__)

# Configuration constants
FILE_SIZE_THRESHOLD = 700 * 1024 * 1024  # 700MB in bytes
CHUNK_SIZE_MB = 200
THRESHOLD_MB = 700
CHUNK_TIMEOUT = 600  # 10 minutes per chunk
MAX_PARALLEL_CHUNKS = 4

# Move the function outside the class to make it picklable
def run_diarization_process(job_id: str, audio_path: str, model_name: str, 
                            language: Optional[str], result_queue: Queue, compute_type: str = "float32"):
    """Run the diarization process in a separate process"""
    try:
        # Get Hugging Face token from environment variable
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        
        # Initialize service
        diarization_service = DiarizationService(
            default_model=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type=compute_type,  # Use the passed compute_type
            batch_size=8,
        )
        
        logger.info(f"Processing request {job_id}: {audio_path}")
        
        # Process the audio file
        result = diarization_service.process_audio_file(
            file_path=audio_path,
            model=model_name,
            language=None if language == "auto" else language,
            hf_token=hf_token,
        )
        
        # Put result in queue
        result_queue.put({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Error in processing subprocess: {str(e)}")
        # Put error in queue
        result_queue.put({"success": False, "error": str(e)})

# Create a persistent job storage that saves to disk
class PersistentJobStorage:
    def __init__(self, storage_dir=None):
        self.storage_dir = storage_dir or os.path.join(os.getcwd(), 'local_storage', 'jobs')
        os.makedirs(self.storage_dir, exist_ok=True)
        self.memory_cache = {}
        logger.info(f"Initialized persistent job storage in {self.storage_dir}")
    
    def _get_job_path(self, job_id):
        return os.path.join(self.storage_dir, f"{job_id}.json")
    
    def __contains__(self, job_id):
        # Check memory cache first
        if job_id in self.memory_cache:
            return True
        # Then check file system
        return os.path.exists(self._get_job_path(job_id))
    
    def __getitem__(self, job_id):
        # Check memory cache first
        if job_id in self.memory_cache:
            return self.memory_cache[job_id]
            
        # Load from disk if not in memory
        job_path = self._get_job_path(job_id)
        if not os.path.exists(job_path):
            raise KeyError(f"Job {job_id} not found")
            
        try:
            with open(job_path, 'r') as f:
                data = json.load(f)
                # Update memory cache
                self.memory_cache[job_id] = data
                return data
        except Exception as e:
            logger.error(f"Error loading job {job_id}: {str(e)}")
            raise KeyError(f"Job {job_id} could not be loaded: {str(e)}")
    
    def __setitem__(self, job_id, value):
        # Update memory cache
        self.memory_cache[job_id] = value
        
        # Save to disk
        job_path = self._get_job_path(job_id)
        try:
            with open(job_path, 'w') as f:
                json.dump(value, f, indent=2, default=str)
            logger.info(f"Saved job {job_id} to disk")
        except Exception as e:
            logger.error(f"Error saving job {job_id}: {str(e)}")
    
    def get(self, job_id, default=None):
        try:
            return self[job_id]
        except KeyError:
            return default
    
    def update_job(self, job_id, updates, create_if_missing=False):
        """Update specific fields of a job"""
        try:
            if job_id in self:
                job_data = self[job_id]
                # Update the dictionary
                if isinstance(updates, dict):
                    self._deep_update(job_data, updates)
                    # Save updated job
                    self[job_id] = job_data
                    return True
            elif create_if_missing:
                self[job_id] = updates
                return True
            return False
        except Exception as e:
            logger.error(f"Error updating job {job_id}: {str(e)}")
            return False
    
    def _deep_update(self, d, u):
        """Recursively update nested dictionaries"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v

# Create persistent job storage
job_storage = PersistentJobStorage()

class UnifiedTranscriptionService:
    """Unified service for both batch and real-time transcription"""
    
    def __init__(self):
        self.storage_client = AzureFilesStorage()
        self.active_sessions = {}
        self.realtime_storage = RealtimeStorageService(self.storage_client)  # Enhanced realtime storage
        
    # ========================
    # Shared utility methods
    # ========================
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a job by ID"""
        if job_id in job_storage:
            return job_storage[job_id]
        return None
    
    def cleanup_job_files(self, job_id: str):
        """Clean up intermediate results but preserve audio files."""
        try:
            logger.info(f"Skipping cleanup of audio files for job {job_id} to preserve local files")
            
            # Do not clean up temp files
            # if job_id in job_storage:
            #     job_data = job_storage[job_id]
            #     if "metadata" in job_data:
            #         # Clean up single file
            #         if "temp_path" in job_data["metadata"]:
            #             temp_path = job_data["metadata"]["temp_path"]
            #             if os.path.exists(temp_path):
            #                 os.unlink(temp_path)
            #         
            #         # Clean up chunk files
            #         if "temp_paths" in job_data["metadata"]:
            #             for temp_path in job_data["metadata"]["temp_paths"]:
            #                 if os.path.exists(temp_path):
            #                     os.unlink(temp_path)
            
            # Don't clean up processing blobs
            # proc_blobs = self.storage_client.list_blobs(
            #     container_name="audio-processing",
            #     prefix=f"job-{job_id}/"
            # )
            # for blob in proc_blobs:
            #     self.storage_client.delete_blob("audio-processing", blob)
            
            # Keep the final result and all intermediate files
            # output_blobs = self.storage_client.list_blobs(
            #     container_name="audio-output",
            #     prefix=f"job-{job_id}/chunks/"
            # )
            # for blob in output_blobs:
            #     # Make sure we're not accidentally deleting the final result
            #     if "final_result.json" not in blob:
            #         self.storage_client.delete_blob("audio-output", blob)
            #         logger.info(f"Deleted chunk file: {blob}")
            #     else:
            #         logger.warning(f"Skipped deletion of final result file: {blob}")
                
        except Exception as e:
            logger.error(f"Error in cleanup_job_files for job {job_id}: {str(e)}")
    
    # =============================
    # Batch transcription methods
    # =============================
    
    def process_audio(self, job_id: str, audio_path: str, model_name: str, 
                     is_chunk: bool = False, chunk_info: Dict = None, 
                     language: Optional[str] = None, timeout: int = 600,
                     compute_type: str = "float32"):
        """Process audio files with parallel processing support."""
        try:
            start_time = time.time()
            result_queue = multiprocessing.Queue()
            
            # Check if there are active WebSocket sessions
            has_active_sessions = len(self.active_sessions) > 0
            if has_active_sessions:
                logger.info(f"Active WebSocket sessions detected while processing batch job {job_id}. Adjusting resource allocation.")
            
            # Set CUDA device based on chunk number if using multiple GPUs
            if is_chunk and chunk_info:
                chunk_num = chunk_info["chunk_number"]
                num_gpus = torch.cuda.device_count()
                if num_gpus > 1:
                    # If we have active sessions, use a different GPU if possible
                    if has_active_sessions and num_gpus > 1:
                        gpu_id = num_gpus - 1  # Use the last GPU for batch if WebSocket is active
                    else:
                        gpu_id = (chunk_num - 1) % num_gpus
                    
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    logger.info(f"Processing chunk {chunk_num} on GPU {gpu_id}")
            elif has_active_sessions and torch.cuda.device_count() > 1:
                # For non-chunk processing with active sessions, use a different GPU
                gpu_id = torch.cuda.device_count() - 1
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                logger.info(f"Processing batch job on GPU {gpu_id} to preserve WebSocket connections")
            
            logger.info(f"Processing {'chunk' if is_chunk else 'file'} at {audio_path}")
            
            # Start processing
            process = multiprocessing.Process(
                target=run_diarization_process, 
                args=(job_id, audio_path, model_name, language, result_queue, compute_type)
            )
            process.daemon = True
            process.start()
            
            # Wait for completion with periodic checks
            while time.time() - start_time < timeout:
                try:
                    # Check for results every 5 seconds
                    if process.is_alive():
                        process.join(timeout=5)
                    
                    # Check if result is available
                    if not result_queue.empty():
                        result_data = result_queue.get_nowait()
                        if result_data.get("success", False):
                            # Save results to Azure only for chunks, not for single files
                            if is_chunk:
                                result_blob_name = f"job-{job_id}/chunks/chunk_{chunk_info['chunk_number']}_result.json"
                                
                                # Add chunk metadata if needed
                                result_data["data"].update({
                                    "chunk_number": chunk_info["chunk_number"],
                                    "total_chunks": chunk_info["total_chunks"]
                                })
                                
                                # Save to Azure Files (only for chunks)
                                self.storage_client.save_json(
                                    container_name=self.storage_client.BATCH_OUTPUT,
                                    blob_name=result_blob_name,
                                    data=result_data["data"]
                                )
                            
                            # Update job status
                            if job_id in job_storage:
                                if is_chunk:
                                    parent_job = job_storage[job_id]
                                    parent_job["metadata"]["completed_chunks"] += 1
                                    logger.info(f"Completed chunk {chunk_info['chunk_number']}/{chunk_info['total_chunks']} for job {job_id}")
                                    
                                    # If all chunks complete, trigger merge
                                    if parent_job["metadata"]["completed_chunks"] == parent_job["metadata"]["total_chunks"]:
                                        logger.info(f"All chunks completed for job {job_id}, merging results")
                                        merged_result = self.merge_chunk_results(job_id)
                                        if merged_result:
                                            parent_job.update({
                                                "result": merged_result,
                                                "metadata": {
                                                    **parent_job["metadata"],
                                                    "status": "completed"
                                                }
                                            })
                                            logger.info(f"Successfully merged results for job {job_id}")
                                        else:
                                            parent_job["metadata"]["status"] = "error"
                                            parent_job["metadata"]["error"] = "Failed to merge results"
                                            logger.error(f"Failed to merge results for job {job_id}")
                                else:
                                    # For single files, we don't update the job_storage here
                                    # This will be handled in process_and_update instead
                                    logger.info(f"Completed single file processing for job {job_id}")
                            
                            # Clean up
                            process.terminate()
                            # Only delete if it's a temporary file (not Azure Files storage)
                            if os.path.exists(audio_path) and not audio_path.startswith('/app/storage'):
                                os.unlink(audio_path)
                                logger.info(f"Removed temporary file {audio_path}")
                            else:
                                logger.info(f"Preserved Azure Files storage file: {audio_path}")
                            
                            # Extract number of speakers if available
                            num_speakers = 0
                            if "num_speakers" in result_data["data"]:
                                try:
                                    num_speakers = int(result_data["data"]["num_speakers"])
                                    logger.info(f"Number of speakers detected: {num_speakers}")
                                except (ValueError, TypeError) as e:
                                    logger.error(f"Error converting num_speakers: {e}")
                            
                            # Create the organized result with the proper structure, matching multi-chunk output
                            organized_result = {
                                "metadata": {
                                    "job_id": job_id,  # Using job_id instead of request_id for consistency
                                    "status": "completed",
                                    "merged_at": datetime.now().isoformat(),  # Add this field for single files too
                                    "created_at": job_storage[job_id]["metadata"]["created_at"],
                                    "total_processing_time_seconds": time.time() - start_time,
                                    "workers_used": 1  # Always 1 for single file
                                },
                                "model_info": job_storage[job_id]["model_info"],
                                "total_chunks": 1,  # Always 1 for single file
                                "total_duration": result_data["data"].get("total_duration", 0),
                                "num_speakers_detected": num_speakers,
                                "transcript": result_data["data"].get("transcript", ""),
                                "segments": result_data["data"].get("segments", [])
                            }
                            
                            return organized_result
                    
                    if not process.is_alive():
                        break
                        
                except queue.Empty:
                    continue
                
            # Handle timeout or process death
            if process.is_alive():
                logger.error(f"Processing timed out after {timeout} seconds for {'chunk ' + str(chunk_info['chunk_number']) if is_chunk else 'file'} in job {job_id}")
                process.terminate()
                process.join(1)
                if process.is_alive():
                    os.kill(process.pid, 9)
                
                if job_id in job_storage:
                    if is_chunk:
                        chunk_str = f" (chunk {chunk_info['chunk_number']}/{chunk_info['total_chunks']})"
                    else:
                        chunk_str = ""
                        
                    job_storage[job_id]["metadata"].update({
                        "status": "error",
                        "error": f"Processing timed out after {timeout} seconds{chunk_str}"
                    })
            
            # Clean up - only delete temporary files, not Azure Files storage
            if os.path.exists(audio_path) and not audio_path.startswith('/app/storage'):
                os.unlink(audio_path)
                logger.info(f"Removed temporary file {audio_path}")
            else:
                logger.info(f"Preserved Azure Files storage file: {audio_path}")
                
        except Exception as e:
            logger.error(f"Error in process_audio: {str(e)}", exc_info=True)
            if job_id in job_storage:
                job_storage[job_id]["metadata"].update({
                    "status": "error",
                    "error": str(e)
                })
            # Clean up - only delete temporary files, not Azure Files storage
            if os.path.exists(audio_path) and not audio_path.startswith('/app/storage'):
                os.unlink(audio_path)
                logger.info(f"Removed temporary file {audio_path}")
            else:
                logger.info(f"Preserved Azure Files storage file: {audio_path}")

        # At the end of the function, return a structured empty result if we didn't return earlier
        return {
            "metadata": {
                "job_id": job_id,
                "status": "error",
                "error": "Processing completed but no valid result was produced",
                "total_processing_time_seconds": time.time() - start_time,
            },
            "model_info": job_storage[job_id]["model_info"] if job_id in job_storage else {},
            "total_chunks": 1,
            "total_duration": 0,
            "num_speakers_detected": 0,
            "transcript": "",
            "segments": []
        }
    
    def merge_chunk_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Merge results from all chunks of a job."""
        try:
            start_time = time.time()  # Add this to measure processing time
            
            # Get all chunk results
            chunk_results = []
            
            # Get the total_chunks from job_storage
            if job_id not in job_storage or "metadata" not in job_storage[job_id]:
                logger.error(f"Job {job_id} not found or missing metadata")
                return None
                
            total_chunks = job_storage[job_id]["metadata"]["total_chunks"]
            logger.info(f"Starting merge for job {job_id} with {total_chunks} chunks")
            
            # List all results in the job's chunks folder
            result_files = self.storage_client.list_blobs(
                container_name=self.storage_client.BATCH_OUTPUT,
                prefix=f"job-{job_id}/chunks/"
            )
            
            # Log what files we found
            logger.info(f"Found {len(result_files)} files for job {job_id}: {result_files}")
            
            # Load chunk results and check if they have duration
            chunk_result_files = [f for f in result_files if f.endswith('_result.json')]
            logger.info(f"Found {len(chunk_result_files)} result files for job {job_id}")
            
            for result_file in chunk_result_files:
                chunk_data = self.storage_client.load_json(
                    container_name=self.storage_client.BATCH_OUTPUT,
                    blob_name=result_file
                )
                if chunk_data:
                    # Extract chunk number from filename
                    try:
                        file_name = result_file.split('/')[-1]
                        chunk_number = int(file_name.split('_')[1])
                        chunk_data["chunk_number"] = chunk_number
                        chunk_results.append(chunk_data)
                        logger.info(f"Loaded chunk {chunk_number} result from {result_file}")
                    except Exception as e:
                        logger.error(f"Error parsing chunk number from {result_file}: {str(e)}")
                        
                    # Debug log to check if each chunk has a duration
                    if "total_duration" in chunk_data:
                        logger.info(f"Found duration in chunk {chunk_number}: {chunk_data.get('total_duration')}")
                    else:
                        logger.warning(f"No duration found in chunk from file: {result_file}")
            
            # Handling missing chunks
            if len(chunk_results) != total_chunks:
                logger.error(f"Missing chunk results for job {job_id}. Expected {total_chunks}, got {len(chunk_results)}")
                
                # Even if we're missing some chunks, try to proceed with what we have
                if len(chunk_results) == 0:
                    raise Exception(f"No chunk results found. Expected {total_chunks}")
                
                logger.warning(f"Proceeding with partial merge using {len(chunk_results)} chunks")
            
            # Sort chunks by number
            chunk_results.sort(key=lambda x: x.get("chunk_number", 0))
            
            # Initialize merged result
            merged_result = {
                "segments": [],
                "transcript": "",
                "metadata": {
                    "job_id": job_id,
                    "status": "completed",
                    "merged_at": datetime.now().isoformat(),
                    "total_chunks": total_chunks
                }
            }
            
            # Merge segments and transcripts
            current_time_offset = 0
            all_segments = []
            transcript_parts = []
            total_duration = 0  # Initialize total duration
            num_speakers = 0  # Initialize num_speakers
            
            for chunk_data in chunk_results:
                # Add transcript
                if "transcript" in chunk_data:
                    transcript_parts.append(chunk_data["transcript"])
                
                # Calculate total duration from all chunks
                if "total_duration" in chunk_data:
                    try:
                        chunk_duration = float(chunk_data["total_duration"])
                        total_duration += chunk_duration
                        logger.info(f"Added duration {chunk_duration} from chunk, total now: {total_duration}")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error processing duration from chunk: {e}")
                
                # Adjust segment timestamps
                if "segments" in chunk_data:
                    for segment in chunk_data["segments"]:
                        adjusted_segment = segment.copy()
                        adjusted_segment["start"] += current_time_offset
                        adjusted_segment["end"] += current_time_offset
                        
                        # Adjust word timestamps if present
                        if "words" in segment:
                            adjusted_segment["words"] = [
                                {**word, 
                                 "start": word["start"] + current_time_offset,
                                 "end": word["end"] + current_time_offset}
                                for word in segment["words"]
                            ]
                        
                        all_segments.append(adjusted_segment)
                    
                    # Update time offset based on the end time of the last segment
                    if chunk_data["segments"]:
                        current_time_offset = adjusted_segment["end"]
                        logger.info(f"Updated time offset to {current_time_offset} based on last segment")
                
                # Extract number of speakers if available
                if "num_speakers" in chunk_data:
                    try:
                        chunk_speakers = int(chunk_data["num_speakers"])
                        num_speakers = max(num_speakers, chunk_speakers)
                        logger.info(f"Found {chunk_speakers} speakers in chunk, max now: {num_speakers}")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error processing num_speakers from chunk: {e}")
            
            # Handle edge cases for duration and speaker count
            if total_duration == 0 and all_segments:
                total_duration = all_segments[-1]["end"]
                logger.info(f"Calculated total duration from last segment: {total_duration}")
            
            if total_duration == 0:
                # Try to extract duration from the original chunk metadata
                for chunk in chunk_results:
                    if "metadata" in chunk and "end_time" in chunk["metadata"] and "start_time" in chunk["metadata"]:
                        try:
                            chunk_duration = float(chunk["metadata"]["end_time"]) - float(chunk["metadata"]["start_time"])
                            total_duration += chunk_duration / 1000  # Convert ms to seconds
                            logger.info(f"Added duration from metadata: {chunk_duration/1000} seconds")
                        except (ValueError, TypeError) as e:
                            logger.error(f"Error calculating duration from metadata: {e}")
            
            if num_speakers == 0 and all_segments:
                speakers = set()
                for segment in all_segments:
                    if "speaker" in segment:
                        speakers.add(segment["speaker"])
                if speakers:
                    num_speakers = len(speakers)
                    logger.info(f"Counted {num_speakers} unique speakers from segments")
            
            # Initialize merged result with metadata at the top
            merged_result = {
                "metadata": {
                    "job_id": job_id,
                    "status": "completed",
                    "merged_at": datetime.now().isoformat(),
                    "created_at": job_storage[job_id]["metadata"]["created_at"],
                    "total_processing_time_seconds": time.time() - start_time,
                    "workers_used": min(MAX_PARALLEL_CHUNKS, total_chunks)  # Use actual worker count
                },
                # Include model_info from job_storage
                "model_info": job_storage[job_id].get("model_info", {}),
                "total_chunks": total_chunks,
                "total_duration": total_duration,
                "num_speakers_detected": num_speakers,  # Always include this field
                "transcript": "\n".join(transcript_parts),
                "segments": all_segments
            }
            
            logger.info(f"Final total duration for job {job_id}: {total_duration}")
            logger.info(f"Merged {len(all_segments)} segments across {len(chunk_results)} chunks")
            
            # Save final result
            final_result_path = f"job-{job_id}/final_result.json"
            self.storage_client.save_json(
                container_name=self.storage_client.BATCH_OUTPUT,
                blob_name=final_result_path,
                data=merged_result
            )
            logger.info(f"Saved final result to {final_result_path}")
            
            # IMPORTANT: Update job storage directly with result to ensure it's available
            if job_id in job_storage:
                job_storage[job_id]["result"] = merged_result
                job_storage[job_id]["metadata"]["status"] = "completed"
                logger.info(f"Updated job storage with merged result for {job_id}")
            
            return merged_result
            
        except Exception as e:
            logger.error(f"Error merging results for job {job_id}: {str(e)}", exc_info=True)
            return None
    
    # ==============================
    # WebSocket/Stream methods
    # ==============================
    
    def save_audio_to_blob(self, audio_data: np.ndarray, session_id: str) -> Optional[str]:
        """Save audio data to Azure Blob Storage and return the URL."""
        try:
            logger.info(f"Saving audio for session {session_id}, data type: {type(audio_data)}")
            
            # Handle different audio data types
            if audio_data is None:
                logger.error("Audio data is None")
                return None
                
            # Create temp directory if it doesn't exist
            os.makedirs("temp_audio", exist_ok=True)
            
            # Local file path
            local_path = f"temp_audio/session_{session_id}.wav"
            
            # Save to local file
            logger.info(f"Saving audio to local file: {local_path}")
            
            # Make sure audio_data is a numpy array with correct format
            if isinstance(audio_data, np.ndarray):
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                    
                # Ensure values are normalized between -1.0 and 1.0
                if np.max(np.abs(audio_data)) > 1.0:
                    audio_data = audio_data / 32768.0  # Normalize from int16 range
                    
                # Save to WAV file using soundfile
                import soundfile as sf
                sf.write(local_path, audio_data, 16000)
            else:
                logger.error(f"Unsupported audio data type: {type(audio_data)}")
                return None
            
            # Upload local file to Azure
            with open(local_path, 'rb') as f:
                blob_name = f"session-{session_id}/audio.wav"
                
                success = self.storage_client.upload_file(
                    container_name=self.storage_client.REALTIME_SESSIONS,
                    blob_name=blob_name,
                    data=f,
                    content_type="audio/wav",
                    metadata={
                        "session_id": session_id,
                        "sample_rate": "16000",
                        "channels": "1"
                    }
                )
            
            if success:
                # Generate URL for the audio file
                audio_url = self.storage_client.get_sas_url(
                    container_name=self.storage_client.REALTIME_SESSIONS,
                    blob_name=blob_name,
                    duration_hours=24
                )
                logger.info(f"Audio saved to Azure Blob Storage: {audio_url}")
                
                # Clean up temp file
                try:
                    os.remove(local_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file: {str(e)}")
                    
                return audio_url
            else:
                logger.error("Failed to upload to Azure Blob Storage")
                return None
                
        except Exception as e:
            logger.error(f"Error saving audio to Azure: {str(e)}")
            return None
    
    def create_audio_recorder(self, config: Dict[str, Any], session_id: str, 
                             callback_functions: Dict[str, callable]) -> AudioToTextRecorder:
        """Create a new AudioToTextRecorder instance with the given configuration"""
        return AudioToTextRecorder(
            model=config.get("model", "tiny"),
            language=config.get("language", ""),
            compute_type=config.get("compute_type", "default"),
            input_device_index=config.get("input_device_index"),
            gpu_device_index=config.get("gpu_device_index", 0),
            device=config.get("device", "cuda"),
            on_recording_start=callback_functions.get("on_recording_start"),
            on_recording_stop=callback_functions.get("on_recording_stop"),
            on_transcription_start=callback_functions.get("on_transcription_start"),
            ensure_sentence_starting_uppercase=config.get("ensure_sentence_starting_uppercase", True),
            ensure_sentence_ends_with_period=config.get("ensure_sentence_ends_with_period", True),
            use_microphone=config.get("use_microphone", True),
            spinner=False,
            
            # Realtime transcription parameters
            enable_realtime_transcription=config.get("enable_realtime_transcription", True),
            use_main_model_for_realtime=config.get("use_main_model_for_realtime", False),
            realtime_model_type=config.get("realtime_model_type", "tiny"),
            realtime_processing_pause=config.get("realtime_processing_pause", 0.2),
            #init_realtime_after_seconds=config.get("init_realtime_after_seconds", 0.2),
            on_realtime_transcription_update=callback_functions.get("on_realtime_transcription_update"),
            on_realtime_transcription_stabilized=callback_functions.get("on_realtime_transcription_stabilized"),
            #realtime_batch_size=config.get("realtime_batch_size", 16),
            
            # Voice activation parameters
            silero_sensitivity=config.get("silero_sensitivity", 0.4),
            silero_use_onnx=config.get("silero_use_onnx", False),
            silero_deactivity_detection=config.get("silero_deactivity_detection", False),
            webrtc_sensitivity=config.get("webrtc_sensitivity", 3),
            post_speech_silence_duration=config.get("post_speech_silence_duration", 0.6),
            min_length_of_recording=config.get("min_length_of_recording", 0.5),
            min_gap_between_recordings=config.get("min_gap_between_recordings", 0),
            pre_recording_buffer_duration=config.get("pre_recording_buffer_duration", 1.0),
            on_vad_detect_start=callback_functions.get("on_vad_detect_start"),
            on_vad_detect_stop=callback_functions.get("on_vad_detect_stop"),
            
            # Wake word parameters
            wakeword_backend=config.get("wakeword_backend", "pvporcupine"),
            openwakeword_model_paths=config.get("openwakeword_model_paths"),
            openwakeword_inference_framework=config.get("openwakeword_inference_framework", "onnx"),
            wake_words=config.get("wake_words", ""),
            wake_words_sensitivity=config.get("wake_words_sensitivity", 0.6),
            wake_word_activation_delay=config.get("wake_word_activation_delay", 0.0),
            wake_word_timeout=config.get("wake_word_timeout", 5.0),
            wake_word_buffer_duration=config.get("wake_word_buffer_duration", 0.1),
            on_wakeword_detected=callback_functions.get("on_wakeword_detected"),
            
            # Additional parameters
            level=logging.INFO,
            debug_mode=config.get("debug_mode", False),
            #beam_size=config.get("beam_size", 5),
            beam_size_realtime=config.get("beam_size_realtime", 3),
            initial_prompt=config.get("initial_prompt"),
            #initial_prompt_realtime=config.get("initial_prompt_realtime"),
            suppress_tokens=config.get("suppress_tokens", [-1]),
            print_transcription_time=config.get("print_transcription_time", False),
            early_transcription_on_silence=config.get("early_transcription_on_silence", 0),
            allowed_latency_limit=config.get("allowed_latency_limit", 100),
            no_log_file=config.get("no_log_file", False),
            use_extended_logging=config.get("use_extended_logging", False),
        )

# Create a singleton instance of the service
transcription_service = UnifiedTranscriptionService()