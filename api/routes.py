from fastapi import APIRouter, HTTPException, BackgroundTasks, Response, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uuid
import os
import logging
from datetime import datetime
import json
import tempfile
import io
import math
from pydub import AudioSegment
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Union
import multiprocessing
import queue
import torch

# Import shared schemas
from .schemas import (
    TranscriptionRequest, 
    TranscriptionResult,
    AudioBlobSource,
    AudioUploadSource,
    ModelConfig,
    ProcessingConfig
)

# Import Azure storage
from storage.azure_storage import AzureFilesStorage

# Import unified service from main
from .main import transcription_service, job_storage
from .service import run_diarization_process

router = APIRouter()
logger = logging.getLogger(__name__)

# Configuration constants
FILE_SIZE_THRESHOLD = 700 * 1024 * 1024  # 700MB in bytes
CHUNK_SIZE_MB = 200
THRESHOLD_MB = 700
CHUNK_TIMEOUT = 600  # 10 minutes per chunk
MAX_PARALLEL_CHUNKS = 4

# We keep these functions for compatibility, but they delegate to the service
def split_audio_file(audio_data: bytes, job_id: str, file_ext: str, chunk_size_mb: int = 200):
    """Split audio file into chunks of specified size."""
    logger.info(f"Splitting audio file for job {job_id} into {chunk_size_mb}MB chunks")
    
    # Load audio using pydub
    with io.BytesIO(audio_data) as audio_io:
        audio = AudioSegment.from_file(audio_io, format=file_ext.replace(".", ""))
    
    # Calculate total duration and chunk duration
    total_duration_ms = len(audio)
    file_size_mb = len(audio_data) / (1024 * 1024)
    
    # Calculate how many ms per MB to determine chunk duration
    ms_per_mb = total_duration_ms / file_size_mb
    chunk_duration_ms = int(chunk_size_mb * ms_per_mb)
    
    # Calculate number of chunks
    num_chunks = math.ceil(total_duration_ms / chunk_duration_ms)
    logger.info(f"File will be split into {num_chunks} chunks of approx. {chunk_size_mb}MB each")
    
    chunks = []
    for i in range(num_chunks):
        start_ms = i * chunk_duration_ms
        end_ms = min((i + 1) * chunk_duration_ms, total_duration_ms)
        
        # Extract chunk
        chunk_audio = audio[start_ms:end_ms]
        
        # Export to bytes
        chunk_buffer = io.BytesIO()
        chunk_audio.export(chunk_buffer, format=file_ext.replace(".", ""))
        chunk_data = chunk_buffer.getvalue()
        
        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        temp_path = temp_file.name
        temp_file.close()
        
        # Write chunk to file
        with open(temp_path, "wb") as f:
            f.write(chunk_data)
        
        # Create storage client
        storage_client = transcription_service.storage_client
        
        # Upload to Azure Files (batch processing)
        chunk_blob_name = f"job-{job_id}/chunks/chunk_{i+1}{file_ext}"
        with open(temp_path, 'rb') as f:
            storage_client.upload_file(
                container_name=storage_client.BATCH_PROCESSING,
                blob_name=chunk_blob_name,
                data=f,
                metadata={
                    "chunk_number": str(i+1),
                    "total_chunks": str(num_chunks),
                    "start_time": str(start_ms),
                    "end_time": str(end_ms)
                }
            )
        
        chunks.append({
            "chunk_number": i+1,
            "total_chunks": num_chunks,
            "temp_path": temp_path,
            "blob_path": f"{storage_client.BATCH_PROCESSING}/{chunk_blob_name}",
            "start_time": start_ms,
            "end_time": end_ms
        })
        
        logger.info(f"Created chunk {i+1}/{num_chunks}")
        
    return chunks

def merge_chunk_results(job_id: str):
    """Delegate to service method"""
    return transcription_service.merge_chunk_results(job_id)

# Removed duplicate cleanup_job_files function - using the one at line 505 instead

def process_audio(job_id: str, audio_path: str, model_name: str, is_chunk: bool = False, chunk_info: Dict = None, language: Optional[str] = None, timeout: int = 600):
    """Process audio files with parallel processing support."""
    try:
        start_time = time.time()
        result_queue = multiprocessing.Queue()
        
        # Set CUDA device based on chunk number if using multiple GPUs
        if is_chunk and chunk_info:
            chunk_num = chunk_info["chunk_number"]
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                gpu_id = (chunk_num - 1) % num_gpus
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                logger.info(f"Processing chunk {chunk_num} on GPU {gpu_id}")
        
        logger.info(f"Processing {'chunk' if is_chunk else 'file'} at {audio_path}")
        
        # Start processing
        process = multiprocessing.Process(
            target=run_diarization_process, 
            args=(job_id, audio_path, model_name, language, result_queue)
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
                        # Save results to Azure Files only for chunks, not for single files
                        storage_client = transcription_service.storage_client
                        if is_chunk:
                            result_blob_name = f"job-{job_id}/chunks/chunk_{chunk_info['chunk_number']}_result.json"
                            
                            # Add chunk metadata if needed
                            result_data["data"].update({
                                "chunk_number": chunk_info["chunk_number"],
                                "total_chunks": chunk_info["total_chunks"]
                            })
                            
                            # Save to Azure Files (only for chunks)
                            storage_client.save_json(
                                container_name=storage_client.BATCH_OUTPUT,
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
                                    merged_result = merge_chunk_results(job_id)
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
                        # Do not delete temporary files
                        # if os.path.exists(audio_path):
                        #     os.unlink(audio_path)
                        #     logger.info(f"Removed temporary file {audio_path}")
                        logger.info(f"Preserved temporary file {audio_path}")
                        
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
        
        # Clean up
        # if os.path.exists(audio_path):
        #     os.unlink(audio_path)
            
    except Exception as e:
        logger.error(f"Error in process_audio: {str(e)}", exc_info=True)
        if job_id in job_storage:
            job_storage[job_id]["metadata"].update({
                "status": "error",
                "error": str(e)
            })
        # Clean up
        # if os.path.exists(audio_path):
        #     os.unlink(audio_path)

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

def merge_chunk_results(job_id: str) -> Optional[Dict[str, Any]]:
    """Merge results from all chunks of a job."""
    try:
        storage_client = transcription_service.storage_client
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
        result_files = storage_client.list_blobs(
            container_name=storage_client.BATCH_OUTPUT,
            prefix=f"job-{job_id}/chunks/"
        )
        
        # Log what files we found
        logger.info(f"Found {len(result_files)} files for job {job_id}: {result_files}")
        
        # Load chunk results and check if they have duration
        chunk_result_files = [f for f in result_files if f.endswith('_result.json')]
        logger.info(f"Found {len(chunk_result_files)} result files for job {job_id}")
        
        for result_file in chunk_result_files:
            chunk_data = storage_client.load_json(
                container_name=storage_client.BATCH_OUTPUT,
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
        
        # If total_duration is still 0, calculate from segments
        if total_duration == 0 and all_segments:
            total_duration = all_segments[-1]["end"]
            logger.info(f"Calculated total duration from last segment: {total_duration}")
        
        # If we still have no duration, use sum of chunk durations from metadata
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
        
        # If no speakers were detected, try to count unique speakers from segments
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
        storage_client.save_json(
            container_name=storage_client.BATCH_OUTPUT,
            blob_name=final_result_path,
            data=merged_result
        )
        logger.info(f"Saved final result to {final_result_path}")
        
        # IMPORTANT: Update job storage directly with result to ensure it's available
        if job_id in job_storage:
            job_storage[job_id]["result"] = merged_result
            job_storage[job_id]["metadata"]["status"] = "completed"
            logger.info(f"Updated job storage with merged result for {job_id}")
        
        # We'll let cleanup_job_files handle the cleanup of chunk results
        return merged_result
        
    except Exception as e:
        logger.error(f"Error merging results for job {job_id}: {str(e)}", exc_info=True)
        return None

def cleanup_job_files(job_id: str):
    """Clean up only temporary local files, preserving all Azure Files data."""
    try:
        # Clean up temp files (local only)
        if job_id in job_storage:
            job_data = job_storage[job_id]
            if "metadata" in job_data:
                # Clean up single file
                if "temp_path" in job_data["metadata"]:
                    temp_path = job_data["metadata"]["temp_path"]
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        logger.info(f"Removed temporary file: {temp_path}")
                
                # Clean up chunk files
                if "temp_paths" in job_data["metadata"]:
                    for temp_path in job_data["metadata"]["temp_paths"]:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            logger.info(f"Removed temporary chunk file: {temp_path}")
        
        # PRESERVE ALL AZURE FILES DATA
        # No longer deleting files from Azure Files storage
        # All processing files and output files are preserved for data retention
        logger.info(f"Preserved all Azure Files data for job {job_id} - no files deleted from storage")
            
    except Exception as e:
        logger.error(f"Error cleaning up job files for job {job_id}: {str(e)}")

@router.post("/transcribe", response_model=dict)
async def transcribe_uploaded_audio(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    request_data: Optional[str] = Form(None),
):
    """Submit audio file for transcription and diarization via HTTP POST."""
    # Parse request data if provided
    model_name = "tiny"
    language = "auto"
    compute_type = "float32"  # Default to float32 to avoid precision errors
    batch_options = {}
    
    if request_data:
        try:
            parsed_data = json.loads(request_data)
            
            # Handle the new unified schema
            if "mode" in parsed_data and parsed_data.get("mode") == "batch":
                # Extract model config
                if "model" in parsed_data:
                    model_config = parsed_data["model"]
                    if "name" in model_config:
                        model_name = model_config["name"]
                    if "language" in model_config:
                        language = model_config["language"]
                    if "compute_type" in model_config:
                        compute_type = model_config["compute_type"]
                    if "device" in model_config:
                        device = model_config["device"]
                
                # Extract batch options
                if "batch_options" in parsed_data:
                    batch_options = parsed_data["batch_options"]
            else:
                # Legacy format
                if "model" in parsed_data:
                    model_config = parsed_data["model"]
                    if "name" in model_config:
                        model_name = model_config["name"]
                    if "language" in model_config:
                        language = model_config["language"]
                    if "compute_type" in model_config:
                        compute_type = model_config["compute_type"]
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in request_data")
    
    job_id = str(uuid.uuid4())
    logger.info(f"Processing new transcription job (direct upload): {job_id}")
    
    # Get file extension from uploaded file
    file_ext = os.path.splitext(audio_file.filename)[1]
    if not file_ext:
        file_ext = ".wav"  # Default to .wav if no extension
    
    try:
        # Read the file content
        content = await audio_file.read()
        file_size = len(content)
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB, threshold: {THRESHOLD_MB} MB")
        
        # Check if file should be split (>700MB)
        is_large_file = file_size > FILE_SIZE_THRESHOLD
        logger.info(f"Is large file requiring splitting? {is_large_file}")
        
        # Initialize job metadata
        job_metadata = {
            "request_id": job_id,
            "status": "processing",
            "created_at": datetime.now().isoformat(),
            "filename": audio_file.filename,
            "file_size_mb": file_size_mb
        }
        
        # Initialize storage client
        storage_client = transcription_service.storage_client
        
        if is_large_file:
            # Split large file for parallel processing
            chunks = split_audio_file(content, job_id, file_ext, CHUNK_SIZE_MB)
            
            # Update job metadata for multi-chunk processing
            temp_paths = [chunk["temp_path"] for chunk in chunks]
            blob_paths = [chunk["blob_path"] for chunk in chunks]
            
            job_metadata.update({
                "status": "processing_multi_chunk",
                "total_chunks": len(chunks),
                "completed_chunks": 0,
                "temp_paths": temp_paths,
                "chunk_paths": blob_paths
            })
            
            # Create job record
            job_storage[job_id] = {
                "metadata": job_metadata,
                "model_info": {
                    "modelid": f"whisperx-{model_name}",
                    "name": model_name,
                    "version": "latest"
                }
            }
            
            # Process chunks in parallel
            max_workers = min(len(chunks), MAX_PARALLEL_CHUNKS)
            logger.info(f"Starting parallel processing with {max_workers} workers for job {job_id}")
            
            async def process_chunks():
                # Get system information
                total_cores = multiprocessing.cpu_count()
                active_sessions = len(transcription_service.active_sessions)
                
                # Resource allocation formula:
                # - 1 core for system operations
                # - 0.75 cores per active WebSocket session (minimum 1)
                # - Remaining cores for batch processing
                
                # Calculate cores to reserve
                system_cores = 1  # Reserve 1 core for system processes
                websocket_cores = math.ceil(active_sessions * 0.75) if active_sessions > 0 else 0
                reserved_cores = system_cores + websocket_cores
                
                # Calculate available cores for batch processing
                available_cores = max(1, total_cores - reserved_cores)
                
                # Cap max workers based on multiple factors
                adjusted_max_workers = min(available_cores, MAX_PARALLEL_CHUNKS, len(chunks))
                
                # Log resource allocation decision
                if active_sessions > 0:
                    logger.info(f"Resource allocation: {total_cores} total cores, {reserved_cores} reserved " +
                               f"({system_cores} system, {websocket_cores} for {active_sessions} WebSocket sessions), " +
                               f"{available_cores} available, using {adjusted_max_workers} workers for batch processing")
                else:
                    logger.info(f"Using {adjusted_max_workers}/{total_cores} cores for batch processing (no active WebSocket sessions)")
                
                with ThreadPoolExecutor(max_workers=adjusted_max_workers) as executor:
                    futures = []
                    # Calculate timeout per chunk based on total chunks
                    chunk_timeout = CHUNK_TIMEOUT
                    
                    for chunk in chunks:
                        chunk_info = {
                            "chunk_number": chunk["chunk_number"],
                            "total_chunks": chunk["total_chunks"]
                        }
                        
                        future = executor.submit(
                            transcription_service.process_audio,
                            job_id,
                            chunk["temp_path"],
                            model_name,
                            True,  # is_chunk
                            chunk_info,
                            None if language == "auto" else language,
                            chunk_timeout,
                            compute_type  # Pass compute_type
                        )
                        futures.append(future)
                    
                    # Wait for all chunks to complete
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            logger.info(f"Completed a chunk for job {job_id}")
                        except Exception as e:
                            logger.error(f"Error in chunk processing for job {job_id}: {str(e)}")
                
                # Check if we need to manually merge
                if job_id in job_storage:
                    job_data = job_storage[job_id]
                    if "metadata" in job_data and job_data["metadata"].get("status") != "completed":
                        if job_data["metadata"].get("completed_chunks", 0) == job_data["metadata"].get("total_chunks", 0):
                            logger.info(f"All chunks completed for job {job_id}, manually triggering merge")
                            merged_result = merge_chunk_results(job_id)
                            if merged_result:
                                logger.info(f"Successfully merged results for job {job_id}")
                            else:
                                logger.error(f"Failed to merge results for job {job_id}")
                
                # Cleanup after all processing and merging is done
                logger.info(f"Starting cleanup for job {job_id}")
                background_tasks.add_task(cleanup_job_files, job_id)
            
            # Start async processing
            background_tasks.add_task(process_chunks)
            
        else:
            # Handle single file processing
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            temp_path = temp_file.name
            temp_file.close()
            
            # Write content to temp file
            with open(temp_path, "wb") as f:
                f.write(content)
            
            # Upload to Azure Files Storage (batch processing)
            blob_name = f"job-{job_id}/audio{file_ext}"
            with open(temp_path, 'rb') as file_data:
                upload_success = storage_client.upload_file(
                    container_name=storage_client.BATCH_PROCESSING,
                    blob_name=blob_name,
                    data=file_data
                )
            
            if not upload_success:
                raise Exception(f"Failed to upload file to Azure Files Storage")
            
            # Update job metadata
            job_metadata.update({
                "temp_path": temp_path,
                "azure_path": f"{storage_client.BATCH_PROCESSING}/{blob_name}"
            })
            
            # Create job record
            job_storage[job_id] = {
                "metadata": job_metadata,
                "model_info": {
                    "modelid": f"whisperx-{model_name}",
                    "name": model_name,
                    "version": "latest"
                }
            }
            
            # Process single file - simplified with the unified service
            async def process_and_update():
                # Start timing
                global_start_time = time.time()
                
                # Get system information
                total_cores = multiprocessing.cpu_count()
                active_sessions = len(transcription_service.active_sessions)
                
                # Calculate cores to reserve using same formula as batch processing
                system_cores = 1
                websocket_cores = math.ceil(active_sessions * 0.75) if active_sessions > 0 else 0
                reserved_cores = system_cores + websocket_cores
                
                if active_sessions > 0:
                    # Try to adjust process priority if possible
                    try:
                        import os
                        current_nice = os.nice(0)  # Get current niceness
                        os.nice(5)  # Increase niceness (reduce priority) for batch processing
                        logger.info(f"Set batch processing priority to nice level {os.nice(0)} (from {current_nice})")
                    except:
                        # This might fail on Windows or due to permissions
                        logger.info(f"Could not adjust process priority for batch processing")
                    
                    logger.info(f"Processing single file with reduced priority (active WebSocket sessions: {active_sessions})")
                
                try:
                    # First process the audio
                    result = transcription_service.process_audio(
                        job_id,
                        temp_path,
                        model_name,
                        False,  # not a chunk
                        None,   # no chunk info
                        None if language == "auto" else language,
                        CHUNK_TIMEOUT,
                        compute_type  # Pass compute_type from request
                    )
                    
                    # Update the total_duration from the last segment if it's 0
                    if "total_duration" not in result or result["total_duration"] == 0:
                        if "segments" in result and result["segments"]:
                            segments = result["segments"]
                            if segments and isinstance(segments[-1], dict) and "end" in segments[-1]:
                                result["total_duration"] = segments[-1]["end"]
                                logger.info(f"Updated total_duration from last segment: {result['total_duration']}")
                    
                    # Extract num_speakers_detected from result["segments"] if it's 0
                    if "num_speakers_detected" not in result or result["num_speakers_detected"] == 0:
                        if "segments" in result and result["segments"]:
                            # Get unique speakers from segments
                            speakers = set()
                            for segment in result["segments"]:
                                if "speaker" in segment:
                                    speakers.add(segment["speaker"])
                            
                            if speakers:
                                result["num_speakers_detected"] = len(speakers)
                                logger.info(f"Updated num_speakers_detected from segments: {result['num_speakers_detected']}")
                    
                    # Save the result to Azure Files for consistency with multi-chunk files
                    storage_client = transcription_service.storage_client
                    storage_client.save_json(
                        container_name=storage_client.BATCH_OUTPUT,
                        blob_name=f"job-{job_id}/final_result.json",
                        data=result
                    )
                    
                    # Update the result with consistent fields
                    if "metadata" not in result:
                        result["metadata"] = {}
                        
                    result["metadata"]["workers_used"] = 1
                    result["metadata"]["total_processing_time_seconds"] = time.time() - global_start_time
                    result["total_chunks"] = 1
                    
                    # Update job storage with the properly structured result
                    if job_id in job_storage:
                        # THIS is the critical fix - properly update both sections
                        job_storage[job_id]["result"] = result
                        job_storage[job_id]["metadata"]["status"] = "completed"
                        
                        # Log successful completion
                        logger.info(f"Successfully processed single file job {job_id}")
                except Exception as e:
                    logger.error(f"Error in single file processing for job {job_id}: {str(e)}")
                    if job_id in job_storage:
                        job_storage[job_id]["metadata"].update({
                            "status": "error",
                            "error": str(e)
                        })
                finally:
                    # Clean up when done
                    background_tasks.add_task(cleanup_job_files, job_id)
            
            # Start processing in the background
            background_tasks.add_task(process_and_update)
        
        return job_storage[job_id]
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}", exc_info=True)
        
        # Clean up any temporary files
        if 'temp_paths' in locals():
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
            
        raise HTTPException(status_code=500, detail=f"Error processing uploaded file: {str(e)}")

@router.get("/v1/jobs/{job_id}", response_model=dict)
async def get_job_status(job_id: str):
    """Get the status or results of a previously submitted transcription job."""
    if job_id not in job_storage:
        logger.warning(f"Job not found: {job_id}")
        raise HTTPException(status_code=404, detail="Job not found")
    
    try:
        job_data = job_storage[job_id]
        
        # If we have a result, ensure it has the expected structure
        if "result" in job_data and isinstance(job_data["result"], dict):
            result = job_data["result"]
            
            # For consistency, always include these fields
            if "total_chunks" not in result:
                result["total_chunks"] = 1
                
            if "metadata" not in result:
                result["metadata"] = {}
                
            if "workers_used" not in result["metadata"]:
                result["metadata"]["workers_used"] = 1
                
            # Most important fix: Calculate total_duration from the last segment if it's missing or zero
            if "total_duration" not in result or result["total_duration"] == 0:
                if "segments" in result and result["segments"]:
                    segments = result["segments"]
                    if segments and isinstance(segments[-1], dict) and "end" in segments[-1]:
                        result["total_duration"] = segments[-1]["end"]
                        logger.info(f"Updated missing total_duration from last segment in get_job_status: {result['total_duration']}")
            
            # Fix for num_speakers_detected
            if "num_speakers_detected" not in result or result["num_speakers_detected"] == 0:
                if "segments" in result and result["segments"]:
                    # Get unique speakers from segments
                    speakers = set()
                    for segment in result["segments"]:
                        if "speaker" in segment:
                            speakers.add(segment["speaker"])
                    
                    if speakers:
                        result["num_speakers_detected"] = len(speakers)
                        logger.info(f"Updated num_speakers_detected from segments in get_job_status: {result['num_speakers_detected']}")
        
        # Check if data is JSON serializable
        try:
            # This will fail if there's binary data
            json.dumps(job_data)
            return JSONResponse(content=job_data)
        except (TypeError, OverflowError):
            # Convert any non-serializable data to a safe format
            return JSONResponse(content={
                "metadata": {
                    "status": "error",
                    "request_id": job_id,
                    "message": "Job results contain non-serializable data"
                },
                "error": "The job results cannot be displayed directly. Please use the /text endpoint."
            })
    
    except Exception as e:
        logger.error(f"Error retrieving job {job_id}: {str(e)}", exc_info=True)
        return JSONResponse(content={
            "metadata": {
                "status": "error",
                "request_id": job_id,
                "message": f"Error retrieving job: {str(e)}"
            }
        })


@router.get("/v1/jobs/{job_id}/text")
async def get_job_text(job_id: str):
    """Get only the transcribed text from a job."""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    try:
        job_data = job_storage[job_id]
        status = job_data.get("metadata", {}).get("status", "unknown")
        
        # Check Azure Files for results even if status isn't completed
        if status == "error" and "timeout" in job_data.get("metadata", {}).get("error", ""):
            storage_client = transcription_service.storage_client
            # Try to find the result file
            result_files = storage_client.list_blobs(
                container_name=storage_client.BATCH_OUTPUT,
                prefix=f"job-{job_id}"
            )
            for result_file in result_files:
                if result_file.endswith(".json"):
                    results = storage_client.load_json(
                        container_name=storage_client.BATCH_OUTPUT,
                        blob_name=result_file
                    )
                    if results and "transcript" in results:
                        return JSONResponse(content={
                            "status": "completed",
                            "text": results["transcript"]
                        })
        
        # Regular status checking
        if status != "completed":
            return JSONResponse(content={
                "status": status,
                "message": "Transcription not yet completed"
            })
        
        if "transcript" in job_data:
            return JSONResponse(content={
                "status": "completed",
                "text": job_data["transcript"]
            })
        
        # Check for transcript in the new format
        if "results" in job_data and "transcript" in job_data["results"]:
            return JSONResponse(content={
                "status": "completed",
                "text": job_data["results"]["transcript"]
            })
        
        # Fallback to segments
        elif "results" in job_data and "segments" in job_data["results"]:
            try:
                segments_text = []
                for segment in job_data["results"]["segments"]:
                    if isinstance(segment, dict) and "text" in segment:
                        segments_text.append(segment.get("text", ""))
                
                full_text = " ".join(segments_text)
                return JSONResponse(content={
                    "status": "completed",
                    "text": full_text
                })
            except Exception as e:
                logger.error(f"Error combining segment texts: {str(e)}", exc_info=True)
                return JSONResponse(content={
                    "status": "error", 
                    "message": "Failed to process segments"
                })
        
        # Legacy format - directly in job_data
        elif "segments" in job_data:
            try:
                segments_text = []
                for segment in job_data["segments"]:
                    if isinstance(segment, dict) and "text" in segment:
                        segments_text.append(segment.get("text", ""))
                
                full_text = " ".join(segments_text)
                return JSONResponse(content={
                    "status": "completed",
                    "text": full_text
                })
            except Exception as e:
                logger.error(f"Error combining segment texts: {str(e)}", exc_info=True)
                return JSONResponse(content={
                    "status": "error", 
                    "message": "Failed to process segments"
                })
        
        else:
            logger.error(f"Job {job_id} completed but no text or segments found in data structure")
            return JSONResponse(content={
                "status": "error", 
                "message": "No text available in completed job"
            })
            
    except Exception as e:
        logger.error(f"Error retrieving text for job {job_id}: {str(e)}", exc_info=True)
        return JSONResponse(content={
            "status": "error", 
            "message": f"Error retrieving text: {str(e)}"
        })

@router.get("/v1/jobs/{job_id}/debug")
async def debug_job_data(job_id: str):
    """Debug endpoint to inspect job data."""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = job_storage[job_id]
    data_type = type(job_data).__name__
    
    # Check if metadata exists and is a dict
    metadata_exists = "metadata" in job_data and isinstance(job_data["metadata"], dict)
    
    # Check if results exists
    results_exists = "results" in job_data
    results_type = type(job_data.get("results", {})).__name__
    
    # Check transcript
    transcript_exists = results_exists and "transcript" in job_data["results"]
    transcript_type = type(job_data.get("results", {}).get("transcript", "")).__name__ if results_exists else "N/A"
    
    # Check segments
    segments_exist = results_exists and "segments" in job_data["results"]
    segments_type = type(job_data.get("results", {}).get("segments", [])).__name__ if results_exists else "N/A"
    num_segments = len(job_data.get("results", {}).get("segments", [])) if segments_exist else 0
    
    return JSONResponse(content={
        "data_type": data_type,
        "metadata_exists": metadata_exists,
        "results_exists": results_exists,
        "results_type": results_type,
        "transcript_exists": transcript_exists,
        "transcript_type": transcript_type,
        "segments_exist": segments_exist,
        "segments_type": segments_type,
        "num_segments": num_segments,
        "status": job_data.get("metadata", {}).get("status", "unknown")
    })

@router.get("/v1/system-info")
async def get_system_info():
    """Get system resource information."""
    # Get basic system info
    cpu_count = multiprocessing.cpu_count()
    active_sessions = len(transcription_service.active_sessions)
    
    # Calculate recommended allocation using the same formula
    system_cores = 1
    websocket_cores = math.ceil(active_sessions * 0.75) if active_sessions > 0 else 0
    reserved_cores = system_cores + websocket_cores
    available_cores = max(1, cpu_count - reserved_cores)
    
    # Get running jobs
    active_jobs = [job_id for job_id in job_storage if 
                   job_storage[job_id].get("metadata", {}).get("status") in 
                   ["processing", "processing_multi_chunk"]]
    
    # Try to get process stats if psutil is available
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)
        total_memory = psutil.virtual_memory().total
        memory_info = {
            "process_memory_mb": mem_info.rss / (1024 * 1024),
            "memory_percent": mem_info.rss / total_memory * 100,
            "total_memory_gb": total_memory / (1024 * 1024 * 1024)
        }
    except ImportError:
        memory_info = {"error": "psutil not installed"}
        cpu_percent = None
    
    return JSONResponse(content={
        "cpu": {
            "total_cores": cpu_count,
            "cpu_percent": cpu_percent
        },
        "memory": memory_info,
        "allocation": {
            "active_websocket_sessions": active_sessions,
            "reserved_cores": reserved_cores,
            "available_cores": available_cores,
            "recommended_batch_workers": min(available_cores, MAX_PARALLEL_CHUNKS)
        },
        "service": {
            "active_websocket_sessions": active_sessions,
            "active_jobs": len(active_jobs),
            "total_jobs_in_storage": len(job_storage),
            "active_job_ids": active_jobs
        }
    })