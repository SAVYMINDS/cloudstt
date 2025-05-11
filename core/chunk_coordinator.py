import os
import logging
import uuid
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import multiprocessing
from multiprocessing import Process, Queue
import tempfile
import torch

from core.diarization_processor import DiarizationService
from storage.azure_storage import AzureBlobStorage
from core.audio_utils import split_audio_file, is_large_file

logger = logging.getLogger(__name__)

# Module-level function to run in a separate process
def process_chunk(chunk_id, chunk_path, model_name, language, result_queue):
    try:
        # Get Hugging Face token from environment variable
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        
        # Initialize service
        diarization_service = DiarizationService(
            default_model=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16",
            batch_size=16,
        )
        
        logger.info(f"Processing chunk {chunk_id}: {chunk_path}")
        
        # Process the audio chunk
        result = diarization_service.process_audio_file(
            file_path=chunk_path,
            model=model_name,
            language=language,
            hf_token=hf_token,
        )
        
        # Put result in queue
        result_queue.put({"chunk_id": chunk_id, "success": True, "data": result})
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
        # Put error in queue
        result_queue.put({"chunk_id": chunk_id, "success": False, "error": str(e)})

class ChunkProcessor:
    """
    Manages processing of large audio files by splitting them into chunks
    and coordinating parallel processing.
    """
    
    def __init__(self, job_storage):
        """
        Initialize the chunk processor.
        
        Args:
            job_storage: Reference to the shared job storage dictionary
        """
        self.job_storage = job_storage
        
    def process_large_file(
        self,
        job_id: str,
        audio_path: str,
        model_name: str,
        language: Optional[str] = None,
        chunk_size_mb: int = 200,
        max_parallel_chunks: int = 2,  # Process 2 chunks at a time to balance resources
        timeout_per_chunk: int = 300  # 5 minutes timeout per chunk
    ):
        """
        Process a large audio file by splitting it into chunks and processing them in parallel.
        
        Args:
            job_id: ID of the job
            audio_path: Path to the audio file
            model_name: Name of the model to use
            language: Language code
            chunk_size_mb: Size of each chunk in MB
            max_parallel_chunks: Maximum number of chunks to process in parallel
            timeout_per_chunk: Processing timeout in seconds per chunk
        """
        try:
            # Update job status
            chunks_dir = tempfile.mkdtemp(prefix=f"job_{job_id}_chunks_")
            logger.info(f"Created temporary directory for chunks: {chunks_dir}")
            if job_id in self.job_storage:
                self.job_storage[job_id]["metadata"]["status"] = "splitting_audio"
                
            # Split the audio file
            chunk_paths = split_audio_file(
                file_path=audio_path,
                output_dir=chunks_dir,
                chunk_size_mb=chunk_size_mb,
                file_prefix=f"job_{job_id}"
            )
            
            if not chunk_paths:
                logger.error(f"Failed to split audio file for job {job_id}")
                self.job_storage[job_id]["metadata"]["status"] = "error"
                self.job_storage[job_id]["metadata"]["message"] = "Failed to split audio file"
                return
            
            num_chunks = len(chunk_paths)
            logger.info(f"Split audio file into {num_chunks} chunks  for job {job_id}")

             # Store chunks in Azure
            storage_client = AzureBlobStorage()
            chunk_info = storage_client.store_chunks(job_id, chunk_paths)
            # Update job status with chunk information
            if job_id in self.job_storage:
                self.job_storage[job_id]["metadata"]["status"] = "processing_chunks"
                self.job_storage[job_id]["metadata"]["total_chunks"] = num_chunks
                self.job_storage[job_id]["metadata"]["processed_chunks"] = 0
            
            # Process chunks with limited parallelism
            all_results = []
            
            # Create a queue for results
            result_queue = multiprocessing.Queue()
            
            # Process chunks in batches
            for i in range(0, len(chunk_paths), max_parallel_chunks):
                batch_chunk_paths = chunk_paths[i:i + max_parallel_chunks]
                processes = []
                
                # Start processes for this batch
                for j, chunk_path in enumerate(batch_chunk_paths):
                    chunk_idx = i + j
                    chunk_id = f"{job_id}_chunk_{chunk_idx}"
                    
                    logger.info(f"Starting process for chunk {chunk_idx+1}/{len(chunk_paths)}")
                    
                    # Create and start the process
                    process = multiprocessing.Process(
                        target=process_chunk,
                        args=(chunk_id, chunk_path, model_name, language, result_queue)
                    )
                    process.daemon = True
                    process.start()
                    processes.append((process, chunk_id, chunk_path, chunk_idx))
                
                # Wait for this batch to complete with timeout
                remaining_processes = list(processes)
                batch_start_time = time.time()
                
                while remaining_processes and time.time() - batch_start_time < timeout_per_chunk:
                    # Check if any results are available
                    while not result_queue.empty():
                        result_data = result_queue.get()
                        chunk_id = result_data["chunk_id"]
                        
                        # Remove this chunk from remaining processes
                        remaining_processes = [p for p in remaining_processes if p[1] != chunk_id]
                        
                        if result_data.get("success", False):
                            logger.info(f"Chunk {chunk_id} processed successfully")
                            all_results.append({
                                "chunk_idx": next(p[3] for p in processes if p[1] == chunk_id),
                                "data": result_data["data"]
                            })
                        else:
                            error_msg = result_data.get("error", "Unknown error")
                            logger.error(f"Error processing chunk {chunk_id}: {error_msg}")
                    
                    # Sleep a short time before checking again
                    time.sleep(0.1)
                
                # Terminate any remaining processes that didn't complete in time
                for process, chunk_id, chunk_path, _ in remaining_processes:
                    logger.warning(f"Process for chunk {chunk_id} timed out, terminating")
                    process.terminate()
                    process.join(1)
                    
                    # If still alive, force kill
                    if process.is_alive():
                        logger.warning(f"Process for chunk {chunk_id} still alive after terminate, killing")
                        try:
                            os.kill(process.pid, 9)  # SIGKILL
                        except:
                            pass
                
                # Update processed chunk count
                if job_id in self.job_storage:
                    self.job_storage[job_id]["metadata"]["processed_chunks"] = i + len(batch_chunk_paths) - len(remaining_processes)
            
            # All chunks processed, update status
            if job_id in self.job_storage:
                # Update job status
                if not all_results:
                    error_msg = "No chunks were successfully processed"
                    logger.error(error_msg)
                    self.job_storage[job_id]["metadata"]["status"] = "error"
                    self.job_storage[job_id]["metadata"]["message"] = error_msg
                    return
                
                # Sort results by chunk index
                all_results.sort(key=lambda x: x["chunk_idx"])
                
                # Merge the results
                logger.info(f"Merging results from {len(all_results)} chunks")
                self.job_storage[job_id]["metadata"]["status"] = "merging_results"
                
                merged_result = self._merge_chunk_results(job_id, all_results)
                
                if merged_result:
                    # Update job storage with merged results
                    self.job_storage[job_id] = merged_result
                    self.job_storage[job_id]["metadata"]["status"] = "completed"
                    logger.info(f"Large file processing completed for job {job_id}")
                else:
                    self.job_storage[job_id]["metadata"]["status"] = "error"
                    self.job_storage[job_id]["metadata"]["message"] = "Failed to merge chunk results"
            
            # Clean up temporary files
            for chunk_path in chunk_paths:
                try:
                    if os.path.exists(chunk_path):
                        os.unlink(chunk_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up chunk file {chunk_path}: {str(e)}")
            
            try:
                os.rmdir(temp_dir)
                logger.info(f"Removed temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary directory {temp_dir}: {str(e)}")
            
        except Exception as e:
            error_msg = f"Error processing large file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            if job_id in self.job_storage:
                self.job_storage[job_id]["metadata"]["status"] = "error"
                self.job_storage[job_id]["metadata"]["message"] = error_msg
    
    def _merge_chunk_results(self, job_id: str, chunk_results: List[Dict]) -> Dict:
        """
        Merge the results from multiple chunks into a single result.
        
        Args:
            job_id: ID of the job
            chunk_results: List of chunk processing results
            
        Returns:
            Merged result dictionary
        """
        try:
            # Start with a copy of the base job data
            if job_id not in self.job_storage:
                return None
            
            base_job = self.job_storage[job_id]
            
            # Create merged result structure
            merged_result = {
                "metadata": {
                    "request_id": job_id,
                    "status": "completed",
                    "created_at": base_job["metadata"]["created_at"],
                    "completed_at": datetime.now().isoformat(),
                    "is_large_file": True,
                    "total_chunks": len(chunk_results),
                    "processed_chunks": len(chunk_results)
                },
                "model_info": base_job["model_info"],
                "segments": [],
                "transcript": "",
                "num_speakers_detected": 0
            }
            
            # Merge chunk results
            all_segments = []
            transcript_parts = []
            num_speakers = 0
            cumulative_duration = 0
            
            for chunk_result_data in chunk_results:
                chunk_data = chunk_result_data["data"]
                
                # Skip chunks with errors
                if not chunk_data or chunk_data.get("metadata", {}).get("status") != "success":
                    continue
                
                # Get segments from this chunk
                chunk_segments = []
                if "segments" in chunk_data:
                    chunk_segments = chunk_data["segments"]
                
                # Adjust timestamps by adding the cumulative duration
                adjusted_segments = []
                for segment in chunk_segments:
                    adjusted_segment = segment.copy()
                    adjusted_segment["start"] += cumulative_duration
                    adjusted_segment["end"] += cumulative_duration
                    
                    # Adjust word timestamps if present
                    if "words" in segment:
                        adjusted_words = []
                        for word in segment["words"]:
                            adjusted_word = word.copy()
                            adjusted_word["start"] += cumulative_duration
                            adjusted_word["end"] += cumulative_duration
                            adjusted_words.append(adjusted_word)
                        adjusted_segment["words"] = adjusted_words
                    
                    adjusted_segments.append(adjusted_segment)
                
                all_segments.extend(adjusted_segments)
                
                # Get transcript from this chunk
                if "transcript" in chunk_data:
                    transcript_parts.append(chunk_data["transcript"])
                
                # Get max speaker number
                if "num_speakers_detected" in chunk_data:
                    num_speakers = max(num_speakers, chunk_data["num_speakers_detected"])
                
                # Update cumulative duration
                chunk_duration = 0
                if "metadata" in chunk_data and "total_duration" in chunk_data["metadata"]:
                    chunk_duration = chunk_data["metadata"]["total_duration"]
                elif chunk_segments and chunk_segments[-1]["end"]:
                    chunk_duration = chunk_segments[-1]["end"]
                
                cumulative_duration += chunk_duration
            
            # Store merged segments and transcript
            merged_result["segments"] = all_segments
            merged_result["transcript"] = "\n\n".join(transcript_parts)
            merged_result["num_speakers_detected"] = num_speakers
            
            # Add total duration to metadata
            merged_result["metadata"]["total_duration"] = cumulative_duration
            
            return merged_result
            
        except Exception as e:
            logger.error(f"Error merging chunk results for job {job_id}: {str(e)}", exc_info=True)
            return None