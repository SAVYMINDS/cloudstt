import os
import logging
from typing import List, Tuple, Optional
from pydub import AudioSegment
import math

logger = logging.getLogger(__name__)

def get_audio_file_size(file_path: str) -> int:
    """Get the size of an audio file in bytes."""
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        logger.error(f"Error getting file size for {file_path}: {str(e)}")
        return 0

def is_large_file(file_path: str, threshold_mb: float = 200.0) -> bool:
    """
    Check if an audio file exceeds the size threshold.
    
    Args:
        file_path: Path to the audio file
        threshold_mb: Size threshold in megabytes (default: 200MB)
        
    Returns:
        bool: True if file size exceeds threshold
    """
    threshold_bytes = threshold_mb * 1024 * 1024  # Convert MB to bytes
    file_size = get_audio_file_size(file_path)
    return file_size > threshold_byte
def split_audio_file(
    file_path: str, 
    output_dir: str, 
    chunk_size_mb: int = 1000,
    file_prefix: str = "chunk"
) -> List[str]:
    """
    Split a large audio file into smaller chunks.
    
    Args:
        file_path: Path to the audio file
        output_dir: Directory to save chunks
        chunk_size_mb: Size of each chunk in MB
        file_prefix: Prefix for chunk files
        
    Returns:
        List of paths to the created audio chunks
    """
    try:
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the audio file
        logger.info(f"Loading audio file for splitting: {file_path}")
        audio = AudioSegment.from_file(file_path)
        
        # Get file extension
        file_ext = os.path.splitext(file_path)[1]
        if not file_ext:
            file_ext = ".wav"  # Default to .wav if no extension
        
        # Calculate chunk duration based on size
        # Rough estimate: 1MB ~= 5.5 seconds of CD-quality audio (44.1kHz, 16-bit, stereo)
        # This may vary based on codec, but it's a starting point
        duration_ms = len(audio)
        estimated_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        ms_per_mb = duration_ms / estimated_size_mb
        
        chunk_duration_ms = chunk_size_mb * ms_per_mb
        
        # Get number of chunks
        num_chunks = math.ceil(duration_ms / chunk_duration_ms)
        logger.info(f"Splitting {file_path} into {num_chunks} chunks")
        
        chunk_paths = []
        
        for i in range(num_chunks):
            # Calculate start and end times for this chunk
            start_ms = i * chunk_duration_ms
            end_ms = min((i + 1) * chunk_duration_ms, duration_ms)
            
            # Extract the chunk
            chunk = audio[start_ms:end_ms]
            
            # Generate chunk filename
            chunk_filename = f"{file_prefix}_{i:03d}{file_ext}"
            chunk_path = os.path.join(output_dir, chunk_filename)
            
            # Export chunk to file (using original codec to maintain quality)
            logger.info(f"Exporting chunk {i+1}/{num_chunks} to {chunk_path}")
            
            # Get format from extension (wav, mp3, etc.)
            format = file_ext.replace('.', '')
            
            # Export with high quality settings for wav
            if format.lower() == 'wav':
                chunk.export(
                    chunk_path, 
                    format=format,
                    parameters=["-acodec", "pcm_s16le"]  # Use standard 16-bit PCM codec
                )
            else:
                chunk.export(chunk_path, format=format)
                
            chunk_paths.append(chunk_path)
            
        logger.info(f"Successfully split {file_path} into {len(chunk_paths)} chunks")
        return chunk_paths
    
    except Exception as e:
        logger.error(f"Error splitting audio file {file_path}: {str(e)}", exc_info=True)
        return []