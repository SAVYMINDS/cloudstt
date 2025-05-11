"""
WhisperX-based audio transcription and diarization script.
Run this file directly to process an audio file.
"""

import os
import json
import logging
import uuid
import argparse
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

import whisperx
import torch
import gc

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import sys
sys.path.append('..')  # Add parent directory to path
from storage.azure_storage import AzureBlobStorage

class DiarizationService:
    """Service for processing audio files with speaker diarization."""
    
    def __init__(self, 
                 default_model: str = "small", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 compute_type: str = "float32",
                 batch_size: int = 16):
        """
        Initialize the diarization service.
        
        Args:
            default_model: Default WhisperX model size
            device: Device to use for computation (cuda/cpu)
            compute_type: Computation precision
            batch_size: Batch size for transcription
        """
        self.default_model = default_model
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.storage_client = AzureBlobStorage()
        
        # Ensure storage containers exist
        self.storage_client.setup_containers()
        
        logger.info(f"Initialized DiarizationService with {device} device")
    
    def process_audio_file(self, 
                          file_path: str, 
                          model: Optional[str] = None,
                          language: Optional[str] = None,
                          hf_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an audio file with WhisperX for transcription and diarization.
        
        Args:
            file_path: Path to the audio file
            model: WhisperX model size
            language: Language code
            hf_token: Hugging Face token for diarization
            
        Returns:
            Dict containing processed results with speaker segments
        """
        request_id = str(uuid.uuid4())
        start_time = datetime.now()
        logger.info(f"Processing request {request_id}: {file_path}")
        
        if not os.path.exists(file_path):
            error_msg = f"Audio file not found: {file_path}"
            logger.error(error_msg)
            return {
                "metadata": {
                    "status": "error",
                    "request_id": request_id,
                    "created_at": start_time.isoformat(),
                    "message": error_msg
                }
            }
        
        try:
            # 1. Load model and transcribe audio
            model_name = model or self.default_model
            logger.info(f"Loading WhisperX model: {model_name}")
            
            whisper_model = whisperx.load_model(
                model_name, 
                self.device, 
                compute_type=self.compute_type
            )
            
            # Get model version information
            model_version = getattr(whisper_model, "version", "unknown")
            model_id = f"whisperx-{model_name}-{model_version}"
            
            logger.info(f"Loading audio: {file_path}")
            audio = whisperx.load_audio(file_path)
            
            logger.info("Transcribing audio")
            result = whisper_model.transcribe(
                audio, 
                batch_size=self.batch_size,
                language=language
            )
            
            detected_language = result["language"]
            logger.info(f"Detected language: {detected_language}")
            
            # Clean up whisper model to save memory
            del whisper_model
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # 2. Align whisper output
            logger.info("Loading alignment model")
            align_model, metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=self.device
            )
            
            logger.info("Aligning transcription")
            result = whisperx.align(
                result["segments"], 
                align_model, 
                metadata, 
                audio, 
                self.device, 
                return_char_alignments=False  # Changed to True as requested
            )
            
            # Clean up alignment model
            del align_model
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # 3. Perform diarization
            logger.info("Loading diarization model")
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token,
                device=self.device
            )
            
            logger.info("Performing speaker diarization")
            diarize_segments = diarize_model(audio)
            
            # 4. Assign speakers to words/segments
            logger.info("Assigning speakers to segments")
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # Clean up diarization model
            del diarize_model
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            # 5. Format results to match desired schema
            formatted_result = self._format_results(
                result, 
                request_id, 
                start_time,
                model_name,
                model_id,
                model_version
            )
            
            # 6. Save results to Azure Blob Storage
            output_filename = f"{Path(file_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Save locally
            local_path = f"output_{output_filename}"
            with open(local_path, 'w') as f:
                json.dump(formatted_result, f, indent=2)
            logger.info(f"Results saved locally to: {local_path}")
            
            # Save to Azure Blob Storage
            self.storage_client.save_json(
                container_name=self.storage_client.OUTPUT_CONTAINER,
                blob_name=output_filename,
                data=formatted_result
            )
            logger.info(f"Results saved to Azure: {output_filename}")
            
            return formatted_result
            
        except Exception as e:
            error_msg = f"Error processing audio file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "metadata": {
                    "status": "error",
                    "request_id": request_id,
                    "created_at": start_time.isoformat(),
                    "message": error_msg
                }
            }
    
    def _format_results(self, 
                   result: Dict[str, Any], 
                   request_id: str,
                   start_time: datetime,
                   model_name: str,
                   model_id: str,
                   model_version: str) -> Dict[str, Any]:

        segments = result.get("segments", [])
        total_duration = segments[-1]["end"] if segments else 0
    
    # Map speaker IDs (SPEAKER_00, SPEAKER_01) to Speaker 1, Speaker 2, etc.
        speaker_map = {}
        for segment in segments:
            speaker = segment.get("speaker")
            if speaker and speaker not in speaker_map:
                speaker_number = len(speaker_map) + 1
                speaker_map[speaker] = f"Speaker {speaker_number}"
    
    # Format segments with word-level information
        formatted_segments = []
    
    # Create the full transcript with speaker labels
        transcript_lines = []
        current_speaker = None
        current_text = ""
    
        for segment in segments:
        # Process words
            words = []
            word_count = 0
        
            if "words" in segment:
                for word_info in segment["words"]:
                # Build word object with available fields
                    word_obj = {
                    "word": word_info.get("word", ""),
                    "start": word_info.get("start", 0),
                    "end": word_info.get("end", 0),
                }
                
                # Add confidence only if it exists
                    if "confidence" in word_info:
                        word_obj["confidence"] = word_info["confidence"]
                
                    words.append(word_obj)
                    word_count += 1
        
        # Create the formatted segment
            formatted_segment = {
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
                "text": segment.get("text", ""),
                "word_count": word_count,
                "words": words
            }
        
        # Add confidence only if it exists in the segment
            if "confidence" in segment:
                formatted_segment["confidence"] = segment["confidence"]
        
        # Add speaker if available
            speaker = segment.get("speaker")
            speaker_label = None
            if speaker and speaker in speaker_map:
                speaker_label = speaker_map[speaker]
                formatted_segment["speaker"] = speaker_label
        
                formatted_segments.append(formatted_segment)
        
        # Build transcript with speaker changes
            if speaker_label:
                segment_text = segment.get("text", "").strip()
                if segment_text:
                    if speaker_label != current_speaker:
                    # If we have accumulated text for the previous speaker, add it to transcript
                        if current_speaker and current_text.strip():
                            transcript_lines.append(f"{current_speaker}: {current_text.strip()}")
                    
                    # Start new speaker's text
                        current_speaker = speaker_label
                        current_text = segment_text
                    else:
                    # Same speaker continues, append text with space
                        current_text += " " + segment_text
    
    # Add the final speaker's text if any
        if current_speaker and current_text.strip():
            transcript_lines.append(f"{current_speaker}: {current_text.strip()}")
    
    # Join transcript lines
        full_transcript = "\n".join(transcript_lines)
    
    # Create the complete response structure
        return {
            "metadata": {
            "status": "success",
            "request_id": request_id,
            "total_duration": total_duration,
            "created_at": start_time.isoformat(),
        },
        "model_info": {
            "modelid": model_id,
            "name": model_name,
            "version": model_version
        },
        "segments": formatted_segments,
        "num_speakers_detected": len(speaker_map),
        "transcript": full_transcript
    }


def main():
    """Main function to process an audio file."""
    parser = argparse.ArgumentParser(description="Process audio file with WhisperX")
    parser.add_argument("--file", type=str, default="/home/azureuser/cloudfiles/code/Users/maliksalim.savyminds/SST/storage/order.wav",
                        help="Path to the audio file")
    parser.add_argument("--model", type=str, default="medium",
                        help="WhisperX model size")
    
    args = parser.parse_args()
    
    # Get Hugging Face token from environment variable
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        logger.warning("HUGGINGFACE_TOKEN not found in environment. Diarization may fail.")
    
    # Initialize service with settings directly in code
    diarization_service = DiarizationService(
        default_model="medium",
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float32",
        batch_size=16
    )
    
    # Process the audio file
    result = diarization_service.process_audio_file(
        file_path=args.file,
        model=args.model,
        hf_token=hf_token
    )
    
    # Print summary of results
    print("\n" + "="*50)
    print(f"Processed audio file: {args.file}")
    print(f"Request ID: {result['metadata']['request_id']}")
    print(f"Status: {result['metadata']['status']}")
    if result['metadata']['status'] == 'success':
        print(f"Number of speakers detected: {result['num_speakers_detected']}")
        print(f"Number of segments: {len(result['segments'])}")
        print(f"Total duration: {result['metadata']['total_duration']} seconds")
    else:
        print(f"Error: {result['metadata'].get('message', 'Unknown error')}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()

