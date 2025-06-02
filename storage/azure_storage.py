"""
Azure Files Storage adapter that uses mounted Azure Files share via SMB.
This replaces the local storage simulation with actual Azure Files storage.
"""

import os
import json
import logging
import shutil
from typing import Optional, List, BinaryIO, Dict, Any
from datetime import datetime
from pathlib import Path
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AzureFilesStorage:
    """
    Azure Files storage adapter that uses mounted Azure Files share via SMB.
    Files are stored on the mounted Azure Files share instead of local storage.
    """
    
    # Top-level processing type directories
    BATCH_DIR = "batch"
    REALTIME_DIR = "realtime"
    SHARED_DIR = "shared"
    
    # Batch processing subdirectories
    BATCH_INPUT = f"{BATCH_DIR}/input"
    BATCH_PROCESSING = f"{BATCH_DIR}/processing"
    BATCH_OUTPUT = f"{BATCH_DIR}/output"
    BATCH_METADATA = f"{BATCH_DIR}/metadata"
    
    # Real-time processing subdirectories
    REALTIME_SESSIONS = f"{REALTIME_DIR}/sessions"
    REALTIME_CHUNKS = f"{REALTIME_DIR}/chunks"
    REALTIME_OUTPUT = f"{REALTIME_DIR}/output"
    REALTIME_METADATA = f"{REALTIME_DIR}/metadata"
    
    # Shared directories
    SHARED_MODELS = f"{SHARED_DIR}/models"
    SHARED_TEMP = f"{SHARED_DIR}/temp"
    
    # Legacy container names (for backward compatibility)
    INPUT_CONTAINER = BATCH_INPUT
    PROCESSING_CONTAINER = BATCH_PROCESSING
    OUTPUT_CONTAINER = BATCH_OUTPUT
    METADATA_CONTAINER = BATCH_METADATA
    
    # List of all directories to create
    DIRECTORIES = [
        BATCH_INPUT, BATCH_PROCESSING, BATCH_OUTPUT, BATCH_METADATA,
        REALTIME_SESSIONS, REALTIME_CHUNKS, REALTIME_OUTPUT, REALTIME_METADATA,
        SHARED_MODELS, SHARED_TEMP
    ]
    
    # Legacy containers list (for backward compatibility)
    CONTAINERS = [BATCH_INPUT, BATCH_PROCESSING, BATCH_OUTPUT, BATCH_METADATA]
    
    def __init__(self, mount_path: Optional[str] = None):
        """
        Initialize the Azure Files storage adapter.
        
        Args:
            mount_path: Path where Azure Files share is mounted. 
                       Defaults to '/app/azurestorage' (container mount path)
                       Falls back to 'local_storage' for local development
        """
        # Determine the base directory
        if mount_path:
            self.base_dir = mount_path
            self.is_azure_files = mount_path != 'local_storage'
        else:
            # Check for Azure Files mount at the primary location
            if os.path.exists('/app/azurestorage'):
                # Running in container with mounted Azure Files
                self.base_dir = '/app/azurestorage'
                self.is_azure_files = True
                logger.info(f"Found Azure Files mount at {self.base_dir}")
            elif os.path.exists('/app/storage'):
                # Fallback to old mount location
                self.base_dir = '/app/storage'
                self.is_azure_files = True
                logger.info(f"Found Azure Files mount at {self.base_dir} (legacy path)")
            else:
                # Local development fallback or no Azure Files mount
                self.base_dir = os.path.join(os.getcwd(), 'local_storage')
                self.is_azure_files = False
                logger.warning("Azure Files mount not found, falling back to local storage")
        
        self.setup_containers()
        
        if self.is_azure_files:
            logger.info(f"Using Azure Files storage mounted at {self.base_dir}")
        else:
            logger.info(f"Using local file storage in {self.base_dir} (development mode)")
    
    def _is_azure_files_mounted(self, path: str) -> bool:
        """Check if the given path is actually an Azure Files mount point."""
        try:
            # Check if it's a mount point by looking at /proc/mounts
            with open('/proc/mounts', 'r') as f:
                mounts = f.read()
                # Look for SMB/CIFS mounts at this path
                return any(path in line and ('cifs' in line or 'smb' in line) for line in mounts.split('\n'))
        except Exception:
            # If we can't check mounts, assume it's not mounted
            return False
    
    def setup_containers(self) -> Dict[str, bool]:
        """
        Create all required directories if they don't exist.
        
        Returns:
            Dictionary with directory names and creation status
        """
        results = {}
        
        for directory_name in self.DIRECTORIES:
            directory_path = os.path.join(self.base_dir, directory_name)
            try:
                os.makedirs(directory_path, exist_ok=True)
                if self.is_azure_files:
                    logger.info(f"Azure Files directory '{directory_path}' created/exists")
                else:
                    logger.info(f"Local directory '{directory_path}' created/exists")
                results[directory_name] = True
            except Exception as e:
                logger.error(f"Error creating directory '{directory_path}': {str(e)}")
                results[directory_name] = False
        
        return results
    
    def _get_file_path(self, container_name: str, file_name: str) -> str:
        """Get the full path for a file in the Azure Files share."""
        return os.path.join(self.base_dir, container_name, file_name)
    
    def _ensure_directory_exists(self, file_path: str) -> bool:
        """Ensure the directory structure exists for the given file path."""
        try:
            directory = os.path.dirname(file_path)
            os.makedirs(directory, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory structure for {file_path}: {str(e)}")
            return False
    
    def upload_file(self, 
                   container_name: str, 
                   blob_name: str, 
                   data: BinaryIO,
                   content_type: Optional[str] = None,
                   metadata: Optional[Dict[str, str]] = None) -> bool:
        """
        Save a file to Azure Files storage.
        
        Args:
            container_name: Target container name
            blob_name: Name for the file (can include path structure)
            data: File-like object containing the data to upload
            content_type: Content type (stored in metadata)
            metadata: Optional metadata to save alongside the file
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Get the full path for the file
            file_path = self._get_file_path(container_name, blob_name)
            
            # Ensure directory structure exists
            if not self._ensure_directory_exists(file_path):
                return False
            
            # Save the file
            with open(file_path, 'wb') as f:
                if hasattr(data, 'read'):
                    f.write(data.read())
                else:
                    f.write(data)
            
            # Save metadata if provided
            if metadata or content_type:
                combined_metadata = metadata or {}
                if content_type:
                    combined_metadata['content_type'] = content_type
                
                metadata_path = f"{file_path}.metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(combined_metadata, f, indent=2)
            
            storage_type = "Azure Files" if self.is_azure_files else "local storage"
            logger.info(f"File saved successfully to {storage_type}: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving file to {container_name}/{blob_name}: {str(e)}")
            return False
    
    def download_file(self, container_name: str, blob_name: str) -> Optional[bytes]:
        """
        Read a file from Azure Files storage.
        
        Args:
            container_name: Source container name
            blob_name: Name of the file to download
            
        Returns:
            File content as bytes or None if download failed
        """
        try:
            file_path = self._get_file_path(container_name, blob_name)
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} does not exist")
                return None
            
            # Read the file
            with open(file_path, 'rb') as f:
                content = f.read()
            
            storage_type = "Azure Files" if self.is_azure_files else "local storage"
            logger.info(f"File downloaded successfully from {storage_type}: {file_path}")
            return content
            
        except Exception as e:
            logger.error(f"Error reading file from {container_name}/{blob_name}: {str(e)}")
            return None
    
    def get_blob_size(self, container_name: str, blob_name: str) -> Optional[int]:
        """
        Get the size of a file in bytes.
        
        Args:
            container_name: Container name
            blob_name: File name
            
        Returns:
            Size of the file in bytes or None if operation failed
        """
        try:
            file_path = self._get_file_path(container_name, blob_name)
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} does not exist")
                return None
            
            # Get file size
            size = os.path.getsize(file_path)
            logger.info(f"File {file_path} size: {size} bytes")
            return size
            
        except Exception as e:
            logger.error(f"Error getting file size for {container_name}/{blob_name}: {str(e)}")
            return None
    
    def download_file_to_path(self, container_name: str, blob_name: str, file_path: str) -> bool:
        """
        Copy a file from Azure Files storage to another path.
        
        Args:
            container_name: Container name
            blob_name: File name
            file_path: Destination path
            
        Returns:
            True if copy was successful, False otherwise
        """
        try:
            # Get the source path
            source_path = self._get_file_path(container_name, blob_name)
            
            # Check if source file exists
            if not os.path.exists(source_path):
                logger.warning(f"Source file {source_path} does not exist")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Copy the file
            shutil.copy2(source_path, file_path)
            logger.info(f"File copied from {source_path} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error copying file from {container_name}/{blob_name} to {file_path}: {str(e)}")
            return False
    
    def save_json(self, container_name: str, blob_name: str, data: Dict[str, Any]) -> bool:
        """
        Save JSON data to Azure Files storage.
        
        Args:
            container_name: Target container name
            blob_name: Name for the file (can include path structure)
            data: Dictionary to be saved as JSON
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Get the full path for the file
            file_path = self._get_file_path(container_name, blob_name)
            
            # Ensure directory structure exists
            if not self._ensure_directory_exists(file_path):
                return False
            
            # Save the JSON data
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Get file size for logging
            file_size = os.path.getsize(file_path)
            storage_type = "Azure Files" if self.is_azure_files else "local storage"
            logger.info(f"✅ JSON saved successfully to {storage_type}: {file_path} ({file_size} bytes)")
            
            # Special logging for realtime storage
            if "realtime" in container_name:
                if "complete_session_result" in blob_name:
                    logger.info(f"🎉 Complete realtime session result saved to Azure Files!")
                elif "realtime_transcript" in blob_name:
                    logger.info(f"💾 Realtime transcript saved to Azure Files")
                elif "session_info" in blob_name:
                    logger.info(f"📋 Session metadata saved to Azure Files")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving JSON to {container_name}/{blob_name}: {str(e)}")
            return False
    
    def load_json(self, container_name: str, blob_name: str) -> Optional[Dict[str, Any]]:
        """
        Load JSON data from Azure Files storage.
        
        Args:
            container_name: Source container name
            blob_name: Name of the file to load
            
        Returns:
            Loaded JSON data or None if load failed
        """
        try:
            file_path = self._get_file_path(container_name, blob_name)
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} does not exist")
                return None
            
            # Load the JSON data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            storage_type = "Azure Files" if self.is_azure_files else "local storage"
            logger.info(f"JSON loaded successfully from {storage_type}: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading JSON from {container_name}/{blob_name}: {str(e)}")
            return None
    
    def get_sas_url(self, container_name: str, blob_name: str, duration_hours: int = 1, permissions=None) -> Optional[str]:
        """
        Get a URL for file access.
        
        For Azure Files mounted via SMB, this returns a local file path URL.
        In a real Azure Files implementation with REST API, this would return a SAS URL.
        
        Args:
            container_name: Container name
            blob_name: File name
            duration_hours: Duration for access (not applicable for mounted shares)
            permissions: Permissions (not applicable for mounted shares)
            
        Returns:
            File path URL
        """
        file_path = self._get_file_path(container_name, blob_name)
        if self.is_azure_files:
            # For mounted Azure Files, return the mounted path
            return f"file://{file_path}"
        else:
            # For local development
            return f"file://{os.path.abspath(file_path)}"
    
    def list_blobs(self, container_name: str, prefix: Optional[str] = None) -> List[str]:
        """
        List files in Azure Files storage.
        
        Args:
            container_name: Container name
            prefix: Optional prefix to filter files
            
        Returns:
            List of file names
        """
        try:
            container_path = os.path.join(self.base_dir, container_name)
            
            # Check if directory exists
            if not os.path.exists(container_path):
                logger.warning(f"Container directory {container_path} does not exist")
                return []
            
            # Get all files in the directory
            files = []
            prefix_path = os.path.join(container_path, prefix) if prefix else container_path
            prefix_dir = os.path.dirname(prefix_path) if prefix else container_path
            
            for root, _, file_names in os.walk(prefix_dir):
                for file_name in file_names:
                    if file_name.endswith('.metadata.json'):
                        continue
                        
                    full_path = os.path.join(root, file_name)
                    relative_path = os.path.relpath(full_path, container_path)
                    
                    if prefix is None or relative_path.startswith(prefix):
                        files.append(relative_path)
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files in {container_name}: {str(e)}")
            return []
    
    def delete_blob(self, container_name: str, blob_name: str, force: bool = False) -> bool:
        """
        Delete a file from Azure Files storage.
        
        Args:
            container_name: Container name
            blob_name: File name
            force: If True, skip safety checks for input files
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            # Safety check: prevent deletion of input files unless forced
            if not force and container_name == self.BATCH_INPUT:
                logger.warning(f"🛡️ PREVENTED deletion of input file {container_name}/{blob_name} (use force=True to override)")
                print(f"🛡️ PREVENTED deletion of input file {container_name}/{blob_name}")
                return False
            
            # Log the call stack to see where this deletion is coming from
            import traceback
            call_stack = traceback.format_stack()
            
            # Use print to ensure it shows up in logs regardless of log level
            print(f"🚨 DELETE_BLOB CALLED for {container_name}/{blob_name}")
            print(f"🔍 Call stack:")
            for line in call_stack[-3:]:  # Reduced to 3 lines for cleaner output
                print(f"   {line.strip()}")
            
            logger.info(f"DELETE_BLOB CALLED for {container_name}/{blob_name}")
            
            file_path = self._get_file_path(container_name, blob_name)
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} does not exist")
                return False
            
            # Delete the file
            os.remove(file_path)
            
            # Delete metadata file if it exists
            metadata_path = f"{file_path}.metadata.json"
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            storage_type = "Azure Files" if self.is_azure_files else "local storage"
            logger.info(f"File deleted from {storage_type}: {file_path}")
            print(f"✅ File deleted from {storage_type}: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file {container_name}/{blob_name}: {str(e)}")
            return False
    
    def blob_exists(self, container_name: str, blob_name: str) -> bool:
        """Check if a file exists in Azure Files storage."""
        file_path = self._get_file_path(container_name, blob_name)
        return os.path.exists(file_path)
    
    def get_chunk_metadata(self, container_name: str, blob_name: str) -> Optional[Dict[str, str]]:
        """Get metadata for a file from Azure Files storage."""
        metadata_path = f"{self._get_file_path(container_name, blob_name)}.metadata.json"
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading metadata for {container_name}/{blob_name}: {str(e)}")
        return None
    
    def cleanup_job_chunks(self, job_id: str) -> bool:
        """Clean up chunks for a job from Azure Files storage."""
        try:
            # Only clean chunks from processing directory, not input
            prefix = f"job-{job_id}"
            for file_name in self.list_blobs(self.BATCH_PROCESSING, prefix):
                if "chunks" in file_name:
                    self.delete_blob(self.BATCH_PROCESSING, file_name, force=True)
            
            storage_type = "Azure Files" if self.is_azure_files else "local storage"
            logger.info(f"Cleaned up job chunks for {job_id} from {storage_type}")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up job chunks for {job_id}: {str(e)}")
            return False

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about the storage configuration."""
        return {
            "storage_type": "Azure Files (SMB)" if self.is_azure_files else "Local Storage",
            "base_path": self.base_dir,
            "is_mounted": self.is_azure_files,
            "directories": self.DIRECTORIES,
            "structure": {
                "batch": {
                    "input": self.BATCH_INPUT,
                    "processing": self.BATCH_PROCESSING,
                    "output": self.BATCH_OUTPUT,
                    "metadata": self.BATCH_METADATA
                },
                "realtime": {
                    "sessions": self.REALTIME_SESSIONS,
                    "chunks": self.REALTIME_CHUNKS,
                    "output": self.REALTIME_OUTPUT,
                    "metadata": self.REALTIME_METADATA
                },
                "shared": {
                    "models": self.SHARED_MODELS,
                    "temp": self.SHARED_TEMP
                }
            }
        }

    # Batch processing helper methods
    def save_batch_input(self, job_id: str, filename: str, data: BinaryIO, metadata: Optional[Dict[str, str]] = None) -> bool:
        """Save audio file for batch processing."""
        return self.upload_file(self.BATCH_INPUT, f"job-{job_id}/{filename}", data, metadata=metadata)
    
    def save_batch_output(self, job_id: str, filename: str, data: Dict[str, Any]) -> bool:
        """Save batch processing results."""
        return self.save_json(self.BATCH_OUTPUT, f"job-{job_id}/{filename}", data)
    
    def get_batch_result(self, job_id: str, filename: str) -> Optional[Dict[str, Any]]:
        """Get batch processing results."""
        return self.load_json(self.BATCH_OUTPUT, f"job-{job_id}/{filename}")
    
    def save_batch_metadata(self, job_id: str, metadata: Dict[str, Any]) -> bool:
        """Save batch job metadata."""
        return self.save_json(self.BATCH_METADATA, f"job-{job_id}/job_info.json", metadata)
    
    def cleanup_batch_job(self, job_id: str, preserve_input: bool = True, preserve_output: bool = True) -> bool:
        """Clean up files for a batch job with selective preservation."""
        try:
            job_prefix = f"job-{job_id}"
            
            # Define which directories to clean based on preservation flags
            directories_to_clean = []
            
            # Always clean processing directory (temporary files)
            directories_to_clean.append(self.BATCH_PROCESSING)
            
            # Conditionally clean other directories
            if not preserve_input:
                directories_to_clean.append(self.BATCH_INPUT)
            if not preserve_output:
                directories_to_clean.append(self.BATCH_OUTPUT)
                directories_to_clean.append(self.BATCH_METADATA)
            
            for directory in directories_to_clean:
                files = self.list_blobs(directory, job_prefix)
                for file_name in files:
                    # Use force=True for processing directory, normal deletion for others
                    force_delete = directory == self.BATCH_PROCESSING
                    self.delete_blob(directory, file_name, force=force_delete)
                    
            preserved_dirs = []
            if preserve_input:
                preserved_dirs.append("input")
            if preserve_output:
                preserved_dirs.append("output")
                
            if preserved_dirs:
                logger.info(f"Cleaned up batch job {job_id} (preserved: {', '.join(preserved_dirs)})")
            else:
                logger.info(f"Cleaned up batch job {job_id} (all files deleted)")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up batch job {job_id}: {str(e)}")
            return False

    # Real-time processing helper methods
    def save_realtime_chunk(self, session_id: str, chunk_id: str, data: BinaryIO, metadata: Optional[Dict[str, str]] = None) -> bool:
        """Save audio chunk for real-time processing."""
        return self.upload_file(self.REALTIME_CHUNKS, f"session-{session_id}/chunk_{chunk_id}.wav", data, metadata=metadata)
    
    def save_realtime_output(self, session_id: str, filename: str, data: Dict[str, Any]) -> bool:
        """Save real-time processing results."""
        return self.save_json(self.REALTIME_OUTPUT, f"session-{session_id}/{filename}", data)
    
    def get_realtime_result(self, session_id: str, filename: str) -> Optional[Dict[str, Any]]:
        """Get real-time processing results."""
        return self.load_json(self.REALTIME_OUTPUT, f"session-{session_id}/{filename}")
    
    def save_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> bool:
        """Save session metadata."""
        return self.save_json(self.REALTIME_METADATA, f"session-{session_id}/session_info.json", metadata)
    
    def cleanup_realtime_session(self, session_id: str, preserve_output: bool = True) -> bool:
        """Clean up files for a real-time session with selective preservation."""
        try:
            session_prefix = f"session-{session_id}"
            
            # Define which directories to clean
            directories_to_clean = [self.REALTIME_SESSIONS, self.REALTIME_CHUNKS]
            
            # Conditionally clean output directories
            if not preserve_output:
                directories_to_clean.extend([self.REALTIME_OUTPUT, self.REALTIME_METADATA])
            
            for directory in directories_to_clean:
                files = self.list_blobs(directory, session_prefix)
                for file_name in files:
                    self.delete_blob(directory, file_name, force=True)
                    
            if preserve_output:
                logger.info(f"Cleaned up real-time session {session_id} (preserved output)")
            else:
                logger.info(f"Cleaned up real-time session {session_id} (all files deleted)")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up real-time session {session_id}: {str(e)}")
            return False

    # Shared utilities
    def save_shared_model(self, model_name: str, data: BinaryIO) -> bool:
        """Save a shared model file."""
        return self.upload_file(self.SHARED_MODELS, model_name, data)
    
    def cleanup_temp_files(self, older_than_hours: int = 24) -> bool:
        """Clean up temporary files older than specified hours."""
        try:
            import time
            cutoff_time = time.time() - (older_than_hours * 3600)
            temp_files = self.list_blobs(self.SHARED_TEMP)
            
            for file_name in temp_files:
                file_path = self._get_file_path(self.SHARED_TEMP, file_name)
                if os.path.exists(file_path) and os.path.getmtime(file_path) < cutoff_time:
                    self.delete_blob(self.SHARED_TEMP, file_name)
                    logger.info(f"Cleaned up old temp file: {file_name}")
            
            return True
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {str(e)}")
            return False

# Backward compatibility alias
AzureBlobStorage = AzureFilesStorage 