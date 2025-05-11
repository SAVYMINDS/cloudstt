"""
Local Storage adapter that mimics Azure Blob Storage interface but saves files locally.
(Replaces Azure Blob Storage for development/testing purposes)
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

class AzureBlobStorage:
    """
    A local file-based storage adapter that mimics the Azure Blob Storage interface.
    Saves files to local directories instead of Azure Blob Storage.
    """
    
    # Container names (these will be subdirectories in the local storage)
    INPUT_CONTAINER = "audio-input"
    PROCESSING_CONTAINER = "audio-processing"
    OUTPUT_CONTAINER = "audio-output"
    METADATA_CONTAINER = "audio-metadata"
    
    # List of containers to create
    CONTAINERS = [INPUT_CONTAINER, PROCESSING_CONTAINER, OUTPUT_CONTAINER, METADATA_CONTAINER]
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the local storage adapter.
        
        Args:
            base_dir: Base directory for local storage. Defaults to 'local_storage' in current directory.
        """
        self.base_dir = base_dir or os.path.join(os.getcwd(), 'local_storage')
        self.setup_containers()
        logger.info(f"Using local file storage in {self.base_dir}")
    
    def setup_containers(self) -> Dict[str, bool]:
        """
        Create all required container directories if they don't exist.
        
        Returns:
            Dictionary with container names and creation status
        """
        results = {}
        
        for container_name in self.CONTAINERS:
            container_path = os.path.join(self.base_dir, container_name)
            try:
                os.makedirs(container_path, exist_ok=True)
                logger.info(f"Container directory '{container_path}' created/exists")
                results[container_name] = True
            except Exception as e:
                logger.error(f"Error creating container directory '{container_path}': {str(e)}")
                results[container_name] = False
        
        return results
    
    def _get_blob_path(self, container_name: str, blob_name: str) -> str:
        """Get the full path for a blob."""
        return os.path.join(self.base_dir, container_name, blob_name)
    
    def upload_file(self, 
                   container_name: str, 
                   blob_name: str, 
                   data: BinaryIO,
                   content_type: Optional[str] = None,
                   metadata: Optional[Dict[str, str]] = None) -> bool:
        """
        Save a file to local storage.
        
        Args:
            container_name: Target container name
            blob_name: Name for the blob (can include path structure)
            data: File-like object containing the data to upload
            content_type: Ignored in local storage
            metadata: Optional metadata to save alongside the file
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Get the full path for the blob
            blob_path = self._get_blob_path(container_name, blob_name)
            
            # Create directory structure
            os.makedirs(os.path.dirname(blob_path), exist_ok=True)
            
            # Save the file
            with open(blob_path, 'wb') as f:
                f.write(data.read())
            
            # Save metadata if provided
            if metadata:
                metadata_path = f"{blob_path}.metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f)
            
            logger.info(f"File saved successfully to {blob_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving file to {container_name}/{blob_name}: {str(e)}")
            return False
    
    def download_file(self, container_name: str, blob_name: str) -> Optional[bytes]:
        """
        Read a file from local storage.
        
        Args:
            container_name: Source container name
            blob_name: Name of the blob to download
            
        Returns:
            File content as bytes or None if download failed
        """
        try:
            blob_path = self._get_blob_path(container_name, blob_name)
            
            # Check if file exists
            if not os.path.exists(blob_path):
                logger.warning(f"File {blob_path} does not exist")
                return None
            
            # Read the file
            with open(blob_path, 'rb') as f:
                return f.read()
            
        except Exception as e:
            logger.error(f"Error reading file from {container_name}/{blob_name}: {str(e)}")
            return None
    
    def get_blob_size(self, container_name: str, blob_name: str) -> Optional[int]:
        """
        Get the size of a local file in bytes.
        
        Args:
            container_name: Container name
            blob_name: Blob name
            
        Returns:
            Size of the file in bytes or None if operation failed
        """
        try:
            blob_path = self._get_blob_path(container_name, blob_name)
            
            # Check if file exists
            if not os.path.exists(blob_path):
                logger.warning(f"File {blob_path} does not exist")
                return None
            
            # Get file size
            size = os.path.getsize(blob_path)
            logger.info(f"File {blob_path} size: {size} bytes")
            return size
            
        except Exception as e:
            logger.error(f"Error getting file size for {container_name}/{blob_name}: {str(e)}")
            return None
    
    def download_file_to_path(self, container_name: str, blob_name: str, file_path: str) -> bool:
        """
        Copy a file from local storage to another path.
        
        Args:
            container_name: Container name
            blob_name: Blob name
            file_path: Destination path
            
        Returns:
            True if copy was successful, False otherwise
        """
        try:
            # Get the source path
            source_path = self._get_blob_path(container_name, blob_name)
            
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
        Save JSON data to local storage.
        
        Args:
            container_name: Target container name
            blob_name: Name for the blob (can include path structure)
            data: Dictionary to be saved as JSON
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Get the full path for the blob
            blob_path = self._get_blob_path(container_name, blob_name)
            
            # Create directory structure
            os.makedirs(os.path.dirname(blob_path), exist_ok=True)
            
            # Save the JSON data
            with open(blob_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"JSON saved successfully to {blob_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving JSON to {container_name}/{blob_name}: {str(e)}")
            return False
    
    def load_json(self, container_name: str, blob_name: str) -> Optional[Dict[str, Any]]:
        """
        Load JSON data from local storage.
        
        Args:
            container_name: Source container name
            blob_name: Name of the blob to load
            
        Returns:
            Loaded JSON data or None if load failed
        """
        try:
            blob_path = self._get_blob_path(container_name, blob_name)
            
            # Check if file exists
            if not os.path.exists(blob_path):
                logger.warning(f"File {blob_path} does not exist")
                return None
            
            # Load the JSON data
            with open(blob_path, 'r') as f:
                return json.load(f)
            
        except Exception as e:
            logger.error(f"Error loading JSON from {container_name}/{blob_name}: {str(e)}")
            return None
    
    def get_sas_url(self, container_name: str, blob_name: str, duration_hours: int = 1, permissions=None) -> Optional[str]:
        """
        Get a URL for local file access (mocked implementation).
        
        Args:
            container_name: Container name
            blob_name: Blob name
            duration_hours: Ignored in local storage
            permissions: Ignored in local storage
            
        Returns:
            Local file path URL
        """
        blob_path = self._get_blob_path(container_name, blob_name)
        return f"file://{os.path.abspath(blob_path)}"
    
    def list_blobs(self, container_name: str, prefix: Optional[str] = None) -> List[str]:
        """
        List files in local storage.
        
        Args:
            container_name: Container name
            prefix: Optional prefix to filter files
            
        Returns:
            List of blob names
        """
        try:
            container_path = os.path.join(self.base_dir, container_name)
            
            # Check if directory exists
            if not os.path.exists(container_path):
                logger.warning(f"Container directory {container_path} does not exist")
                return []
            
            # Get all files in the directory
            blobs = []
            prefix_path = os.path.join(container_path, prefix) if prefix else container_path
            prefix_dir = os.path.dirname(prefix_path) if prefix else container_path
            
            for root, _, files in os.walk(prefix_dir):
                for file in files:
                    if file.endswith('.metadata.json'):
                        continue
                        
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, container_path)
                    
                    if prefix is None or relative_path.startswith(prefix):
                        blobs.append(relative_path)
            
            return blobs
            
        except Exception as e:
            logger.error(f"Error listing files in {container_name}: {str(e)}")
            return []
    
    def delete_blob(self, container_name: str, blob_name: str) -> bool:
        """
        Delete a file from local storage.
        
        Args:
            container_name: Container name
            blob_name: Blob name
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            blob_path = self._get_blob_path(container_name, blob_name)
            
            # Check if file exists
            if not os.path.exists(blob_path):
                logger.warning(f"File {blob_path} does not exist")
                return False
            
            # Delete the file
            os.remove(blob_path)
            
            # Delete metadata file if it exists
            metadata_path = f"{blob_path}.metadata.json"
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            logger.info(f"File deleted: {blob_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file {container_name}/{blob_name}: {str(e)}")
            return False
    
    def blob_exists(self, container_name: str, blob_name: str) -> bool:
        """Check if a blob exists in local storage."""
        blob_path = self._get_blob_path(container_name, blob_name)
        return os.path.exists(blob_path)
    
    def get_chunk_metadata(self, container_name: str, blob_name: str) -> Optional[Dict[str, str]]:
        """Get metadata for a chunk from local storage."""
        metadata_path = f"{self._get_blob_path(container_name, blob_name)}.metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def cleanup_job_chunks(self, job_id: str) -> bool:
        """Clean up chunks for a job from local storage."""
        try:
            for container in self.CONTAINERS:
                prefix = f"job-{job_id}"
                for blob in self.list_blobs(container, prefix):
                    if "chunks" in blob:
                        self.delete_blob(container, blob)
            return True
        except Exception as e:
            logger.error(f"Error cleaning up job chunks for {job_id}: {str(e)}")
            return False 