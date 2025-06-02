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
    
    # Container names (these will be subdirectories in the Azure Files share)
    INPUT_CONTAINER = "audio-input"
    PROCESSING_CONTAINER = "audio-processing"
    OUTPUT_CONTAINER = "audio-output"
    METADATA_CONTAINER = "audio-metadata"
    
    # List of containers to create
    CONTAINERS = [INPUT_CONTAINER, PROCESSING_CONTAINER, OUTPUT_CONTAINER, METADATA_CONTAINER]
    
    def __init__(self, mount_path: Optional[str] = None):
        """
        Initialize the Azure Files storage adapter.
        
        Args:
            mount_path: Path where Azure Files share is mounted. 
                       Defaults to '/app/storage' (container mount path)
                       Falls back to 'local_storage' for local development
        """
        # Determine the base directory
        if mount_path:
            self.base_dir = mount_path
        elif os.path.exists('/app/storage'):
            # Running in container with mounted Azure Files
            self.base_dir = '/app/storage'
        else:
            # Local development fallback
            self.base_dir = os.path.join(os.getcwd(), 'local_storage')
            logger.warning("Azure Files mount not found, falling back to local storage")
        
        self.is_azure_files = self.base_dir == '/app/storage' or (mount_path and mount_path != 'local_storage')
        self.setup_containers()
        
        if self.is_azure_files:
            logger.info(f"Using Azure Files storage mounted at {self.base_dir}")
        else:
            logger.info(f"Using local file storage in {self.base_dir} (development mode)")
    
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
                if self.is_azure_files:
                    logger.info(f"Azure Files container directory '{container_path}' created/exists")
                else:
                    logger.info(f"Local container directory '{container_path}' created/exists")
                results[container_name] = True
            except Exception as e:
                logger.error(f"Error creating container directory '{container_path}': {str(e)}")
                results[container_name] = False
        
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
            
            storage_type = "Azure Files" if self.is_azure_files else "local storage"
            logger.info(f"JSON saved successfully to {storage_type}: {file_path}")
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
    
    def delete_blob(self, container_name: str, blob_name: str) -> bool:
        """
        Delete a file from Azure Files storage.
        
        Args:
            container_name: Container name
            blob_name: File name
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
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
            for container in self.CONTAINERS:
                prefix = f"job-{job_id}"
                for file_name in self.list_blobs(container, prefix):
                    if "chunks" in file_name:
                        self.delete_blob(container, file_name)
            
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
            "containers": self.CONTAINERS
        }

# Backward compatibility alias
AzureBlobStorage = AzureFilesStorage 