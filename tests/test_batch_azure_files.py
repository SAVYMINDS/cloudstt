#!/usr/bin/env python3
"""
Test script for Azure Files batch processing functionality.
This script tests the storage mounting, file operations, and batch processing workflow.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storage.azure_storage import AzureFilesStorage

def test_storage_mounting():
    """Test Azure Files storage mounting and detection."""
    print("🔍 Testing Azure Files Storage Mounting...")
    
    # Test with different mount paths
    test_cases = [
        ("/app/azurestorage", "Container Azure Files mount"),
        ("/app/storage", "Legacy Azure Files mount"),
        ("local_storage", "Local development fallback"),
        (None, "Auto-detection")
    ]
    
    for mount_path, description in test_cases:
        print(f"\n📁 Testing {description}: {mount_path}")
        try:
            storage = AzureFilesStorage(mount_path=mount_path)
            info = storage.get_storage_info()
            
            print(f"   ✅ Storage Type: {info['storage_type']}")
            print(f"   📂 Base Path: {info['base_path']}")
            print(f"   🔗 Is Mounted: {info['is_mounted']}")
            print(f"   📋 Directories: {len(info['directories'])} created")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

def test_batch_file_operations():
    """Test batch file operations with safety features."""
    print("\n🧪 Testing Batch File Operations...")
    
    storage = AzureFilesStorage()
    test_job_id = "test-batch-001"
    
    # Test file upload to input
    print(f"\n📤 Testing file upload to batch input...")
    test_content = b"This is test audio content for batch processing"
    
    try:
        # Save to input directory
        success = storage.save_batch_input(
            job_id=test_job_id,
            filename="test_audio.wav",
            data=tempfile.NamedTemporaryFile(delete=False),
            metadata={"original_name": "test_audio.wav", "size": len(test_content)}
        )
        
        # Write actual content
        input_path = storage._get_file_path(storage.BATCH_INPUT, f"job-{test_job_id}/test_audio.wav")
        with open(input_path, 'wb') as f:
            f.write(test_content)
        
        print(f"   ✅ File saved to input: {input_path}")
        
        # Test file exists
        exists = storage.blob_exists(storage.BATCH_INPUT, f"job-{test_job_id}/test_audio.wav")
        print(f"   ✅ File exists check: {exists}")
        
        # Test file download
        downloaded = storage.download_file(storage.BATCH_INPUT, f"job-{test_job_id}/test_audio.wav")
        print(f"   ✅ File downloaded: {len(downloaded)} bytes")
        
    except Exception as e:
        print(f"   ❌ Error in file operations: {str(e)}")

def test_deletion_safety():
    """Test the file deletion safety features."""
    print("\n🛡️ Testing File Deletion Safety...")
    
    storage = AzureFilesStorage()
    test_job_id = "test-safety-001"
    
    # Create test files in different directories
    test_files = [
        (storage.BATCH_INPUT, "input_file.wav"),
        (storage.BATCH_PROCESSING, "processing_file.wav"),
        (storage.BATCH_OUTPUT, "output_file.json")
    ]
    
    for container, filename in test_files:
        file_path = storage._get_file_path(container, f"job-{test_job_id}/{filename}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(f"Test content for {filename}")
        print(f"   📝 Created test file: {container}/{filename}")
    
    # Test deletion with safety features
    print(f"\n🚫 Testing input file deletion (should be prevented)...")
    result = storage.delete_blob(storage.BATCH_INPUT, f"job-{test_job_id}/input_file.wav")
    print(f"   🛡️ Input deletion prevented: {not result}")
    
    print(f"\n✅ Testing processing file deletion (should work)...")
    result = storage.delete_blob(storage.BATCH_PROCESSING, f"job-{test_job_id}/processing_file.wav", force=True)
    print(f"   🗑️ Processing file deleted: {result}")
    
    print(f"\n🔓 Testing forced input deletion (should work)...")
    result = storage.delete_blob(storage.BATCH_INPUT, f"job-{test_job_id}/input_file.wav", force=True)
    print(f"   🗑️ Forced input deletion: {result}")

def test_batch_cleanup():
    """Test batch job cleanup with preservation options."""
    print("\n🧹 Testing Batch Job Cleanup...")
    
    storage = AzureFilesStorage()
    test_job_id = "test-cleanup-001"
    
    # Create test files in all directories
    test_data = {
        storage.BATCH_INPUT: ["audio.wav", "metadata.json"],
        storage.BATCH_PROCESSING: ["temp_chunk1.wav", "temp_chunk2.wav"],
        storage.BATCH_OUTPUT: ["results.json", "summary.json"],
        storage.BATCH_METADATA: ["job_info.json"]
    }
    
    print(f"   📝 Creating test files for job {test_job_id}...")
    for container, filenames in test_data.items():
        for filename in filenames:
            file_path = storage._get_file_path(container, f"job-{test_job_id}/{filename}")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(f"Test content for {filename}")
    
    # Test cleanup with preservation (default behavior)
    print(f"\n🛡️ Testing cleanup with preservation (default)...")
    storage.cleanup_batch_job(test_job_id)  # Should preserve input and output
    
    # Check what remains
    for container, filenames in test_data.items():
        remaining = storage.list_blobs(container, f"job-{test_job_id}")
        print(f"   📂 {container}: {len(remaining)} files remaining")

def test_directory_structure():
    """Test the directory structure creation."""
    print("\n📁 Testing Directory Structure...")
    
    storage = AzureFilesStorage()
    
    print(f"   📂 Base directory: {storage.base_dir}")
    print(f"   🔗 Is Azure Files: {storage.is_azure_files}")
    
    # Check all required directories
    for directory in storage.DIRECTORIES:
        dir_path = os.path.join(storage.base_dir, directory)
        exists = os.path.exists(dir_path)
        print(f"   {'✅' if exists else '❌'} {directory}: {dir_path}")

def main():
    """Run all tests."""
    print("🚀 Azure Files Batch Processing Test Suite")
    print("=" * 50)
    
    try:
        test_storage_mounting()
        test_directory_structure()
        test_batch_file_operations()
        test_deletion_safety()
        test_batch_cleanup()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 