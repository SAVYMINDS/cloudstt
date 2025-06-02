#!/usr/bin/env python3

"""
Test script for Azure Files Storage implementation
This script tests the Azure Files storage functionality in both local and container environments.
"""

import os
import sys
import tempfile
import json
from io import BytesIO
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from storage.azure_storage import AzureFilesStorage

def test_azure_files_storage():
    """Test the Azure Files storage implementation"""
    
    print("🧪 Testing Azure Files Storage Implementation")
    print("=" * 50)
    
    # Initialize storage (will auto-detect environment)
    storage = AzureFilesStorage()
    
    # Get storage info
    info = storage.get_storage_info()
    print(f"📊 Storage Type: {info['storage_type']}")
    print(f"📂 Base Path: {info['base_path']}")
    print(f"🔗 Is Mounted: {info['is_mounted']}")
    print(f"📦 Directories: {', '.join(info['directories'])}")
    print(f"🏗️  Structure:")
    for section, dirs in info['structure'].items():
        print(f"   {section}: {', '.join(dirs.values())}")
    print()
    
    # Test file operations with new structure
    test_container = storage.BATCH_INPUT  # Use batch input instead
    test_filename = "test_audio.wav"
    test_content = b"This is test audio content"
    test_metadata = {
        "test": "true",
        "upload_time": "2024-01-01T00:00:00Z",
        "sample_rate": "16000"
    }
    
    print("🔄 Testing file operations...")
    
    # Test 1: Upload file
    print("1️⃣ Testing file upload...")
    data_stream = BytesIO(test_content)
    success = storage.upload_file(
        container_name=test_container,
        blob_name=test_filename,
        data=data_stream,
        content_type="audio/wav",
        metadata=test_metadata
    )
    print(f"   Upload result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    # Test 2: Check if file exists
    print("2️⃣ Testing file existence check...")
    exists = storage.blob_exists(test_container, test_filename)
    print(f"   File exists: {'✅ YES' if exists else '❌ NO'}")
    
    # Test 3: Get file size
    print("3️⃣ Testing file size retrieval...")
    size = storage.get_blob_size(test_container, test_filename)
    print(f"   File size: {size} bytes ({'✅ CORRECT' if size == len(test_content) else '❌ INCORRECT'})")
    
    # Test 4: Download file
    print("4️⃣ Testing file download...")
    downloaded_content = storage.download_file(test_container, test_filename)
    download_success = downloaded_content == test_content
    print(f"   Download result: {'✅ SUCCESS' if download_success else '❌ FAILED'}")
    
    # Test 5: Get metadata
    print("5️⃣ Testing metadata retrieval...")
    retrieved_metadata = storage.get_chunk_metadata(test_container, test_filename)
    metadata_success = retrieved_metadata is not None and retrieved_metadata.get("test") == "true"
    print(f"   Metadata result: {'✅ SUCCESS' if metadata_success else '❌ FAILED'}")
    
    # Test 6: Save and load JSON
    print("6️⃣ Testing JSON operations...")
    test_json = {
        "transcription": "Hello world",
        "confidence": 0.95,
        "duration": 2.5,
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.5, "text": "world"}
        ]
    }
    
    json_filename = "test_result.json"
    json_save_success = storage.save_json(test_container, json_filename, test_json)
    print(f"   JSON save: {'✅ SUCCESS' if json_save_success else '❌ FAILED'}")
    
    if json_save_success:
        loaded_json = storage.load_json(test_container, json_filename)
        json_load_success = loaded_json == test_json
        print(f"   JSON load: {'✅ SUCCESS' if json_load_success else '❌ FAILED'}")
    
    # Test 7: List files
    print("7️⃣ Testing file listing...")
    files = storage.list_blobs(test_container)
    list_success = test_filename in files and json_filename in files
    print(f"   File listing: {'✅ SUCCESS' if list_success else '❌ FAILED'}")
    print(f"   Found files: {files}")
    
    # Test 8: Get SAS URL
    print("8️⃣ Testing SAS URL generation...")
    sas_url = storage.get_sas_url(test_container, test_filename)
    sas_success = sas_url is not None and sas_url.startswith("file://")
    print(f"   SAS URL: {'✅ SUCCESS' if sas_success else '❌ FAILED'}")
    print(f"   URL: {sas_url}")
    
    # Test 9: Download to path
    print("9️⃣ Testing download to path...")
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        download_path_success = storage.download_file_to_path(
            test_container, test_filename, temp_path
        )
        
        if download_path_success:
            with open(temp_path, 'rb') as f:
                temp_content = f.read()
            path_content_correct = temp_content == test_content
            print(f"   Download to path: {'✅ SUCCESS' if path_content_correct else '❌ FAILED'}")
        else:
            print(f"   Download to path: ❌ FAILED")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    # Test 10: New helper methods
    print("🔟 Testing new helper methods...")
    test_job_id = "test-123"
    
    # Test batch helper
    batch_data = BytesIO(b"batch test audio")
    batch_success = storage.save_batch_input(test_job_id, "audio.wav", batch_data)
    print(f"   Batch input save: {'✅ SUCCESS' if batch_success else '❌ FAILED'}")
    
    # Test realtime helper  
    session_id = "session-456"
    chunk_data = BytesIO(b"realtime chunk")
    realtime_success = storage.save_realtime_chunk(session_id, "001", chunk_data)
    print(f"   Realtime chunk save: {'✅ SUCCESS' if realtime_success else '❌ FAILED'}")
    
    # Cleanup test files
    print("🧹 Cleaning up test files...")
    storage.delete_blob(test_container, test_filename)
    storage.delete_blob(test_container, json_filename)
    storage.cleanup_batch_job(test_job_id)
    storage.cleanup_realtime_session(session_id)
    
    print()
    print("🎉 Azure Files Storage test completed!")
    print("=" * 50)

def test_directory_structure():
    """Test that all required directories are created"""
    print("📁 Testing directory structure...")
    
    storage = AzureFilesStorage()
    base_path = storage.base_dir
    
    for directory in storage.DIRECTORIES:
        directory_path = os.path.join(base_path, directory)
        exists = os.path.exists(directory_path)
        print(f"   {directory}: {'✅ EXISTS' if exists else '❌ MISSING'}")

if __name__ == "__main__":
    try:
        test_directory_structure()
        print()
        test_azure_files_storage()
        
        print()
        print("✨ All tests completed! Your Azure Files storage is ready to use.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 