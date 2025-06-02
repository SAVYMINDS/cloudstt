# Azure Files Storage Setup for CloudSTT

This document explains how to set up and use Azure Files storage with SMB mounting for your CloudSTT application.

## Overview

We've migrated from local blob storage simulation to **Azure Files with SMB mounting**. This provides:

- ✅ **Persistent storage** across container restarts
- ✅ **High performance** with SMB protocol
- ✅ **Shared storage** accessible from multiple container instances
- ✅ **Automatic fallback** to local storage for development

## Architecture

```
Azure Container App
├── Volume Mount: /app/storage
│   └── Azure Files Share (SMB)
│       ├── audio-input/
│       ├── audio-processing/
│       ├── audio-output/
│       └── audio-metadata/
└── Application Code
    └── AzureFilesStorage Class
```

## Files Changed

1. **`storage/azure_storage.py`** - Replaced with new `AzureFilesStorage` implementation
2. **`storage/azure_storage_backup.py`** - Backup of original local storage implementation
3. **`storage/azure_files_storage.py`** - New Azure Files implementation (duplicate for reference)
4. **`setup_azure_files.sh`** - Azure infrastructure setup script
5. **`test_azure_files_storage.py`** - Comprehensive test suite

## Quick Setup

### 1. Set Up Azure Infrastructure

Run the automated setup script:

```bash
# Make sure you're logged into Azure CLI
az login

# Run the setup script
./setup_azure_files.sh
```

This script will:
- Create the storage account `cloudsttstorage`
- Create the file share `cloudsttshare`
- Set up container app storage mount
- Create required directory structure

### 2. Deploy Your Container App

Your existing `container-app-update.yaml` is already configured correctly:

```yaml
volumes:
  - name: storage
    storageType: AzureFile
    storageName: cloudsttstorage
    accessMode: ReadWrite
    shareName: cloudsttshare
containers:
  - name: cloudstt-api-gpu
    volumeMounts:
      - volumeName: storage
        mountPath: /app/storage
```

### 3. Test the Implementation

Run the test suite to verify everything works:

```bash
# Local testing (uses local fallback)
python test_azure_files_storage.py

# In container (uses Azure Files)
# This will automatically detect the mounted share
```

## How It Works

### Automatic Environment Detection

The `AzureFilesStorage` class automatically detects the environment:

```python
# In container with Azure Files mount
if os.path.exists('/app/storage'):
    self.base_dir = '/app/storage'  # Use mounted Azure Files

# Local development fallback
else:
    self.base_dir = 'local_storage'  # Use local storage
```

### Storage Operations

All storage operations work identically whether using Azure Files or local storage:

```python
from storage.azure_storage import AzureFilesStorage

# Initialize (auto-detects environment)
storage = AzureFilesStorage()

# Upload file
with open('audio.wav', 'rb') as f:
    storage.upload_file('audio-input', 'job-123/audio.wav', f)

# Download file
content = storage.download_file('audio-input', 'job-123/audio.wav')

# Save JSON results
results = {"transcription": "Hello world"}
storage.save_json('audio-output', 'job-123/results.json', results)
```

## Configuration Details

### Azure Resources

- **Resource Group**: `mvpv1`
- **Storage Account**: `cloudsttstorage`
- **File Share**: `cloudsttshare`
- **Mount Path**: `/app/storage`
- **Access Mode**: `ReadWrite`

### Directory Structure

```
/app/storage/  (or local_storage/ in development)
├── audio-input/       # Uploaded audio files
├── audio-processing/  # Files being processed
├── audio-output/      # Transcription results
└── audio-metadata/    # Job metadata and status
```

### Environment Variables (Optional)

While not required for mounted shares, you can set these for monitoring:

```bash
AZURE_STORAGE_ACCOUNT_NAME=cloudsttstorage
AZURE_FILES_SHARE_NAME=cloudsttshare
```

## Development vs Production

### Local Development
- Uses `local_storage/` directory
- Falls back automatically when Azure Files mount not available
- Identical API and behavior

### Production (Container)
- Uses Azure Files mounted at `/app/storage`
- Automatic detection via mount path existence
- High performance SMB access

## Troubleshooting

### 1. Mount Issues

Check if Azure Files is properly mounted:

```bash
# In container
ls -la /app/storage
df -h | grep storage
```

### 2. Permissions

Ensure the container has read/write access:

```bash
# Test write access
touch /app/storage/test.txt
rm /app/storage/test.txt
```

### 3. Storage Account Access

Verify storage account configuration:

```bash
az storage account show --name cloudsttstorage --resource-group mvpv1
az storage share show --name cloudsttshare --account-name cloudsttstorage
```

### 4. Container App Storage Mount

Check container app storage configuration:

```bash
az containerapp env storage list --name cloudstt-env --resource-group mvpv1
```

## Performance Considerations

### Azure Files Performance Tiers

- **Standard**: Good for most workloads, cost-effective
- **Premium**: Higher IOPS and throughput for intensive workloads

### SMB Protocol Benefits

- **Native OS integration**: Direct file system access
- **High throughput**: Optimized for large files
- **Low latency**: Minimal overhead compared to REST APIs

### File Organization

- Use subdirectories for job organization: `job-{id}/`
- Implement cleanup policies for old files
- Monitor storage usage and costs

## Migration from Previous Setup

The new implementation maintains **100% API compatibility** with the previous local storage simulation:

- ✅ All method signatures identical
- ✅ Same return types and error handling
- ✅ Same container structure
- ✅ Metadata handling preserved
- ✅ No application code changes required

## Monitoring and Maintenance

### Storage Usage

Monitor file share usage:

```bash
az storage share stats --name cloudsttshare --account-name cloudsttstorage
```

### Cleanup Scripts

Implement regular cleanup of old job files:

```python
# Clean up jobs older than 7 days
storage = AzureFilesStorage()
storage.cleanup_old_jobs(days=7)
```

### Backup Strategy

Consider backup policies for critical transcription results:

- Azure Files backup (if available in your region)
- Regular export to Azure Blob Storage
- Cross-region replication for disaster recovery

## Cost Optimization

### File Share Quota

- Set appropriate quota based on usage patterns
- Monitor and adjust as needed
- Use lifecycle policies for archival

### Access Patterns

- Optimize for read-heavy vs write-heavy workloads
- Consider caching frequently accessed files
- Implement compression for large audio files

## Next Steps

1. **Deploy and test** in your container environment
2. **Monitor performance** and adjust configurations
3. **Implement backup** and retention policies
4. **Set up monitoring** and alerting
5. **Consider premium tier** if performance requirements increase

## Support

For issues with this implementation:

1. Check the test suite output: `python test_azure_files_storage.py`
2. Verify Azure Files mount in container: `ls -la /app/storage`
3. Review container logs for storage-related errors
4. Check Azure portal for storage account status

The implementation provides detailed logging for troubleshooting storage operations. 