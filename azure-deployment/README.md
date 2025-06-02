# Azure Deployment Files

This directory contains all files related to deploying the CloudSTT application to Azure Container Apps.

## Files Overview

### Container App Configuration
- `containerapp.yaml` - Base container app configuration
- `containerapp-fixed.yaml` - Fixed container app configuration
- `containerapp-with-mount.yaml` - Configuration with volume mount
- `containerapp-with-volume-mount.yaml` - Enhanced volume mount configuration
- `containerapp-fixed-mount.yaml` - Fixed volume mount configuration
- `container-app-update.yaml` - Update configuration

### Volume Mount Configuration
- `add-volume-mount.json` - JSON configuration for adding volume mounts
- `volume-mount-patch.json` - Patch configuration for volume mounts
- `ingress-update.json` - Ingress configuration updates

### Deployment Scripts
- `setup_azure_files.sh` - Sets up Azure Files storage infrastructure
- `deploy_fixed_storage.sh` - Deploys the app with fixed storage configuration
- `update_container_config.sh` - Updates container configuration

### Documentation
- `AZURE_FILES_SETUP.md` - Detailed setup guide for Azure Files integration

## Usage

1. First run the Azure Files setup:
   ```bash
   ./setup_azure_files.sh
   ```

2. Deploy the application:
   ```bash
   ./deploy_fixed_storage.sh
   ```

3. Update configuration if needed:
   ```bash
   ./update_container_config.sh
   ```

For detailed instructions, see `AZURE_FILES_SETUP.md`. 