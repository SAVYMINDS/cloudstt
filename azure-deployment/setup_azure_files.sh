#!/bin/bash

# Azure Files Setup Script for CloudSTT Application
# This script sets up Azure Files storage account and file share for SMB mounting

set -e

# Configuration - Update these values as needed
RESOURCE_GROUP="mvpv1"
STORAGE_ACCOUNT_NAME="cloudsttstorage"
FILE_SHARE_NAME="cloudsttshare"
LOCATION="westus3"
CONTAINER_ENV_NAME="cloudstt-env-gpu"

echo "🚀 Setting up Azure Files for CloudSTT Application"
echo "=================================================="
echo "Resource Group: $RESOURCE_GROUP"
echo "Storage Account: $STORAGE_ACCOUNT_NAME"
echo "File Share: $FILE_SHARE_NAME"
echo "Location: $LOCATION"
echo ""

# Check if logged in to Azure CLI
echo "✅ Checking Azure CLI authentication..."
if ! az account show &> /dev/null; then
    echo "❌ Please log in to Azure CLI first: az login"
    exit 1
fi

# Check if resource group exists
echo "✅ Checking resource group..."
if ! az group show --name $RESOURCE_GROUP &> /dev/null; then
    echo "❌ Resource group $RESOURCE_GROUP does not exist. Please create it first."
    exit 1
fi

# Create storage account if it doesn't exist
echo "✅ Creating/checking storage account..."
if ! az storage account show --name $STORAGE_ACCOUNT_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo "📦 Creating storage account $STORAGE_ACCOUNT_NAME..."
    az storage account create \
        --name $STORAGE_ACCOUNT_NAME \
        --resource-group $RESOURCE_GROUP \
        --location $LOCATION \
        --sku Standard_LRS \
        --kind StorageV2 \
        --access-tier Hot \
        --allow-blob-public-access false \
        --min-tls-version TLS1_2
    echo "✅ Storage account created successfully!"
else
    echo "✅ Storage account $STORAGE_ACCOUNT_NAME already exists"
fi

# Get storage account key
echo "🔑 Getting storage account key..."
STORAGE_KEY=$(az storage account keys list \
    --resource-group $RESOURCE_GROUP \
    --account-name $STORAGE_ACCOUNT_NAME \
    --query '[0].value' \
    --output tsv)

# Create file share if it doesn't exist
echo "✅ Creating/checking file share..."
if ! az storage share show --name $FILE_SHARE_NAME --account-name $STORAGE_ACCOUNT_NAME --account-key $STORAGE_KEY &> /dev/null; then
    echo "📁 Creating file share $FILE_SHARE_NAME..."
    az storage share create \
        --name $FILE_SHARE_NAME \
        --account-name $STORAGE_ACCOUNT_NAME \
        --account-key $STORAGE_KEY \
        --quota 100
    echo "✅ File share created successfully!"
else
    echo "✅ File share $FILE_SHARE_NAME already exists"
fi

# Create container storage for the Container App Environment
echo "✅ Setting up container app storage..."
if ! az containerapp env storage show \
    --name $STORAGE_ACCOUNT_NAME \
    --environment-name $CONTAINER_ENV_NAME \
    --resource-group $RESOURCE_GROUP &> /dev/null; then
    
    echo "🔗 Creating container app storage mount..."
    az containerapp env storage set \
        --access-mode ReadWrite \
        --azure-file-account-name $STORAGE_ACCOUNT_NAME \
        --azure-file-account-key $STORAGE_KEY \
        --azure-file-share-name $FILE_SHARE_NAME \
        --storage-name $STORAGE_ACCOUNT_NAME \
        --name $CONTAINER_ENV_NAME \
        --resource-group $RESOURCE_GROUP
    echo "✅ Container app storage mount created successfully!"
else
    echo "✅ Container app storage mount already exists"
fi

# Create directory structure in the file share
echo "📂 Creating directory structure in file share..."
# Create the container directories in the file share
for container in "audio-input" "audio-processing" "audio-output" "audio-metadata"; do
    echo "  Creating directory: $container"
    az storage directory create \
        --name $container \
        --share-name $FILE_SHARE_NAME \
        --account-name $STORAGE_ACCOUNT_NAME \
        --account-key $STORAGE_KEY \
        --fail-on-exist false || true
done

echo ""
echo "🎉 Azure Files setup completed successfully!"
echo "============================================"
echo ""
echo "📋 Summary:"
echo "  - Storage Account: $STORAGE_ACCOUNT_NAME"
echo "  - File Share: $FILE_SHARE_NAME"
echo "  - Mount Path: /app/storage"
echo "  - Access Mode: ReadWrite"
echo ""
echo "📝 Environment Variables for your app:"
echo "  AZURE_STORAGE_ACCOUNT_NAME=$STORAGE_ACCOUNT_NAME"
echo "  AZURE_STORAGE_ACCOUNT_KEY=$STORAGE_KEY"
echo "  AZURE_FILES_SHARE_NAME=$FILE_SHARE_NAME"
echo ""
echo "🚢 Your container app configuration should include:"
echo "  volumes:"
echo "    - name: storage"
echo "      storageType: AzureFile"
echo "      storageName: $STORAGE_ACCOUNT_NAME"
echo "      accessMode: ReadWrite"
echo "      shareName: $FILE_SHARE_NAME"
echo "  volumeMounts:"
echo "    - volumeName: storage"
echo "      mountPath: /app/storage"
echo ""
echo "✅ Ready to deploy your application with Azure Files storage!" 