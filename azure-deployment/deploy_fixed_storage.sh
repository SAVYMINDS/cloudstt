#!/bin/bash

# Deploy Fixed Azure Files Storage Configuration
# This script updates your container app with the fixed storage mounting and file deletion prevention

set -e

echo "🚀 Deploying Fixed Azure Files Storage Configuration..."

# Configuration
RESOURCE_GROUP="mvpv1"
CONTAINER_APP_NAME="cloudstt-api-gpu"
ACR_NAME="cloudsttacr"
IMAGE_TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📋 Configuration:${NC}"
echo -e "  Resource Group: ${RESOURCE_GROUP}"
echo -e "  Container App: ${CONTAINER_APP_NAME}"
echo -e "  ACR: ${ACR_NAME}"
echo -e "  Image Tag: ${IMAGE_TAG}"
echo ""

# Check if logged into Azure
echo -e "${BLUE}🔐 Checking Azure login...${NC}"
if ! az account show &> /dev/null; then
    echo -e "${RED}❌ Not logged into Azure. Please run 'az login' first.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Azure login verified${NC}"

# Check if container app exists
echo -e "${BLUE}🔍 Checking if container app exists...${NC}"
if ! az containerapp show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo -e "${RED}❌ Container app '$CONTAINER_APP_NAME' not found in resource group '$RESOURCE_GROUP'${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Container app found${NC}"

# Check if storage account exists
echo -e "${BLUE}🗄️ Checking storage account...${NC}"
if ! az storage account show --name cloudsttstorage --resource-group $RESOURCE_GROUP &> /dev/null; then
    echo -e "${RED}❌ Storage account 'cloudsttstorage' not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Storage account verified${NC}"

# Build and push new image with fixes
echo -e "${BLUE}🔨 Building and pushing updated image...${NC}"
echo -e "${YELLOW}⚠️ This will build a new image with the storage fixes${NC}"
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}⏭️ Skipping image build. Using existing image.${NC}"
else
    # Login to ACR
    az acr login --name $ACR_NAME
    
    # Build and push
    docker build -t $ACR_NAME.azurecr.io/cloudstt-api-gpu:$IMAGE_TAG .
    docker push $ACR_NAME.azurecr.io/cloudstt-api-gpu:$IMAGE_TAG
    echo -e "${GREEN}✅ Image built and pushed${NC}"
fi

# Update container app with new configuration
echo -e "${BLUE}🔄 Updating container app configuration...${NC}"
az containerapp update \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --yaml containerapp-fixed-mount.yaml

echo -e "${GREEN}✅ Container app updated successfully${NC}"

# Wait for deployment to complete
echo -e "${BLUE}⏳ Waiting for deployment to complete...${NC}"
sleep 30

# Check deployment status
echo -e "${BLUE}🔍 Checking deployment status...${NC}"
PROVISIONING_STATE=$(az containerapp show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --query "properties.provisioningState" -o tsv)

if [ "$PROVISIONING_STATE" = "Succeeded" ]; then
    echo -e "${GREEN}✅ Deployment completed successfully${NC}"
    
    # Get the app URL
    APP_URL=$(az containerapp show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --query "properties.configuration.ingress.fqdn" -o tsv)
    echo -e "${GREEN}🌐 App URL: https://$APP_URL${NC}"
    
    # Test the health endpoint
    echo -e "${BLUE}🏥 Testing health endpoint...${NC}"
    if curl -s "https://$APP_URL/health" | grep -q "healthy"; then
        echo -e "${GREEN}✅ Health check passed${NC}"
    else
        echo -e "${YELLOW}⚠️ Health check failed - app may still be starting${NC}"
    fi
    
else
    echo -e "${RED}❌ Deployment failed. Status: $PROVISIONING_STATE${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo -e "${BLUE}📋 What's Fixed:${NC}"
echo -e "  ✅ Azure Files mount detection improved"
echo -e "  ✅ Input file deletion prevention enabled"
echo -e "  ✅ Selective cleanup preserves important files"
echo -e "  ✅ Enhanced debugging and logging"
echo ""
echo -e "${BLUE}🧪 Test your API:${NC}"
echo -e "  curl -X POST https://$APP_URL/v1/transcribe \\"
echo -e "    -F \"audio_file=@your_audio.wav\" \\"
echo -e "    -F 'request_data={\"mode\":\"batch\",\"model\":\"tiny\",\"language\":\"en\"}'"
echo ""
echo -e "${BLUE}📊 Monitor logs:${NC}"
echo -e "  az containerapp logs show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --follow" 