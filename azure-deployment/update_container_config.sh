#!/bin/bash

# Update Container App Configuration Only
# This script updates your existing container app with the fixed Azure Files storage configuration

set -e

echo "🔄 Updating Container App Configuration..."

# Configuration
RESOURCE_GROUP="mvpv1"
CONTAINER_APP_NAME="cloudstt-api-gpu"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📋 Configuration:${NC}"
echo -e "  Resource Group: ${RESOURCE_GROUP}"
echo -e "  Container App: ${CONTAINER_APP_NAME}"
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

# Get current app URL before update
APP_URL=$(az containerapp show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --query "properties.configuration.ingress.fqdn" -o tsv)
echo -e "${BLUE}🌐 Current App URL: https://$APP_URL${NC}"

# Update container app with new configuration
echo -e "${BLUE}🔄 Updating container app configuration...${NC}"
echo -e "${YELLOW}⚠️ This will update the container app with improved Azure Files mounting${NC}"

az containerapp update \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --yaml containerapp-fixed-mount.yaml

echo -e "${GREEN}✅ Container app configuration updated successfully${NC}"

# Wait for deployment to complete
echo -e "${BLUE}⏳ Waiting for configuration update to complete...${NC}"
sleep 20

# Check deployment status
echo -e "${BLUE}🔍 Checking deployment status...${NC}"
PROVISIONING_STATE=$(az containerapp show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --query "properties.provisioningState" -o tsv)

if [ "$PROVISIONING_STATE" = "Succeeded" ]; then
    echo -e "${GREEN}✅ Configuration update completed successfully${NC}"
    
    # Test the health endpoint
    echo -e "${BLUE}🏥 Testing health endpoint...${NC}"
    sleep 10  # Give it a moment to restart
    
    if curl -s "https://$APP_URL/health" | grep -q "healthy"; then
        echo -e "${GREEN}✅ Health check passed${NC}"
    else
        echo -e "${YELLOW}⚠️ Health check failed - app may still be restarting${NC}"
        echo -e "${BLUE}💡 Try again in a few minutes${NC}"
    fi
    
else
    echo -e "${RED}❌ Configuration update failed. Status: $PROVISIONING_STATE${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 Configuration Update Complete!${NC}"
echo -e "${BLUE}📋 What's Updated:${NC}"
echo -e "  ✅ Azure Files mount path: /app/azurestorage"
echo -e "  ✅ Environment variables for debugging"
echo -e "  ✅ File preservation settings"
echo -e "  ✅ Improved concurrency settings"
echo ""
echo -e "${BLUE}🧪 Test your batch API:${NC}"
echo -e "  curl -X POST https://$APP_URL/v1/transcribe \\"
echo -e "    -F \"audio_file=@storage/order.wav\" \\"
echo -e "    -F 'request_data={\"mode\":\"batch\",\"model\":\"tiny\",\"language\":\"en\",\"compute_type\":\"float32\"}'"
echo ""
echo -e "${BLUE}📊 Monitor logs for file operations:${NC}"
echo -e "  az containerapp logs show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --follow"
echo ""
echo -e "${BLUE}🔍 Look for these log messages:${NC}"
echo -e "  ✅ 'Found Azure Files mount at /app/azurestorage'"
echo -e "  🛡️ 'PREVENTED deletion of input file' (protection working)"
echo -e "  🚨 'DELETE_BLOB CALLED' (with call stack for debugging)" 