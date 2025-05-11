#!/bin/bash

API_URL="https://cloudstt-api.lemonforest-4eb4d55a.eastus2.azurecontainerapps.io"

echo "Testing API endpoints..."
echo "------------------------"

# Test the root endpoint
echo "1. Testing root endpoint: $API_URL"
curl -s $API_URL

# Test the docs endpoint
echo -e "\n\n2. Testing documentation endpoint: $API_URL/docs"
curl -s -I $API_URL/docs

# Test WebSocket connection
echo -e "\n\n3. Testing WebSocket connection (will timeout after 5 seconds)"
echo "Command: websocat --no-close -t 5 wss://${API_URL#https://}/v1/transcribe"
websocat --no-close -t 5 wss://${API_URL#https://}/v1/transcribe || echo "WebSocket test timed out or connection failed"

echo -e "\n\nAPI test completed" 