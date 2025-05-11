#!/usr/bin/env python3
import requests
import json
import websockets
import asyncio
import os
import sys

API_URL = "https://cloudstt-api.lemonforest-4eb4d55a.eastus2.azurecontainerapps.io"

def test_rest_api():
    """Test the REST API endpoints"""
    print("\n=== Testing REST API ===")
    
    # Test root endpoint
    print(f"\nTesting root endpoint: {API_URL}/")
    try:
        response = requests.get(f"{API_URL}/")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        if response.status_code == 200:
            print("✅ Root endpoint is working")
        else:
            print("❌ Root endpoint returned an error")
    except Exception as e:
        print(f"❌ Error connecting to root endpoint: {e}")
    
    # Test documentation endpoint
    print(f"\nTesting docs endpoint: {API_URL}/docs")
    try:
        response = requests.get(f"{API_URL}/docs")
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print("✅ Documentation endpoint is working")
        else:
            print("❌ Documentation endpoint returned an error")
    except Exception as e:
        print(f"❌ Error connecting to docs endpoint: {e}")

async def test_websocket():
    """Test the WebSocket endpoint"""
    print("\n=== Testing WebSocket API ===")
    
    ws_url = f"wss://{API_URL.replace('https://', '')}/v1/transcribe"
    print(f"\nConnecting to WebSocket: {ws_url}")
    
    try:
        async with websockets.connect(ws_url, max_size=None, ping_interval=30, ping_timeout=10) as websocket:
            print("✅ WebSocket connection established")
            
            # Send a test message
            message = {"type": "ping"}
            await websocket.send(json.dumps(message))
            print(f"Sent: {message}")
            
            # Wait for response (timeout after 10 seconds)
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10)
                print(f"Received: {response}")
                print("✅ WebSocket communication successful")
            except asyncio.TimeoutError:
                print("❌ Timeout waiting for WebSocket response")
    
    except Exception as e:
        print(f"❌ Error with WebSocket connection: {e}")

async def main():
    """Run all tests"""
    print(f"Testing API deployed at: {API_URL}")
    
    # Test REST API
    test_rest_api()
    
    # Test WebSocket
    await test_websocket()
    
    print("\nTest completed!")

if __name__ == "__main__":
    asyncio.run(main()) 