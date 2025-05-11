#!/usr/bin/env python3
import asyncio
import websockets
import json

async def test_websocket():
    uri = "wss://cloudstt-api.lemonforest-4eb4d55a.eastus2.azurecontainerapps.io/v1/transcribe"
    
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected successfully!")
            
            # Send a test message
            test_message = {"type": "ping"}
            await websocket.send(json.dumps(test_message))
            print(f"Sent: {test_message}")
            
            # Wait for response
            response = await websocket.recv()
            print(f"Received: {response}")
            
    except Exception as e:
        print(f"Error connecting to WebSocket: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket()) 