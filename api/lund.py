import requests
import base64
import os
import json

# Endpoint and authentication
url = "https://stt-unified-endpoint.eastus2.inference.ml.azure.com/score"
api_key = "E0wJKpcVqdhLeWJ4TLIX6ivFXLU75JSDl6VyxzHOK7dP4wLBNp6fJQQJ99BDAAAAAAAAAAAAINFRAZML3zWE"

# Path to your audio file
audio_file_path = "order.wav"  # Change this to your audio file path

# Read and convert the audio file to base64
with open(audio_file_path, "rb") as audio_file:
    audio_bytes = audio_file.read()
    base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

# Create request payload
payload = {
    "audio_data": base64_audio,
    "format": "wav",
    "sample_rate": 16000
}

# Set headers with authentication
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}'
}

# Send the request
try:
    print(f"Sending request to {url}")
    print(f"Headers: {headers}")
    print(f"Payload size: {len(base64_audio)} characters")
    
    response = requests.post(url, json=payload, headers=headers)
    
    # Print response details
    print(f"\nStatus code: {response.status_code}")
    print(f"Response headers: {dict(response.headers)}")
    
    # Try to parse the response as JSON
    try:
        result = response.json()
        print("\nResponse content:")
        print(json.dumps(result, indent=2))
    except json.JSONDecodeError:
        print("\nResponse content (not JSON):")
        print(response.text)
        
except Exception as e:
    print(f"Error: {e}")