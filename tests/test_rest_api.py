#!/usr/bin/env python3
import requests
import json

def test_api():
    base_url = "https://cloudstt-api.lemonforest-4eb4d55a.eastus2.azurecontainerapps.io"
    
    # Test root endpoint
    print(f"Testing root endpoint: {base_url}/")
    try:
        response = requests.get(f"{base_url}/")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test the documentation endpoint
    print("\nTesting docs endpoint: {base_url}/docs")
    try:
        response = requests.get(f"{base_url}/docs")
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            print("Documentation page is available")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api() 