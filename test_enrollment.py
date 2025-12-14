#!/usr/bin/env python3
"""Test script for device enrollment and authentication."""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test server health endpoint."""
    print("Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.status_code == 200

def test_enrollment(device_name, noise_sources, num_samples=20):
    """Test device enrollment."""
    print(f"Enrolling device: {device_name}")
    payload = {
        "device_name": device_name,
        "noise_sources": noise_sources,
        "num_samples": num_samples
    }
    
    response = requests.post(f"{BASE_URL}/enroll", json=payload)
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}\n")
    return data

def test_list_devices():
    """List all enrolled devices."""
    print("Listing all devices...")
    response = requests.get(f"{BASE_URL}/devices")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Devices: {json.dumps(data, indent=2)}\n")
    return data

def test_authentication(device_id, noise_sources):
    """Test device authentication."""
    print(f"Authenticating device: {device_id}")
    payload = {
        "device_id": device_id,
        "noise_sources": noise_sources
    }
    
    response = requests.post(f"{BASE_URL}/authenticate", json=payload)
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}\n")
    return data

def test_challenge_response(device_id):
    """Test challenge-response protocol."""
    print(f"Testing challenge-response for device: {device_id}")
    
    # Request challenge
    print("1. Requesting challenge...")
    response = requests.post(f"{BASE_URL}/challenge", json={"device_id": device_id})
    print(f"Status: {response.status_code}")
    challenge_data = response.json()
    print(f"Challenge: {json.dumps(challenge_data, indent=2)}\n")
    
    # Verify response (normally client would sign the nonce with device secret)
    print("2. Verifying response...")
    verify_payload = {
        "challenge_id": challenge_data["challenge_id"],
        "device_id": device_id,
        "noise_samples": [[1, 2, 3, 4, 5] for _ in range(5)]  # Dummy noise samples
    }
    
    response = requests.post(f"{BASE_URL}/verify", json=verify_payload)
    print(f"Status: {response.status_code}")
    verify_data = response.json()
    print(f"Verification: {json.dumps(verify_data, indent=2)}\n")
    return verify_data

if __name__ == "__main__":
    print("=" * 60)
    print("QNA-Auth System Testing")
    print("=" * 60 + "\n")
    
    # Test health
    if not test_health():
        print("Server is not healthy! Exiting.")
        exit(1)
    
    # Test enrollment
    print("=" * 60)
    print("Testing Device Enrollment")
    print("=" * 60 + "\n")
    
    enrollment_result = test_enrollment(
        device_name="TestDevice_QRNG",
        noise_sources=["qrng"],
        num_samples=20
    )
    
    device_id = enrollment_result.get("device_id")
    if not device_id:
        print("Enrollment failed! Exiting.")
        exit(1)
    
    # List devices
    print("=" * 60)
    print("Listing Enrolled Devices")
    print("=" * 60 + "\n")
    test_list_devices()
    
    # Test authentication
    print("=" * 60)
    print("Testing Device Authentication")
    print("=" * 60 + "\n")
    test_authentication(device_id, ["qrng"])
    
    # Test challenge-response
    print("=" * 60)
    print("Testing Challenge-Response Protocol")
    print("=" * 60 + "\n")
    test_challenge_response(device_id)
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
