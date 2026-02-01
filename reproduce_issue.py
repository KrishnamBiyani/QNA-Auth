
import requests
import numpy as np
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def generate_samples(count, std_dev=0.002):
    return [np.random.normal(0, std_dev, 48000).tolist() for _ in range(count)]

def run_test():
    print("Generating samples...")
    # Enrollment samples
    enroll_samples = generate_samples(10, std_dev=0.002)
    
    # Auth samples (SAME distribution, DIFFERENT data)
    auth_samples = generate_samples(5, std_dev=0.002)
    
    # 1. Enroll
    print("Enrolling Device...")
    resp = requests.post(f"{BASE_URL}/enroll", json={
        "device_name": "TestDevice_Repro",
        "sources": ["microphone"],
        "client_samples": {"microphone": enroll_samples}
    })
    if resp.status_code != 201:
        print("Enrollment failed:", resp.text)
        return
    
    device_id = resp.json()["device_id"]
    print(f"Device ID: {device_id}")
    
    # 2. Authenticate
    print("Authenticating...")
    resp = requests.post(f"{BASE_URL}/authenticate", json={
        "device_id": device_id,
        "sources": ["microphone"],
        "client_samples": {"microphone": auth_samples}
    })
    
    result = resp.json()
    print("Auth Result:", json.dumps(result, indent=2))
    
    similarity = result.get("similarity", 0)
    authenticated = result.get("authenticated", False)
    
    if authenticated:
        print("SUCCESS: Authenticated correctly.")
    else:
        print(f"FAILURE: Failed to authenticate valid device. Similarity {similarity} < 0.85")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"Error: {e}")
