
import requests
import numpy as np
import json

BASE_URL = "http://localhost:8000"

def generate_samples(count, std_dev=0.002):
    return [np.random.normal(0, std_dev, 48000).tolist() for _ in range(count)]

def run_sweep():
    # 1. Enroll with RMS ~ 0.002
    print("Enrolling Device (RMS ~0.002)...")
    enroll_samples = generate_samples(10, std_dev=0.002)
    resp = requests.post(f"{BASE_URL}/enroll", json={
        "device_name": "TestDevice_Sweep",
        "sources": ["microphone"],
        "client_samples": {"microphone": enroll_samples}
    })
    device_id = resp.json()["device_id"]
    print(f"Device ID: {device_id}")
    
    # 2. Sweep Auth RMS
    rms_levels = [0.002, 0.0022, 0.0025, 0.003, 0.004, 0.005, 0.010]
    print("\nStarting RMS Sweep...")
    print(f"{'Target StdDev':<15} | {'Measured RMS':<15} | {'Similarity':<12} | {'Auth'}")
    print("-" * 60)
    
    for std in rms_levels:
        auth_samples = generate_samples(5, std_dev=std)
        # Calculate actual RMS of first sample for reference
        actual_rms = np.sqrt(np.mean(np.array(auth_samples[0])**2))
        
        resp = requests.post(f"{BASE_URL}/authenticate", json={
            "device_id": device_id,
            "sources": ["microphone"],
            "client_samples": {"microphone": auth_samples}
        })
        result = resp.json()
        sim = result.get("similarity", 0)
        auth = result.get("authenticated", False)
        
        print(f"{std:<15.5f} | {actual_rms:<15.5f} | {sim:<12.4f} | {auth}")

if __name__ == "__main__":
    run_sweep()
