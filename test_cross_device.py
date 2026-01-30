
import requests
import numpy as np
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CROSS_DEVICE_TEST")

BASE_URL = "http://localhost:8000"

def generate_samples(mean, std, count=10, size=48000):
    """Generate synthetic noise samples."""
    return [np.random.normal(mean, std, size).tolist() for _ in range(count)]

def run_cross_device_test():
    logger.info(" starting Cross-Device Authentication Test...")

    # 1. Enroll Device A (Quiet Profile)
    logger.info("\n[Step 1] Enrolling 'Device A' (Quiet Profile)...")
    initial_samples = generate_samples(0.0, 0.002, count=20) # RMS ~0.002
    
    enroll_payload = {
        "device_name": "Device_A_Quiet",
        "num_samples": 20,
        "sources": ["microphone"],
        "client_samples": {
            "microphone": initial_samples
        }
    }
    
    resp = requests.post(f"{BASE_URL}/enroll", json=enroll_payload)
    if resp.status_code != 201:
        logger.error(f"Enrollment failed: {resp.text}")
        return
        
    device_a_id = resp.json()["device_id"]
    logger.info(f" -> Enrolled Device A with ID: {device_a_id}")

    # 2. Authenticate Device A with correct samples (Quiet)
    logger.info("\n[Step 2] Authenticating Device A with Matching (Quiet) samples...")
    auth_samples = generate_samples(0.0, 0.002, count=5) # Same profile
    
    auth_payload = {
        "device_id": device_a_id,
        "sources": ["microphone"],
        "num_samples_per_source": 5,
        "client_samples": {
            "microphone": auth_samples
        }
    }
    
    resp = requests.post(f"{BASE_URL}/authenticate", json=auth_payload)
    result = resp.json()
    logger.info(f" -> Result: Authenticated={result['authenticated']}, Similarity={result['similarity']:.4f}")
    
    if not result['authenticated']:
        logger.error("FAILURE: Self-authentication failed!")
    else:
        logger.info("SUCCESS: Self-authentication passed.")

    # 3. Authenticate Device A with DIFFERENT samples (Loud) - Simulating Device B
    logger.info("\n[Step 3] Attacking Device A with Device B (Loud) samples...")
    attack_samples = generate_samples(0.0, 0.010, count=5) # 5x louder, RMS ~0.010
    
    attack_payload = {
        "device_id": device_a_id,
        "sources": ["microphone"],
        "num_samples_per_source": 5,
        "client_samples": {
            "microphone": attack_samples
        }
    }
    
    resp = requests.post(f"{BASE_URL}/authenticate", json=attack_payload)
    result = resp.json()
    logger.info(f" -> Result: Authenticated={result['authenticated']}, Similarity={result['similarity']:.4f}")
    
    if result['authenticated']:
        logger.error("FAILURE: False Acceptance! Different device was authenticated.")
    else:
        logger.info("SUCCESS: Attack rejected. Cross-device authentication works!")

if __name__ == "__main__":
    run_cross_device_test()
