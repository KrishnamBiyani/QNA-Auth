import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def collect_device_data(device_name, sources, num_samples):
    print(f"Collecting {device_name} ({num_samples} samples)...")
    payload = {
        "device_name": device_name,
        "sources": sources,
        "num_samples": num_samples
    }
    try:
        response = requests.post(f"{BASE_URL}/enroll", json=payload, timeout=300)
        if response.status_code == 201:
            print(f"✅ Success: {device_name}")
        else:
            print(f"❌ Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

profiles = [
    {"name": "Training_Dev1_Cam", "sources": ["camera"], "samples": 10},
    {"name": "Training_Dev2_Mic", "sources": ["microphone"], "samples": 10},
    {"name": "Training_Dev3_Both", "sources": ["camera", "microphone"], "samples": 10},
    {"name": "Training_Dev4_Cam", "sources": ["camera"], "samples": 10},
    {"name": "Training_Dev5_Mic", "sources": ["microphone"], "samples": 10}
]

def main():
    # Wait for server to be ready
    print("Waiting for server...")
    for i in range(10):
        try:
            r = requests.get(f"{BASE_URL}/health")
            if r.status_code == 200:
                print("Server ready!")
                break
        except:
            pass
        time.sleep(2)
    else:
        print("Server failed to start.")
        sys.exit(1)

    print("Starting auto-collection for training...")
    for p in profiles:
        collect_device_data(p["name"], p["sources"], p["samples"])
        time.sleep(1)
    print("Collection Complete.")

if __name__ == "__main__":
    main()
