#!/usr/bin/env python3
"""
Training Data Collection Script
Collects noise samples from multiple devices for training the Siamese model
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"

def collect_device_data(device_name, noise_sources, num_samples=50):
    """
    Collect data from a device by enrolling it
    
    Args:
        device_name: Unique name for this device
        noise_sources: List of noise sources to use ['qrng', 'camera', 'microphone']
        num_samples: Number of samples to collect (default: 50)
    
    Returns:
        Device ID if successful, None otherwise
    """
    print(f"\n{'='*60}")
    print(f"Collecting data from: {device_name}")
    print(f"Sources: {', '.join(noise_sources)}")
    print(f"Samples: {num_samples}")
    print(f"{'='*60}")
    
    payload = {
        "device_name": device_name,
        "sources": noise_sources,  # API expects 'sources' not 'noise_sources'
        "num_samples": num_samples
    }
    
    try:
        response = requests.post(f"{BASE_URL}/enroll", json=payload, timeout=300)
        
        if response.status_code == 201:
            data = response.json()
            device_id = data.get("device_id")
            print(f"✅ Success! Device ID: {device_id}")
            print(f"   Collected {data['metadata']['num_samples']} samples")
            print(f"   Embedding shape: {data['metadata']['embedding_shape']}")
            return device_id
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"   Error: {response.json()}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def list_enrolled_devices():
    """List all enrolled devices"""
    try:
        response = requests.get(f"{BASE_URL}/devices")
        if response.status_code == 200:
            data = response.json()
            return data.get("devices", [])
    except:
        return []

def main():
    """Main data collection workflow"""
    print("\n" + "="*60)
    print("QNA-Auth Training Data Collection")
    print("="*60)
    
    # Check server health
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Server is not healthy!")
            return
        print("✅ Server is healthy and ready")
    except:
        print("❌ Cannot connect to server!")
        print(f"   Make sure the server is running on {BASE_URL}")
        return
    
    # Show existing devices
    existing = list_enrolled_devices()
    if existing:
        print(f"\n📊 Already enrolled: {len(existing)} devices")
        for dev_id in existing:
            print(f"   - {dev_id}")
    
    print("\n" + "="*60)
    print("Data Collection Strategy:")
    print("="*60)
    print("For best results, collect data from:")
    print("  • 5-10 different physical devices (phones, laptops, desktops)")
    print("  • 50-100 samples per device")
    print("  • Mix of noise sources (QRNG, camera, microphone)")
    print("  • Multiple sessions per device (morning/evening)")
    print("\n" + "="*60)
    
    # Collection profiles - REAL ENTROPY ONLY (camera/microphone)
    profiles = [
        {
            "name": "Device1_Camera",
            "sources": ["camera"],
            "samples": 50,
            "description": "Primary device with camera noise (real entropy)"
        },
        {
            "name": "Device2_Microphone",
            "sources": ["microphone"],
            "samples": 50,
            "description": "Secondary device with microphone noise (real entropy)"
        },
        {
            "name": "Device3_Both",
            "sources": ["camera", "microphone"],
            "samples": 50,
            "description": "Third device with camera + microphone (real entropy)"
        },
        {
            "name": "Device4_Camera_More",
            "sources": ["camera"],
            "samples": 75,
            "description": "Fourth device with more camera samples (real entropy)"
        },
        {
            "name": "Device5_Full_Real",
            "sources": ["camera", "microphone"],
            "samples": 75,
            "description": "Fifth device with all real entropy sources"
        }
    ]
    
    print("\nSuggested Collection Profiles:")
    for i, profile in enumerate(profiles, 1):
        print(f"\n{i}. {profile['name']}")
        print(f"   Sources: {', '.join(profile['sources'])}")
        print(f"   Samples: {profile['samples']}")
        print(f"   {profile['description']}")
    
    print("\n" + "="*60)
    print("Options:")
    print("  1. Collect data for a specific profile (1-5)")
    print("  2. Custom device (manual input)")
    print("  3. Auto-collect all profiles (recommended)")
    print("  4. Exit")
    print("="*60)
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        profile_num = input("Enter profile number (1-5): ").strip()
        try:
            profile = profiles[int(profile_num) - 1]
            collect_device_data(
                profile["name"],
                profile["sources"],
                profile["samples"]
            )
        except:
            print("Invalid profile number")
            
    elif choice == "2":
        name = input("Device name: ").strip()
        print("Available sources: qrng, camera, microphone")
        sources = input("Enter sources (comma-separated): ").strip().split(",")
        sources = [s.strip() for s in sources if s.strip()]
        samples = int(input("Number of samples (default 50): ").strip() or "50")
        
        collect_device_data(name, sources, samples)
        
    elif choice == "3":
        print("\n🚀 Starting auto-collection of all profiles...")
        print("⚠️  This will take 10-30 minutes depending on noise sources")
        print("⚠️  QRNG may be rate-limited, fallback will be used")
        
        confirm = input("\nContinue? (yes/no): ").strip().lower()
        if confirm == "yes":
            successful = 0
            failed = 0
            
            for i, profile in enumerate(profiles, 1):
                print(f"\n\n[{i}/{len(profiles)}] Collecting {profile['name']}...")
                time.sleep(2)  # Small delay between collections
                
                device_id = collect_device_data(
                    profile["name"],
                    profile["sources"],
                    profile["samples"]
                )
                
                if device_id:
                    successful += 1
                else:
                    failed += 1
                    
            print("\n" + "="*60)
            print("Collection Complete!")
            print(f"✅ Successful: {successful}")
            print(f"❌ Failed: {failed}")
            print("="*60)
            
            # Show final device count
            all_devices = list_enrolled_devices()
            print(f"\n📊 Total enrolled devices: {len(all_devices)}")
            
            if len(all_devices) >= 5:
                print("\n🎯 You have enough data to train the model!")
                print("   Next step: python model/train.py")
            else:
                print(f"\n⚠️  Recommended: {5 - len(all_devices)} more devices")
                print("   Collect more data for better model performance")
    
    elif choice == "4":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
