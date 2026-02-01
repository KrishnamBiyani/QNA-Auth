#!/usr/bin/env python
"""
Authentic Device Fingerprint Collector

This script collects REAL noise data from the device it's running on:
- QRNG (Quantum Random Number Generator) API responses
- Camera sensor noise (dark frames)
- Microphone self-noise and ambient patterns
- System timing jitter

The collected data creates a unique fingerprint for THIS device only.
No synthetic or fake data - everything is authentic.

Usage:
    python scripts/collect_device_fingerprint.py --device-name "My Laptop" --samples 50
    
The collected fingerprint can then be used with the frontend for authentication.
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import hashlib
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import time

# Project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

# Import noise collectors
from noise_collection.qrng_api import QRNGClient
from noise_collection.camera_noise import CameraNoiseCollector
from noise_collection.mic_noise import MicrophoneNoiseCollector
from noise_collection.sensor_noise import SensorNoiseCollector

# Import preprocessing
try:
    from preprocessing.features import extract_features, FEATURE_VERSION
except ImportError:
    FEATURE_VERSION = "1.0.0"
    extract_features = None


def get_device_id() -> str:
    """
    Generate a unique device ID based on hardware characteristics.
    This ID is consistent for the same device.
    """
    # Collect hardware identifiers
    identifiers = []
    
    # Platform info
    identifiers.append(platform.node())  # Hostname
    identifiers.append(platform.machine())  # CPU architecture
    identifiers.append(platform.processor())  # CPU type
    
    # Try to get MAC address
    try:
        import uuid
        mac = hex(uuid.getnode())
        identifiers.append(mac)
    except:
        pass
    
    # Create hash from identifiers
    combined = "|".join(identifiers)
    device_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    return device_hash


def get_system_info() -> Dict:
    """Collect system information for metadata."""
    return {
        "hostname": platform.node(),
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "collected_at": datetime.now().isoformat()
    }


def collect_qrng_samples(num_samples: int, sample_size: int = 1024) -> List[np.ndarray]:
    """Collect quantum random number samples."""
    print(f"\n📡 Collecting {num_samples} QRNG samples...")
    
    client = QRNGClient()
    samples = []
    
    for i in range(num_samples):
        try:
            sample = client.fetch_quantum_noise(length=sample_size)
            if sample is not None and len(sample) > 0:
                samples.append(sample)
                print(f"   QRNG sample {i+1}/{num_samples} ✓")
            else:
                print(f"   QRNG sample {i+1}/{num_samples} ✗ (empty)")
        except Exception as e:
            print(f"   QRNG sample {i+1}/{num_samples} ✗ ({e})")
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    print(f"   Collected {len(samples)}/{num_samples} QRNG samples")
    return samples


def collect_camera_samples(num_samples: int) -> List[np.ndarray]:
    """Collect camera sensor noise samples (dark frames)."""
    print(f"\n📷 Collecting {num_samples} camera noise samples...")
    print("   (Cover your camera lens for best results)")
    
    collector = CameraNoiseCollector()
    samples = []
    
    if not collector.initialize_camera():
        print("   ✗ Camera not available")
        return samples
    
    try:
        for i in range(num_samples):
            sample = collector.capture_dark_frame()
            if sample is not None:
                # Flatten and subsample for consistent size
                flat = sample.flatten()
                if len(flat) > 1024:
                    flat = flat[:1024]
                samples.append(flat.astype(np.uint8))
                print(f"   Camera sample {i+1}/{num_samples} ✓")
            else:
                print(f"   Camera sample {i+1}/{num_samples} ✗")
            
            time.sleep(0.1)
    finally:
        collector.release()
    
    print(f"   Collected {len(samples)}/{num_samples} camera samples")
    return samples


def collect_microphone_samples(num_samples: int, duration: float = 0.5) -> List[np.ndarray]:
    """Collect microphone noise samples."""
    print(f"\n🎤 Collecting {num_samples} microphone noise samples...")
    print("   (Recording ambient/self-noise)")
    
    collector = MicrophoneNoiseCollector()
    samples = []
    
    for i in range(num_samples):
        try:
            sample = collector.capture_ambient_noise(duration=duration)
            if sample is not None and len(sample) > 0:
                # Normalize to uint8 range
                flat = sample.flatten()
                normalized = ((flat - flat.min()) / (flat.max() - flat.min() + 1e-10) * 255).astype(np.uint8)
                if len(normalized) > 1024:
                    normalized = normalized[:1024]
                samples.append(normalized)
                print(f"   Microphone sample {i+1}/{num_samples} ✓")
            else:
                print(f"   Microphone sample {i+1}/{num_samples} ✗ (empty)")
        except Exception as e:
            print(f"   Microphone sample {i+1}/{num_samples} ✗ ({e})")
        
        time.sleep(0.1)
    
    print(f"   Collected {len(samples)}/{num_samples} microphone samples")
    return samples


def collect_system_jitter_samples(num_samples: int) -> List[np.ndarray]:
    """Collect system timing jitter samples."""
    print(f"\n⏱️  Collecting {num_samples} system jitter samples...")
    
    collector = SensorNoiseCollector()
    samples = []
    
    for i in range(num_samples):
        try:
            sample = collector.collect_timing_jitter(num_measurements=1024)
            if sample is not None and len(sample) > 0:
                # Normalize to uint8 range
                normalized = ((sample - sample.min()) / (sample.max() - sample.min() + 1e-10) * 255).astype(np.uint8)
                samples.append(normalized)
                print(f"   System jitter sample {i+1}/{num_samples} ✓")
            else:
                print(f"   System jitter sample {i+1}/{num_samples} ✗")
        except Exception as e:
            print(f"   System jitter sample {i+1}/{num_samples} ✗ ({e})")
    
    print(f"   Collected {len(samples)}/{num_samples} system jitter samples")
    return samples


def save_fingerprint_data(
    output_dir: Path,
    device_id: str,
    device_name: str,
    samples_by_source: Dict[str, List[np.ndarray]],
    system_info: Dict
) -> Path:
    """Save collected fingerprint data."""
    
    device_dir = output_dir / device_id
    device_dir.mkdir(parents=True, exist_ok=True)
    
    # Save samples for each source
    total_samples = 0
    sources_saved = []
    
    for source, samples in samples_by_source.items():
        if not samples:
            continue
        
        source_dir = device_dir / source
        source_dir.mkdir(exist_ok=True)
        
        for i, sample in enumerate(samples):
            sample_path = source_dir / f"{i:04d}.npy"
            np.save(sample_path, sample)
        
        total_samples += len(samples)
        sources_saved.append(source)
        print(f"   Saved {len(samples)} {source} samples")
    
    # Save metadata
    metadata = {
        "device_id": device_id,
        "device_name": device_name,
        "total_samples": total_samples,
        "sources": sources_saved,
        "samples_per_source": {s: len(samples_by_source.get(s, [])) for s in sources_saved},
        "system_info": system_info,
        "feature_version": FEATURE_VERSION,
        "collected_at": datetime.now().isoformat(),
        "authentic": True  # This is REAL data from this device
    }
    
    metadata_path = device_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Create manifest for the dataset
    manifest_path = output_dir / "manifest.json"
    manifest = {"devices": [], "last_updated": datetime.now().isoformat()}
    
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    
    # Update or add device entry
    device_entry = {
        "device_id": device_id,
        "device_name": device_name,
        "samples": total_samples,
        "sources": sources_saved,
        "collected_at": datetime.now().isoformat()
    }
    
    # Replace if exists, otherwise append
    manifest["devices"] = [d for d in manifest.get("devices", []) if d["device_id"] != device_id]
    manifest["devices"].append(device_entry)
    manifest["last_updated"] = datetime.now().isoformat()
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    return device_dir


def main():
    parser = argparse.ArgumentParser(
        description="Collect authentic device fingerprint data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic collection (50 samples from each source)
    python scripts/collect_device_fingerprint.py --device-name "My Laptop"
    
    # More samples for better accuracy
    python scripts/collect_device_fingerprint.py --device-name "Work PC" --samples 100
    
    # Specific sources only
    python scripts/collect_device_fingerprint.py --device-name "Server" --sources qrng,system
    
    # Quick test (10 samples)
    python scripts/collect_device_fingerprint.py --device-name "Test" --samples 10
        """
    )
    
    parser.add_argument("--device-name", type=str, required=True,
                       help="Human-readable name for this device (e.g., 'My Laptop')")
    parser.add_argument("--samples", type=int, default=50,
                       help="Number of samples to collect from each source (default: 50)")
    parser.add_argument("--sources", type=str, default="qrng,camera,microphone,system",
                       help="Comma-separated noise sources (qrng,camera,microphone,system)")
    parser.add_argument("--output-dir", type=str, default="dataset/samples",
                       help="Output directory for collected data")
    parser.add_argument("--device-id", type=str, default=None,
                       help="Custom device ID (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Parse sources
    sources = [s.strip().lower() for s in args.sources.split(",")]
    valid_sources = ["qrng", "camera", "microphone", "system"]
    sources = [s for s in sources if s in valid_sources]
    
    if not sources:
        print("Error: No valid sources specified")
        print(f"Valid sources: {', '.join(valid_sources)}")
        return 1
    
    # Get or generate device ID
    device_id = args.device_id or get_device_id()
    
    print("=" * 60)
    print("🔐 QNA-Auth Device Fingerprint Collector")
    print("=" * 60)
    print(f"\nDevice Name:     {args.device_name}")
    print(f"Device ID:       {device_id}")
    print(f"Samples/Source:  {args.samples}")
    print(f"Sources:         {', '.join(sources)}")
    print(f"Output:          {args.output_dir}")
    print("\n⚠️  This collects REAL data from YOUR device.")
    print("   The fingerprint will be unique to this hardware.")
    print("=" * 60)
    
    # Collect system info
    system_info = get_system_info()
    
    # Collect samples from each source
    samples_by_source: Dict[str, List[np.ndarray]] = {}
    
    if "qrng" in sources:
        samples_by_source["qrng"] = collect_qrng_samples(args.samples)
    
    if "camera" in sources:
        samples_by_source["camera"] = collect_camera_samples(args.samples)
    
    if "microphone" in sources:
        samples_by_source["microphone"] = collect_microphone_samples(args.samples)
    
    if "system" in sources:
        samples_by_source["system"] = collect_system_jitter_samples(args.samples)
    
    # Check if we got any data
    total_collected = sum(len(s) for s in samples_by_source.values())
    
    if total_collected == 0:
        print("\n❌ No samples collected. Check your hardware connections.")
        return 1
    
    # Save data
    print(f"\n💾 Saving fingerprint data...")
    output_dir = ROOT / args.output_dir
    device_dir = save_fingerprint_data(
        output_dir, device_id, args.device_name,
        samples_by_source, system_info
    )
    
    print("\n" + "=" * 60)
    print("✅ FINGERPRINT COLLECTION COMPLETE")
    print("=" * 60)
    print(f"\nDevice ID:       {device_id}")
    print(f"Device Name:     {args.device_name}")
    print(f"Total Samples:   {total_collected}")
    print(f"Saved to:        {device_dir}")
    
    print("\n📋 Next Steps:")
    print("   1. Run collection on other devices for comparison")
    print("   2. Train the model: python scripts/train_and_evaluate.py")
    print("   3. Start the server: python run.py start")
    print("   4. Use the frontend to authenticate this device")
    print(f"\n   Your Device ID for authentication: {device_id}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
