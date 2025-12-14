"""
Test noise collection from camera and microphone
"""
import sys
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from noise_collection import CameraNoiseCollector, MicrophoneNoiseCollector
from preprocessing.features import NoisePreprocessor, FeatureVector

print("=" * 70)
print("Testing Camera and Microphone Noise Collection")
print("=" * 70)

preprocessor = NoisePreprocessor()
feature_converter = FeatureVector()

# Test Camera
print("\n" + "="* 70)
print("1. Testing Camera Noise Collection")
print("=" * 70)

try:
    camera_collector = CameraNoiseCollector(camera_index=0)
    if camera_collector.initialize_camera():
        print("✅ Camera initialized")
        
        # Capture a few frames
        frames = camera_collector.capture_multiple_frames(num_frames=3, exposure_time=0.1)
        print(f"✅ Captured {len(frames)} frames")
        
        if frames:
            # Extract noise from first frame
            noise = camera_collector.extract_noise_features(frames[0])
            print(f"✅ Extracted noise: shape={noise.shape}, dtype={noise.dtype}")
            print(f"   Stats: min={noise.min():.2f}, max={noise.max():.2f}, mean={noise.mean():.2f}")
            
            # Try to extract features
            try:
                features = preprocessor.extract_all_features(noise)
                print(f"✅ Extracted {len(features)} features:")
                for key, value in list(features.items())[:5]:
                    print(f"      {key}: {value}")
                
                # Convert to feature vector
                fv = feature_converter.to_vector(features)
                print(f"✅ Feature vector: shape={fv.shape}, dtype={fv.dtype}")
                
            except Exception as e:
                print(f"❌ Feature extraction failed: {e}")
                import traceback
                traceback.print_exc()
        
        camera_collector.release()
        print("✅ Camera released")
    else:
        print("❌ Failed to initialize camera")
        
except Exception as e:
    print(f"❌ Camera test failed: {e}")
    import traceback
    traceback.print_exc()

# Test Microphone
print("\n" + "=" * 70)
print("2. Testing Microphone Noise Collection")
print("=" * 70)

try:
    mic_collector = MicrophoneNoiseCollector(sample_rate=44100)
    print("✅ Microphone collector created")
    
    # Capture samples
    samples = mic_collector.capture_multiple_samples(num_samples=3, duration=0.5)
    print(f"✅ Captured {len(samples)} audio samples")
    
    if samples:
        audio = samples[0]
        print(f"✅ First sample: shape={audio.shape}, dtype={audio.dtype}")
        print(f"   Stats: min={audio.min():.6f}, max={audio.max():.6f}, mean={audio.mean():.6f}")
        
        # Try to extract features
        try:
            features = preprocessor.extract_all_features(audio)
            print(f"✅ Extracted {len(features)} features:")
            for key, value in list(features.items())[:5]:
                print(f"      {key}: {value}")
            
            # Convert to feature vector
            fv = feature_converter.to_vector(features)
            print(f"✅ Feature vector: shape={fv.shape}, dtype={fv.dtype}")
            
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            import traceback
            traceback.print_exc()
    
except Exception as e:
    print(f"❌ Microphone test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)
