"""
Test hardware access for camera and microphone
"""
import cv2
import sounddevice as sd
import numpy as np

print("=" * 60)
print("Testing Hardware Access")
print("=" * 60)

# Test Camera
print("\n1. Testing Camera...")
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"✅ Camera works! Frame shape: {frame.shape}")
        else:
            print("❌ Camera opened but couldn't capture frame")
        cap.release()
    else:
        print("❌ Could not open camera")
        print("   Possible reasons:")
        print("   - No camera device connected")
        print("   - Camera is being used by another application")
        print("   - Camera access is blocked by permissions")
except Exception as e:
    print(f"❌ Camera error: {e}")

# Test Microphone
print("\n2. Testing Microphone...")
try:
    print("\nAvailable audio devices:")
    devices = sd.query_devices()
    print(devices)
    
    print("\nAttempting to record 1 second...")
    recording = sd.rec(
        int(1.0 * 44100),
        samplerate=44100,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    
    if recording is not None and len(recording) > 0:
        print(f"✅ Microphone works! Recorded {len(recording)} samples")
        print(f"   Audio level: mean={np.mean(np.abs(recording)):.6f}")
    else:
        print("❌ Microphone recorded but got empty data")
        
except Exception as e:
    print(f"❌ Microphone error: {e}")
    print("   Possible reasons:")
    print("   - No microphone device connected")
    print("   - Microphone access is blocked by permissions")
    print("   - Audio driver issues")

print("\n" + "=" * 60)
print("Hardware Test Complete")
print("=" * 60)
