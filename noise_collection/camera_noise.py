"""
Camera Dark Frame Noise Collection
Captures sensor noise from camera with lens cap on or in dark environment
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraNoiseCollector:
    """Collects noise from camera sensor (dark frames)"""
    
    def __init__(self, camera_index: int = 0):
        """
        Initialize camera noise collector
        
        Args:
            camera_index: Index of camera to use (0 for default)
        """
        self.camera_index = camera_index
        self.cap = None
        
    def initialize_camera(self) -> bool:
        """
        Initialize camera connection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_index}")
                return False
            
            # Set camera properties for better noise capture
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Warm up camera (discard first few frames)
            for _ in range(5):
                self.cap.read()
                
            logger.info(f"Camera {self.camera_index} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def capture_dark_frame(
        self, 
        exposure_time: float = 0.1,
        grayscale: bool = True
    ) -> Optional[np.ndarray]:
        """
        Capture a single dark frame (noise sample)
        
        Args:
            exposure_time: Time to wait before capture (simulates exposure)
            grayscale: Convert to grayscale
            
        Returns:
            Numpy array of captured noise or None if failed
        """
        if self.cap is None or not self.cap.isOpened():
            if not self.initialize_camera():
                return None
        
        try:
            # Wait for exposure time
            time.sleep(exposure_time)
            
            # Capture frame
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                logger.error("Failed to capture frame")
                return None
            
            # Convert to grayscale if requested
            if grayscale and len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            logger.info(f"Captured dark frame: shape {frame.shape}")
            return frame
            
        except Exception as e:
            logger.error(f"Error capturing dark frame: {e}")
            return None
    
    def capture_multiple_frames(
        self, 
        num_frames: int = 10,
        exposure_time: float = 0.1
    ) -> list:
        """
        Capture multiple dark frames
        
        Args:
            num_frames: Number of frames to capture
            exposure_time: Exposure time per frame
            
        Returns:
            List of captured frames
        """
        frames = []
        
        for i in range(num_frames):
            frame = self.capture_dark_frame(exposure_time=exposure_time)
            
            if frame is not None:
                frames.append(frame)
                logger.info(f"Captured frame {i+1}/{num_frames}")
            else:
                logger.warning(f"Failed to capture frame {i+1}")
        
        return frames
    
    def extract_noise_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract noise characteristics from dark frame
        
        Args:
            frame: Dark frame image
            
        Returns:
            1D array of noise features
        """
        # Flatten frame to 1D array
        flat = frame.flatten().astype(np.float32)
        
        # Apply high-pass filter to isolate noise
        # Subtract local mean to remove low-frequency components
        kernel_size = 5
        blurred = cv2.GaussianBlur(
            frame, 
            (kernel_size, kernel_size), 
            0
        )
        noise = frame.astype(np.float32) - blurred.astype(np.float32)
        
        return noise.flatten()
    
    def get_temporal_noise(
        self, 
        num_frames: int = 10,
        exposure_time: float = 0.05
    ) -> np.ndarray:
        """
        Compute temporal noise by analyzing frame-to-frame variations
        
        Args:
            num_frames: Number of frames to analyze
            exposure_time: Exposure time per frame
            
        Returns:
            Temporal noise array
        """
        frames = self.capture_multiple_frames(num_frames, exposure_time)
        
        if len(frames) < 2:
            logger.error("Not enough frames for temporal analysis")
            return np.array([])
        
        # Stack frames
        stack = np.stack(frames, axis=0)
        
        # Compute temporal standard deviation (pixel-wise)
        temporal_std = np.std(stack, axis=0)
        
        logger.info(f"Temporal noise shape: {temporal_std.shape}")
        return temporal_std.flatten()
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera released")


def main():
    """Test camera noise collection"""
    collector = CameraNoiseCollector(camera_index=0)
    
    print("\n=== Camera Noise Collection Test ===")
    print("Note: Cover camera lens or ensure dark environment for best results")
    
    # Initialize camera
    if not collector.initialize_camera():
        print("Failed to initialize camera")
        return
    
    # Capture single dark frame
    print("\n=== Capturing Single Dark Frame ===")
    frame = collector.capture_dark_frame(exposure_time=0.2)
    
    if frame is not None:
        print(f"Frame shape: {frame.shape}")
        print(f"Frame mean: {frame.mean():.2f}")
        print(f"Frame std: {frame.std():.2f}")
        print(f"Frame range: [{frame.min()}, {frame.max()}]")
        
        # Extract noise features
        noise = collector.extract_noise_features(frame)
        print(f"\nNoise features: {len(noise)} values")
        print(f"Noise mean: {noise.mean():.4f}")
        print(f"Noise std: {noise.std():.4f}")
    
    # Capture temporal noise
    print("\n=== Capturing Temporal Noise ===")
    temporal = collector.get_temporal_noise(num_frames=5)
    
    if len(temporal) > 0:
        print(f"Temporal noise: {len(temporal)} values")
        print(f"Temporal mean: {temporal.mean():.4f}")
        print(f"Temporal std: {temporal.std():.4f}")
    
    # Clean up
    collector.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
