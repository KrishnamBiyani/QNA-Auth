"""
Sensor Jitter and System Noise Collection
Collects timing jitter and system-level noise sources
"""

import time
import numpy as np
from typing import List, Dict
import logging
import psutil
import platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SensorNoiseCollector:
    """Collects system-level noise and timing jitter"""
    
    def __init__(self):
        """Initialize sensor noise collector"""
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information"""
        return {
            'platform': platform.system(),
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }
    
    def collect_timing_jitter(
        self, 
        num_samples: int = 1000,
        target_interval: float = 0.001
    ) -> np.ndarray:
        """
        Collect high-resolution timing jitter
        
        Args:
            num_samples: Number of timing measurements
            target_interval: Target sleep interval in seconds
            
        Returns:
            Array of timing jitter values
        """
        jitter = []
        
        logger.info(f"Collecting {num_samples} timing measurements...")
        
        for _ in range(num_samples):
            start = time.perf_counter()
            time.sleep(target_interval)
            end = time.perf_counter()
            
            # Actual interval minus target interval = jitter
            actual_interval = end - start
            jitter_value = actual_interval - target_interval
            jitter.append(jitter_value)
        
        jitter_array = np.array(jitter)
        logger.info(f"Timing jitter: mean={jitter_array.mean():.6f}s, std={jitter_array.std():.6f}s")
        
        return jitter_array
    
    def collect_cpu_usage_noise(
        self, 
        num_samples: int = 100,
        interval: float = 0.01
    ) -> np.ndarray:
        """
        Collect CPU usage fluctuations as noise source
        
        Args:
            num_samples: Number of measurements
            interval: Sampling interval in seconds
            
        Returns:
            Array of CPU usage percentages
        """
        cpu_samples = []
        
        logger.info(f"Collecting {num_samples} CPU usage samples...")
        
        for _ in range(num_samples):
            cpu_percent = psutil.cpu_percent(interval=interval)
            cpu_samples.append(cpu_percent)
        
        cpu_array = np.array(cpu_samples)
        logger.info(f"CPU usage: mean={cpu_array.mean():.2f}%, std={cpu_array.std():.2f}%")
        
        return cpu_array
    
    def collect_memory_access_noise(
        self, 
        num_iterations: int = 10000
    ) -> np.ndarray:
        """
        Collect memory access timing variations
        
        Args:
            num_iterations: Number of memory operations
            
        Returns:
            Array of memory access times
        """
        timings = []
        data = np.random.rand(1000)
        
        logger.info(f"Collecting {num_iterations} memory access timings...")
        
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = data[np.random.randint(0, len(data))]
            end = time.perf_counter()
            timings.append(end - start)
        
        timing_array = np.array(timings)
        logger.info(f"Memory access: mean={timing_array.mean():.9f}s, std={timing_array.std():.9f}s")
        
        return timing_array
    
    def collect_disk_io_noise(self, num_operations: int = 100) -> np.ndarray:
        """
        Collect disk I/O timing variations
        
        Args:
            num_operations: Number of I/O operations
            
        Returns:
            Array of I/O operation times
        """
        timings = []
        
        logger.info(f"Collecting {num_operations} disk I/O timings...")
        
        # Get disk I/O counters before
        io_before = psutil.disk_io_counters()
        
        for i in range(num_operations):
            start = time.perf_counter()
            io_current = psutil.disk_io_counters()
            end = time.perf_counter()
            
            timings.append(end - start)
            time.sleep(0.01)  # Small delay between measurements
        
        timing_array = np.array(timings)
        logger.info(f"Disk I/O: mean={timing_array.mean():.6f}s, std={timing_array.std():.6f}s")
        
        return timing_array
    
    def collect_network_jitter(self, num_samples: int = 50) -> np.ndarray:
        """
        Collect network statistics as noise source
        
        Args:
            num_samples: Number of measurements
            
        Returns:
            Array of network byte count differences
        """
        network_samples = []
        
        logger.info(f"Collecting {num_samples} network statistics...")
        
        prev_bytes = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
        
        for _ in range(num_samples):
            time.sleep(0.05)
            current = psutil.net_io_counters()
            current_bytes = current.bytes_sent + current.bytes_recv
            diff = current_bytes - prev_bytes
            network_samples.append(diff)
            prev_bytes = current_bytes
        
        network_array = np.array(network_samples, dtype=np.float32)
        logger.info(f"Network jitter: mean={network_array.mean():.2f}, std={network_array.std():.2f}")
        
        return network_array
    
    def collect_combined_system_noise(self) -> Dict[str, np.ndarray]:
        """
        Collect all system noise sources
        
        Returns:
            Dictionary of noise arrays from different sources
        """
        logger.info("Collecting combined system noise...")
        
        noise_sources = {
            'timing_jitter': self.collect_timing_jitter(num_samples=500),
            'cpu_usage': self.collect_cpu_usage_noise(num_samples=50),
            'memory_access': self.collect_memory_access_noise(num_iterations=5000),
            'disk_io': self.collect_disk_io_noise(num_operations=50),
            'network': self.collect_network_jitter(num_samples=30)
        }
        
        return noise_sources
    
    def compute_composite_signature(self, noise_sources: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create a composite signature from multiple noise sources
        
        Args:
            noise_sources: Dictionary of noise arrays
            
        Returns:
            Combined noise signature
        """
        # Normalize each source to [0, 1]
        normalized = []
        
        for name, data in noise_sources.items():
            if len(data) > 0:
                norm = (data - data.min()) / (data.max() - data.min() + 1e-10)
                normalized.append(norm)
        
        # Concatenate all normalized sources
        composite = np.concatenate(normalized)
        
        logger.info(f"Composite signature length: {len(composite)}")
        
        return composite


def main():
    """Test sensor noise collection"""
    collector = SensorNoiseCollector()
    
    print("\n=== System Sensor Noise Collection Test ===")
    print(f"Platform: {collector.system_info['platform']}")
    print(f"Processor: {collector.system_info['processor']}")
    
    # Timing jitter
    print("\n=== Timing Jitter ===")
    jitter = collector.collect_timing_jitter(num_samples=500)
    print(f"Samples: {len(jitter)}")
    print(f"Mean jitter: {jitter.mean():.6f}s")
    print(f"Std jitter: {jitter.std():.6f}s")
    print(f"Range: [{jitter.min():.6f}, {jitter.max():.6f}]s")
    
    # CPU usage noise
    print("\n=== CPU Usage Noise ===")
    cpu = collector.collect_cpu_usage_noise(num_samples=50)
    print(f"Samples: {len(cpu)}")
    print(f"Mean: {cpu.mean():.2f}%")
    print(f"Std: {cpu.std():.2f}%")
    
    # Memory access
    print("\n=== Memory Access Timing ===")
    memory = collector.collect_memory_access_noise(num_iterations=1000)
    print(f"Samples: {len(memory)}")
    print(f"Mean: {memory.mean():.9f}s")
    print(f"Std: {memory.std():.9f}s")
    
    # Combined system noise
    print("\n=== Combined System Noise ===")
    all_noise = collector.collect_combined_system_noise()
    
    print("\nNoise sources collected:")
    for name, data in all_noise.items():
        print(f"  {name}: {len(data)} samples")
    
    # Composite signature
    print("\n=== Composite Signature ===")
    signature = collector.compute_composite_signature(all_noise)
    print(f"Signature length: {len(signature)}")
    print(f"Signature mean: {signature.mean():.6f}")
    print(f"Signature std: {signature.std():.6f}")


if __name__ == "__main__":
    main()
