"""
Storage device detection and I/O optimization utilities.
Detects if files are on SSD or HDD and optimizes access patterns accordingly.
"""
import os
import platform
import subprocess
from typing import Optional, Dict, List
from enum import Enum
import logging


class StorageType(Enum):
    """Types of storage devices."""
    SSD = "ssd"
    HDD = "hdd"
    NVME = "nvme"
    UNKNOWN = "unknown"
    NETWORK = "network"  # NFS, SMB, etc.


def get_device_for_path(path: str) -> Optional[str]:
    """
    Get the block device for a given path.
    
    Returns:
        Device name (e.g., 'sda', 'nvme0n1') or None
    """
    try:
        # Get the filesystem path
        abs_path = os.path.realpath(path)
        
        # Find mount point
        mount_point = abs_path
        while not os.path.ismount(mount_point) and mount_point != '/':
            mount_point = os.path.dirname(mount_point)
        
        if platform.system() == 'Linux':
            # Use findmnt to get device
            result = subprocess.run(
                ['findmnt', '-n', '-o', 'SOURCE', '--target', mount_point],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                device = result.stdout.strip()
                # Extract base device name
                if device.startswith('/dev/'):
                    device = device[5:]  # Remove /dev/
                if '/' in device:
                    device = device.split('/')[0]  # Get base device (e.g., sda1 -> sda)
                return device
    except Exception as e:
        logging.debug(f"Could not detect device for {path}: {e}")
    return None


def is_rotational_disk(device: str) -> bool:
    """
    Check if a device is a rotational (HDD) disk.
    
    Args:
        device: Device name (e.g., 'sda')
        
    Returns:
        True if rotational (HDD), False if SSD/NVMe
    """
    try:
        # Check /sys/block/{device}/queue/rotational
        rotational_path = f"/sys/block/{device}/queue/rotational"
        if os.path.exists(rotational_path):
            with open(rotational_path, 'r') as f:
                return f.read().strip() == '1'
        
        # For NVMe, check if device starts with nvme
        if device.startswith('nvme'):
            return False
            
    except Exception as e:
        logging.debug(f"Could not determine if {device} is rotational: {e}")
    
    return False  # Assume SSD if unknown


def get_storage_type(path: str) -> StorageType:
    """
    Detect the storage type for a given path.
    
    Args:
        path: File or directory path
        
    Returns:
        StorageType enum
    """
    # Check if it's a network filesystem
    try:
        abs_path = os.path.realpath(path)
        
        # Check mount type
        result = subprocess.run(
            ['stat', '-f', '-c', '%T', abs_path],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            fs_type = result.stdout.strip().lower()
            if fs_type in ('nfs', 'cifs', 'smbfs', 'fuse.sshfs', 'fuse'):
                return StorageType.NETWORK
    except:
        pass
    
    # Get device and check if rotational
    device = get_device_for_path(path)
    if device:
        if device.startswith('nvme'):
            return StorageType.NVME
        elif is_rotational_disk(device):
            return StorageType.HDD
        else:
            return StorageType.SSD
    
    return StorageType.UNKNOWN


def get_io_strategy(storage_type: StorageType) -> Dict[str, any]:
    """
    Get recommended I/O strategy for a storage type.
    
    Returns dict with:
        - use_mmap: Whether to use memory mapping
        - read_ahead: Whether to use read-ahead caching
        - batch_size: Optimal batch size for sequential reads
        - prefetch_size: Number of samples to prefetch
        - parallel_workers: Recommended number of parallel workers
        - cache_to_ram: Whether to cache entire file in RAM
    """
    strategies = {
        StorageType.SSD: {
            'use_mmap': True,
            'read_ahead': False,
            'batch_size': 64,
            'prefetch_size': 4,
            'parallel_workers': 16,
            'cache_to_ram': False,
            'sequential_read': False,  # Random access is fine on SSD
        },
        StorageType.NVME: {
            'use_mmap': True,
            'read_ahead': False,
            'batch_size': 128,  # Larger batches for NVMe
            'prefetch_size': 8,
            'parallel_workers': 32,
            'cache_to_ram': False,
            'sequential_read': False,
        },
        StorageType.HDD: {
            'use_mmap': False,  # Sequential reads better
            'read_ahead': True,
            'batch_size': 256,  # Large sequential reads
            'prefetch_size': 16,
            'parallel_workers': 4,  # Fewer workers to avoid seek thrashing
            'cache_to_ram': True,  # Cache entire file if possible
            'sequential_read': True,  # Critical for HDD performance
        },
        StorageType.NETWORK: {
            'use_mmap': False,
            'read_ahead': True,
            'batch_size': 32,  # Smaller batches for network
            'prefetch_size': 8,
            'parallel_workers': 8,
            'cache_to_ram': True,  # Cache locally
            'sequential_read': False,  # Network latency dominates
        },
        StorageType.UNKNOWN: {
            'use_mmap': True,
            'read_ahead': False,
            'batch_size': 64,
            'prefetch_size': 4,
            'parallel_workers': 8,
            'cache_to_ram': False,
            'sequential_read': False,
        }
    }
    
    return strategies.get(storage_type, strategies[StorageType.UNKNOWN])


def detect_and_log_storage_type(path: str, label: str = "Input") -> StorageType:
    """
    Detect storage type and log recommendation.
    
    Args:
        path: Path to check
        label: Label for logging (e.g., "Input", "Cache")
        
    Returns:
        Detected StorageType
    """
    storage_type = get_storage_type(path)
    strategy = get_io_strategy(storage_type)
    
    logging.info(f"{label} storage type: {storage_type.value.upper()}")
    logging.info(f"  I/O strategy: {strategy}")
    
    if storage_type == StorageType.HDD:
        logging.warning(
            f"{label} files detected on HDD. "
            "Consider using --fast_io mode with --cache_dir on SSD for better performance."
        )
    elif storage_type == StorageType.NETWORK:
        logging.warning(
            f"{label} files detected on network filesystem. "
            "Strongly recommend using --fast_io mode to cache locally."
        )
    
    return storage_type


class OptimizedMRCReader:
    """
    MRC file reader that optimizes access based on storage type.
    """
    
    def __init__(self, file_path: str, storage_type: Optional[StorageType] = None):
        self.file_path = file_path
        self.storage_type = storage_type or get_storage_type(file_path)
        self.strategy = get_io_strategy(self.storage_type)
        self._cache = None
        self._file_handle = None
        
    def __enter__(self):
        import mrcfile
        
        if self.strategy['cache_to_ram']:
            # Load entire file into RAM for HDD/network
            if self._cache is None:
                with mrcfile.open(self.file_path, permissive=True) as mrc:
                    self._cache = mrc.data.copy()
            return self
        elif self.strategy['use_mmap']:
            # Use memory mapping for SSD/NVMe
            self._file_handle = mrcfile.mmap(self.file_path, mode='r', permissive=True)
            return self._file_handle
        else:
            # Regular open for other cases
            self._file_handle = mrcfile.open(self.file_path, permissive=True)
            return self._file_handle
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
    
    @property
    def data(self):
        """Access data, either from cache or file."""
        if self._cache is not None:
            return self._cache
        if self._file_handle is not None:
            return self._file_handle.data
        raise RuntimeError("File not opened. Use as context manager.")


def sort_coords_for_sequential_access(coords: List[tuple], 
                                       tomo_shape: tuple) -> List[tuple]:
    """
    Sort coordinates to optimize for sequential disk access.
    
    For HDDs, sorting by Z coordinate (slowest-changing) first
    minimizes disk seeks.
    
    Args:
        coords: List of (z, y, x) coordinates
        tomo_shape: Shape of tomogram (Z, Y, X)
        
    Returns:
        Sorted coordinates
    """
    # Sort by Z (slowest-changing dimension in memory)
    # This creates sequential access patterns for HDDs
    return sorted(coords, key=lambda c: (c[0], c[1], c[2]))


def batch_coords_by_proximity(coords: List[tuple], 
                               batch_size: int = 64) -> List[List[tuple]]:
    """
    Group coordinates into batches where samples are spatially close.
    
    This minimizes disk seeks by reading nearby regions together.
    
    Args:
        coords: List of (z, y, x) coordinates (should be pre-sorted)
        batch_size: Number of coordinates per batch
        
    Returns:
        List of batches
    """
    # Assume coords are already sorted
    batches = []
    for i in range(0, len(coords), batch_size):
        batches.append(coords[i:i + batch_size])
    return batches