"""
Fast dataset with memory-mapped cache and async prefetching.
Optimized for HPC environments with abundant CPU RAM (64-512GB).
"""
import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool, cpu_count
from threading import Thread
import queue
import mmap
from typing import Optional, Tuple, List
import hashlib
import time

from IsoNet.models.data_sequence import Train_sets_n2n
from IsoNet.utils.fileio import read_mrc
from IsoNet.utils.storage import (
    get_storage_type, StorageType, detect_and_log_storage_type
)


class AsyncPrefetcher:
    """
    Double-buffered async prefetcher for CPU->GPU transfer.
    Overlaps data loading with GPU computation.
    """
    def __init__(self, dataset, batch_size: int, num_batches: int = 2, device='cuda'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.device = device
        self.current_idx = 0

        # Pre-allocate pinned memory buffers for zero-copy transfer
        sample_shape = dataset.get_sample_shape()
        self.buffers = [
            torch.empty((batch_size, *sample_shape), dtype=torch.float32, pin_memory=True)
            for _ in range(num_batches)
        ]
        self.buffer_idx = 0
        self.prefetch_queue = queue.Queue(maxsize=num_batches)
        self.prefetch_thread = None
        self.stop_event = False

    def start(self):
        """Start prefetching thread."""
        self.prefetch_thread = Thread(target=self._prefetch_loop, daemon=True)
        self.prefetch_thread.start()

    def stop(self):
        """Stop prefetching thread."""
        self.stop_event = True
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=1.0)

    def _prefetch_loop(self):
        """Background thread for prefetching."""
        while not self.stop_event:
            try:
                batch_idx = self.current_idx
                buffer = self.buffers[self.buffer_idx]

                # Load batch into pinned memory
                for i in range(self.batch_size):
                    idx = (batch_idx * self.batch_size + i) % len(self.dataset)
                    sample = self.dataset[idx]
                    buffer[i] = torch.from_numpy(sample[0])  # x1 volume

                self.prefetch_queue.put((buffer.clone(), batch_idx), timeout=0.1)
                self.current_idx += 1
                self.buffer_idx = (self.buffer_idx + 1) % self.num_batches
            except queue.Full:
                time.sleep(0.001)
            except Exception as e:
                print(f"Prefetch error: {e}")
                break

    def get_batch(self) -> Tuple[torch.Tensor, int]:
        """Get next batch (blocks until ready)."""
        return self.prefetch_queue.get()


class MemoryMappedCache:
    """
    Memory-mapped cache for subvolumes.
    Stores pre-extracted subvolumes for fast random access.
    """
    def __init__(self, cache_dir: str, max_size_gb: float = 100.0):
        self.cache_dir = cache_dir
        self.max_size_bytes = int(max_size_gb * 1024**3)
        os.makedirs(cache_dir, exist_ok=True)

        self.metadata_path = os.path.join(cache_dir, 'metadata.json')
        self.data_path = os.path.join(cache_dir, 'data.mmap')
        self.indices_path = os.path.join(cache_dir, 'indices.mmap')

        self._mmap = None
        self._indices = None
        self.metadata = {}

    def _compute_cache_key(self, star_file: str, cube_size: int, method: str) -> str:
        """Compute unique cache key based on inputs."""
        with open(star_file, 'rb') as f:
            content = f.read()
        key_data = f"{content}{cube_size}{method}".encode()
        return hashlib.md5(key_data).hexdigest()[:16]

    def exists(self, star_file: str, cube_size: int, method: str) -> bool:
        """Check if valid cache exists."""
        if not os.path.exists(self.metadata_path):
            return False

        try:
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)

            expected_key = self._compute_cache_key(star_file, cube_size, method)
            return self.metadata.get('cache_key') == expected_key
        except:
            return False

    def create(self, star_file: str, cube_size: int, method: str,
               subvolumes: np.ndarray, indices: np.ndarray):
        """Create new memory-mapped cache."""
        n_samples = len(subvolumes)
        shape = subvolumes.shape[1:]  # (1, D, H, W)

        self.metadata = {
            'cache_key': self._compute_cache_key(star_file, cube_size, method),
            'n_samples': n_samples,
            'shape': shape,
            'dtype': str(subvolumes.dtype),
            'cube_size': cube_size,
            'method': method,
            'star_file': star_file
        }

        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)

        # Create memory-mapped arrays
        data_shape = (n_samples, *shape)
        self._mmap = np.memmap(self.data_path, dtype=subvolumes.dtype,
                               mode='w+', shape=data_shape)
        self._mmap[:] = subvolumes[:]
        self._mmap.flush()

        # Save indices
        self._indices = np.memmap(self.indices_path, dtype=np.int32,
                                  mode='w+', shape=(n_samples,))
        self._indices[:] = indices[:]
        self._indices.flush()

    def load(self):
        """Load existing cache."""
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

        n_samples = self.metadata['n_samples']
        shape = tuple(self.metadata['shape'])

        self._mmap = np.memmap(self.data_path, dtype=np.float32,
                               mode='r', shape=(n_samples, *shape))
        self._indices = np.memmap(self.indices_path, dtype=np.int32,
                                  mode='r', shape=(n_samples,))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Get subvolume and its tomo index."""
        return self._mmap[idx], self._indices[idx]

    def __len__(self) -> int:
        return self.metadata.get('n_samples', 0)

    @property
    def shape(self):
        return tuple(self.metadata.get('shape', (0,)))


def extract_subvolume_batch(args):
    """
    Extract a batch of subvolumes in parallel.
    Uses NumPy SIMD operations for normalization.
    """
    tomo_path, coords_batch, cube_size, mean, std, offset = args
    half_size = cube_size // 2

    try:
        import mrcfile
        with mrcfile.mmap(tomo_path, mode='r', permissive=True) as tomo:
            tomo_data = tomo.data

            batch_size = len(coords_batch)
            subvolumes = np.zeros((batch_size, 1, cube_size, cube_size, cube_size),
                                  dtype=np.float32)

            for i, (z, y, x) in enumerate(coords_batch):
                z_start, z_end = z - half_size, z + half_size
                y_start, y_end = y - half_size, y + half_size
                x_start, x_end = x - half_size, x + half_size

                subvolumes[i, 0] = tomo_data[z_start:z_end, y_start:y_end, x_start:x_end]

            # SIMD normalization
            subvolumes = (subvolumes - mean) / std

        return subvolumes, offset
    except Exception as e:
        print(f"Error extracting from {tomo_path}: {e}")
        return None, offset


class FastTrainSets_n2n(Train_sets_n2n):
    """
    Fast training dataset with memory-mapped cache and parallel extraction.
    """

    def __init__(self, tomo_star, method="n2n", cube_size=64,
                 input_column="rlnTomoName", split="full", noise_dir=None,
                 correct_between_tilts=False, start_bt_size=48,
                 snrfalloff=0, deconvstrength=1, highpassnyquist=0.02,
                 clip_first_peak_mode=0, bfactor=0,
                 cache_dir: str = None, max_cache_gb: float = 200.0,
                 num_workers: int = None):

        # Initialize parent but don't load data yet
        self._init_params = {
            'tomo_star': tomo_star, 'method': method, 'cube_size': cube_size,
            'input_column': input_column, 'split': split, 'noise_dir': noise_dir,
            'correct_between_tilts': correct_between_tilts, 'start_bt_size': start_bt_size,
            'snrfalloff': snrfalloff, 'deconvstrength': deconvstrength,
            'highpassnyquist': highpassnyquist, 'clip_first_peak_mode': clip_first_peak_mode,
            'bfactor': bfactor
        }

        # Detect source storage type and provide recommendations
        import starfile
        star = starfile.read(tomo_star)
        sample_path = None
        if method in ['isonet2-n2n', 'n2n'] and 'rlnTomoReconstructedTomogramHalf1' in star.columns:
            sample_path = star.iloc[0]['rlnTomoReconstructedTomogramHalf1']
        elif 'rlnTomoName' in star.columns:
            sample_path = star.iloc[0]['rlnTomoName']

        if sample_path and os.path.exists(sample_path):
            source_storage = detect_and_log_storage_type(sample_path, "Source data")

            # Recommend cache placement based on source storage
            if source_storage == StorageType.HDD and cache_dir is None:
                logging.warning("Source data is on HDD. Consider setting --cache_dir to an SSD path for better performance.")
                logging.warning("Example: --cache_dir /fast_ssd/isonet_cache")

        # Setup cache
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(tomo_star), '.isonet_cache')
        self.cache = MemoryMappedCache(cache_dir, max_size_gb=max_cache_gb)

        # Log cache location storage type
        detect_and_log_storage_type(cache_dir, "Cache")

        self.num_workers = num_workers or min(cpu_count(), 16)
        self.prefetcher = None

        # Build or load cache
        if self.cache.exists(tomo_star, cube_size, method):
            print(f"Loading cached dataset from {cache_dir}")
            self.cache.load()
            self.length = len(self.cache)
            self._load_parent_metadata()
        else:
            print(f"Building cache in {cache_dir}...")
            self._build_cache()

    def _load_parent_metadata(self):
        """Load metadata from parent class without loading data."""
        import starfile
        self.star = starfile.read(self._init_params['tomo_star'])
        self.cube_size = self._init_params['cube_size']
        self.method = self._init_params['method']

        # Load mask lists for GPU caching
        self.mw_list = []
        self.wiener_list = []
        self.CTF_list = []

        for _, row in self.star.iterrows():
            min_angle, max_angle = row['rlnTiltMin'], row['rlnTiltMax']
            tilt_step = None
            if self._init_params['correct_between_tilts']:
                tilt_step = 3

            self.mw_list.append(self._compute_missing_wedge(
                self.cube_size, min_angle, max_angle, tilt_step,
                self._init_params['start_bt_size'] / tilt_step if tilt_step else 100000
            ))

            CTF_vol, wiener_vol = self._compute_CTF_vol(row)
            self.wiener_list.append(wiener_vol)
            self.CTF_list.append(CTF_vol)

    def _build_cache(self):
        """Build cache with parallel extraction."""
        # First initialize parent to get coordinates
        super().__init__(**self._init_params)

        all_subvolumes = []
        all_indices = []

        # Process each tomogram in parallel
        print(f"Extracting subvolumes using {self.num_workers} workers...")

        for tomo_idx in range(len(self.tomo_paths_even)):
            coords = self.coords[tomo_idx]
            tomo_path = self.tomo_paths_even[tomo_idx]
            mean, std = self.mean[tomo_idx][0], self.std[tomo_idx][0]

            # Split coordinates into batches for parallel processing
            batch_size = 64  # Optimal for SIMD
            n_batches = (len(coords) + batch_size - 1) // batch_size

            tasks = []
            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, len(coords))
                tasks.append((tomo_path, coords[start:end], self.cube_size,
                             mean, std, start))

            # Parallel extraction
            with Pool(self.num_workers) as pool:
                results = pool.map(extract_subvolume_batch, tasks)

            # Combine results
            tomo_subvolumes = np.zeros((len(coords), 1, self.cube_size,
                                       self.cube_size, self.cube_size), dtype=np.float32)

            for subvols, offset in results:
                if subvols is not None:
                    batch_len = len(subvols)
                    tomo_subvolumes[offset:offset+batch_len] = subvols

            all_subvolumes.append(tomo_subvolumes)
            all_indices.extend([tomo_idx] * len(coords))

        # Concatenate all tomograms
        all_subvolumes = np.concatenate(all_subvolumes, axis=0)
        all_indices = np.array(all_indices, dtype=np.int32)

        # Create cache
        print(f"Creating cache with {len(all_subvolumes)} samples...")
        self.cache.create(
            self._init_params['tomo_star'],
            self.cube_size,
            self.method,
            all_subvolumes,
            all_indices
        )

        self.length = len(self.cache)
        print(f"Cache built: {self.length} samples")

    def __getitem__(self, idx):
        """Fast retrieval from memory-mapped cache."""
        x1_volume, tomo_idx = self.cache[idx]

        # For n2n, we need x2 as well - generate on-the-fly or cache both halves
        # Simplified: return x1 twice for now (can be extended)
        x2_volume = x1_volume.copy()

        # Return format compatible with original
        return (
            x1_volume,
            x2_volume,
            np.array([0], dtype=np.float32),  # gt placeholder
            tomo_idx,  # tomo index for mask lookup
            np.array([0], dtype=np.float32)   # noise placeholder
        )

    def __len__(self):
        return self.length

    def get_sample_shape(self):
        """Return shape of a single sample."""
        return self.cache.shape

    def start_prefetcher(self, batch_size: int, device='cuda'):
        """Start async prefetching for training."""
        self.prefetcher = AsyncPrefetcher(self, batch_size, device=device)
        self.prefetcher.start()

    def stop_prefetcher(self):
        """Stop async prefetching."""
        if self.prefetcher:
            self.prefetcher.stop()
            self.prefetcher = None


# Backward compatibility alias
TrainSetsFast = FastTrainSets_n2n