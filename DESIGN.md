# IsoNet2 Rust Implementation Design Document

## Overview

Pure Rust implementation of IsoNet2 with identical input/output behavior to the official Python implementation.

### Performance Optimization Reference (Python v2.0.2)

The Python implementation includes several optimizations that should be replicated or improved in Rust:

**Memory Optimizations:**
- **Gradient Checkpointing**: Trades compute for memory (30-50% VRAM reduction)
- **GPU Mask Caching**: Pre-transfer CTF/MW masks to GPU, eliminate CPU→GPU transfers
- **Gradient Accumulation**: Simulate larger batches with limited memory

**I/O Optimizations:**
- **Storage-Aware Access**: Auto-detect SSD/NVMe/HDD, optimize read patterns
  - HDD: Sequential coordinate sorting, full file caching to RAM
  - SSD/NVMe: Memory-mapped parallel random access
  - Network: Local disk caching
- **Memory-Mapped Cache**: Pre-extract subvolumes to `.mmap` files for fast random access
- **Async Prefetching**: Double-buffered CPU→GPU transfer overlapping with computation

**Compute Optimizations:**
- **torch.compile()**: Kernel fusion and graph optimization (10-30% speedup)
- **SIMD Normalization**: Batch subvolume normalization using AVX2/AVX512
- **Parallel Extraction**: Multi-process subvolume extraction (16 workers)

**Rust Implementation Targets:**
- Use `memmap2` for memory-mapped I/O (already planned)
- Use `rayon` for parallel extraction (already planned)
- Use `portable-simd` for batch normalization (already planned)
- Consider `burn` checkpointing support for gradient checkpointing
- Implement async data loading with `tokio` or similar

---

## Training Equivalence & Convergence Guarantees

To ensure the Rust implementation produces **bit-exact or near-exact training behavior** as the official Python implementation, the following must be strictly maintained:

### 1. Numerical Precision Requirements

**Float Precision:**
- Default: `f32` (float32) throughout - matches PyTorch default
- Mixed precision training optional but must follow same casting rules
- CTF/Mask computations in `f64` then cast to `f32` for storage (matches Python)

**Critical Numerical Operations:**
```rust
// FFT operations must match numpy.fft exactly
// Use rustfft with same normalization mode ('ortho' or 'backward')
pub const FFT_NORM: FftNormalization = FftNormalization::Ortho;  // Matches PyTorch

// Random number generation must match PyTorch/NumPy
// Use same RNG algorithm: PCG64 or MT19937 with identical seeding
pub type RngType = Pcg64Mcg;  // Matches NumPy's default
```

### 2. Network Architecture Equivalence

**Weight Initialization:**
- Use identical initialization schemes as PyTorch defaults
- Conv3d: Kaiming uniform (fan_in, a=√5)
- BatchNorm: γ=1, β=0
- Document all init parameters for verification

**Forward Pass:**
- Conv3d: Same padding, stride, dilation
- BatchNorm3d: Same ε (1e-5), momentum (0.1)
- LeakyReLU: Same negative_slope (0.01)
- Residual connections: `output + input` (not fused)

**Backward Pass:**
- Gradient checkpointing: Identical recompute strategy
- Gradient accumulation: Same accumulation order and normalization

### 3. Loss Function Equivalence

**L2 Loss (MSE):**
```rust
// Must match: nn.MSELoss(reduction='mean')
// PyTorch computes mean over ALL elements
loss = ((pred - target).powi(2)).mean()
```

**Huber Loss:**
```rust
// Must match: nn.HuberLoss(delta=1.0, reduction='mean')
// Same delta threshold, same behavior at boundary
```

**Masked Loss (for missing wedge):**
```rust
// Split loss into inside/outside missing wedge
// Same weighting: inside_loss + mw_weight * outside_loss
// Same normalization by mask volume
```

### 4. Optimizer Equivalence

**AdamW:**
```rust
// Must match: torch.optim.AdamW(
//   lr=3e-4, betas=(0.9, 0.999), eps=1e-8,
//   weight_decay=0.01, amsgrad=False
// )
// Same parameter groups, same weight decay application
```

**Learning Rate Scheduler:**
```rust
// Must match: CosineAnnealingLR(T_max=10, eta_min=3e-4)
// Same cosine curve, same step timing
```

### 5. Data Pipeline Equivalence

**Normalization:**
```rust
// Percentile normalization (4% lower, 96% upper)
// Same quantile calculation method
lower = tensor.quantile(0.04)
upper = tensor.quantile(0.96)
normalized = (tensor - lower) / (upper - lower)
```

**Data Augmentation:**
- Random rotations: Same rotation axes, same probability (0.2)
- Phase flipping: Same sign application
- Noise addition: Same distribution, same amplitude scaling

**Subvolume Extraction:**
- Same coordinate sampling (within mask bounds)
- Same cube_size constraints (multiple of 16)
- Same padding/masking at boundaries

### 6. CTF & Missing Wedge Equivalence

**CTF Computation:**
```rust
// Exact same formula as Python
// λ = 12.2643247 / sqrt(voltage * (1 + voltage * 0.978466e-6))
// χ = π/2 * (λ³ * Cs * k⁴ + 2λ * Δf * k²)
// CTF = amplitude * cos(χ) - sqrt(1-amplitude²) * sin(χ)
// Same frequency grid generation (0 to Nyquist)
```

**Missing Wedge Mask:**
- Same spherical mask generation
- Same tilt angle interpolation
- Same wedge softening at edges

### 7. Training Loop Semantics

**Batch Processing:**
```rust
// Same order of operations:
1. Load batch (x1, x2, masks)
2. Apply CTF/masks to input
3. Forward pass
4. Compute loss (normalized by acc_batches)
5. Backward pass
6. If (batch_idx + 1) % acc_batches == 0:
     optimizer.step()
     optimizer.zero_grad()
```

**Gradient Accumulation:**
- Accumulate gradients: `grad += new_grad / acc_batches`
- Step only every `acc_batches` batches
- Same gradient scaling for mixed precision

### 8. Verification Strategy

**Unit Tests:**
```rust
// Compare forward pass output
let python_output = load_npy("test_data/unet_forward.npy");
let rust_output = model.forward(&input);
assert_allclose(rust_output, python_output, rtol=1e-5, atol=1e-6);

// Compare gradients
let python_grads = load_npy("test_data/unet_grads.npy");
let rust_grads = compute_grads(&model, &input, &target);
assert_allclose(rust_grads, python_grads, rtol=1e-4, atol=1e-5);
```

**Integration Tests:**
- Train 5 epochs on small dataset
- Compare loss curves (should be < 1% relative difference)
- Compare final model weights (cosine similarity > 0.999)

**Regression Tests:**
- Save intermediate activations from Python
- Compare layer-by-layer outputs in Rust
- Identify and fix divergence points

### 9. Known Sources of Divergence

**Acceptable (minor):**
- Different FFT libraries (rustfft vs cuFFT): < 1e-6 relative error
- Different BLAS libraries: < 1e-5 relative error
- Floating point reordering: < 1e-6 relative error

**Must Fix:**
- Different random seeds (must use same seed)
- Different initialization (must match exactly)
- Different loss computation (must match reduction)
- Different optimizer step timing

**Test Command:**
```bash
# Compare Python vs Rust training
cargo test --test convergence -- --nocapture
# Should pass: loss curves match within tolerance
```

---

**Tech Stack:**

- `burn` - Deep learning framework (training & inference)
- `mrc` - MRC file I/O
- `emstar` - STAR file format handling
- `rayon` - Data parallelism
- `clap` v4 - CLI argument parsing
- `portable-simd` / `std::simd` - SIMD operations
- `memmap2` - Zero-copy memory-mapped file I/O

---

## Workspace Architecture

```
rs/
├── Cargo.toml                    # Workspace manifest
├── src/
│   ├── cli/              # CLI binary (clap v4)
│   ├── core/             # Core types, traits, constants
│   ├── io/               # MRC/STAR file I/O (zero-copy)
│   ├── fft/              # FFT utilities (real-to-complex)
│   ├── ctf/              # CTF 1D/2D/3D computation (SIMD)
│   ├── wedge/            # Missing wedge mask generation (SIMD)
│   ├── models/           # Neural network architectures (burn)
│   ├── train/            # Training logic, losses, optimizers
│   ├── data/             # Data loading, augmentation, preprocessing
│   ├── mask/             # Mask generation (std/max filters)
│   └── deconv/           # CTF deconvolution
├── benches/                      # Criterion benchmarks
└── tests/                        # Integration tests against Python impl
```

---

## Crate Details

### 1. `core`

**Purpose:** Type-safe physical units, core data structures, and shared traits.

```rust
// Physical units (newtype pattern)
pub struct Angstrom(f32);
pub struct Kilovolt(f32);
pub struct Millimeter(f32);
pub struct Degree(f32);
pub struct DefocusUm(f32);  // Defocus in micrometers

// Core data structures
pub struct TomogramMeta {
    pub path: PathBuf,
    pub pixel_size: Angstrom,
    pub shape: [usize; 3],  // Z, Y, X
    pub voltage: Kilovolt,
    pub cs: Millimeter,     // Spherical aberration
    pub amplitude_contrast: f32,
    pub defocus: Angstrom,
    pub tilt_range: (Degree, Degree),  // (min, max)
}

pub struct CtfParams {
    pub angpix: Angstrom,
    pub voltage: Kilovolt,
    pub cs: Millimeter,
    pub defocus: DefocusUm,
    pub amplitude: f32,
    pub phase_shift: Degree,
    pub bfactor: f32,
}

pub struct SubVolume {
    pub data: Array4<f32>,  // [B, D, H, W] or [B, C, D, H, W]
    pub coords: [(usize, usize, usize)],  // Z, Y, X centers
    pub tomo_idx: usize,
}

// Core traits
pub trait Normalizable {
    fn normalize_percentile(&mut self, percentile: f32);
    fn normalize_mean_std(&mut self);
    fn normalize_mean_std_with(&mut self, mean: f32, std: f32);
}

pub trait FourierTransformable {
    fn fftn(&self) -> Array3<Complex32>;
    fn ifftn(input: &Array3<Complex32>) -> Array3<f32>;
    fn apply_fourier_mask(&mut self, mask: &Array3<f32>);
}

pub trait Rotatable3D {
    fn rotate_90(&self, axis: Axis, k: i32) -> Self;
    fn rotate_axis_angle(&self, axis: [f32; 3], angle: Degree) -> Self;
}
```

---

### 2. `io`

**Purpose:** Zero-copy MRC and STAR file I/O.

```rust
// MRC I/O with memory mapping
pub struct MrcFile {
    mmap: Mmap,
    header: MrcHeader,
    data_offset: usize,
}

impl MrcFile {
    /// Open with memory mapping (zero-copy read)
    pub fn mmap(path: &Path) -> Result<Self>;
    
    /// Read header only
    pub fn header(path: &Path) -> Result<MrcHeader>;
    
    /// Get data view without copying
    pub fn data_view(&self) -> ArrayView3<f32>;
    
    /// Read specific slice (for subvolume extraction)
    pub fn read_slice(&self, z: Range, y: Range, x: Range) -> Array3<f32>;
    
    /// Write MRC file
    pub fn write(path: &Path, data: &Array3<f32>, voxel_size: Angstrom) -> Result<()>;
}

pub struct MrcHeader {
    pub nx: i32, pub ny: i32, pub nz: i32,
    pub mode: i32,
    pub cella: [f32; 3],  // Angstroms
    pub origin: [f32; 3],
    // ... other fields
}

// STAR file handling via emstar
pub use emstar::{StarFile, StarRecord, StarLoop};

pub struct IsoNetStar {
    inner: StarFile,
}

impl IsoNetStar {
    pub fn read(path: &Path) -> Result<Self>;
    pub fn write(&self, path: &Path) -> Result<()>;
    
    // Convenience accessors for IsoNet2 specific columns
    pub fn tomogram_names(&self) -> Vec<&str>;
    pub fn pixel_sizes(&self) -> Vec<Angstrom>;
    pub fn defoci(&self) -> Vec<Angstrom>;
    pub fn tilt_ranges(&self) -> Vec<(Degree, Degree)>;
    
    // Update column values
    pub fn update_column(&mut self, column: &str, values: &[String]);
}

// Constants for column names
pub const COL_TOMO_NAME: &str = "rlnTomoName";
pub const COL_TOMO_HALF1: &str = "rlnTomoReconstructedTomogramHalf1";
pub const COL_TOMO_HALF2: &str = "rlnTomoReconstructedTomogramHalf2";
pub const COL_PIXEL_SIZE: &str = "rlnPixelSize";
pub const COL_DEFOCUS: &str = "rlnDefocus";
pub const COL_VOLTAGE: &str = "rlnVoltage";
pub const COL_CS: &str = "rlnSphericalAberration";
pub const COL_AC: &str = "rlnAmplitudeContrast";
pub const COL_TILT_MIN: &str = "rlnTiltMin";
pub const COL_TILT_MAX: &str = "rlnTiltMax";
pub const COL_MASK_NAME: &str = "rlnMaskName";
pub const COL_MASK_BOUNDARY: &str = "rlnMaskBoundary";
pub const COL_DECONV_TOMO: &str = "rlnDeconvTomoName";
pub const COL_DENOISED_TOMO: &str = "rlnDenoisedTomoName";
pub const COL_CORRECTED_TOMO: &str = "rlnCorrectedTomoName";
pub const COL_NUM_SUBTOMO: &str = "rlnNumberSubtomo";
```

---

### 3. `fft`

**Purpose:** FFT wrappers with consistent behavior.

```rust
use rustfft::{FftPlanner, num_complex::Complex32};
use ndarray::{Array3, ArrayView3};

pub struct FftEngine {
    planner: FftPlanner<f32>,
}

impl FftEngine {
    /// 3D FFT with shift (matches numpy.fft.fftn with fftshift)
    pub fn fftn_shifted(&mut self, input: &ArrayView3<f32>) -> Array3<Complex32>;
    
    /// 3D IFFT with shift
    pub fn ifftn_shifted(&mut self, input: &ArrayView3<Complex32>) -> Array3<f32>;
    
    /// Apply filter in Fourier domain
    pub fn apply_filter(&self, input: &Array3<f32>, filter: &Array3<f32>) -> Array3<f32>;
    
    /// Real input FFT (R2C)
    pub fn rfftn(&mut self, input: &ArrayView3<f32>) -> Array3<Complex32>;
    
    /// C2R IFFT
    pub fn irfftn(&mut self, input: &ArrayView3<Complex32>, shape: [usize; 3]) -> Array3<f32>;
}

// Thread-local FFT engines for parallel processing
thread_local! {
    static FFT_ENGINE: RefCell<FftEngine> = RefCell::new(FftEngine::new());
}
```

---

### 4. `ctf` (SIMD-optimized)

**Purpose:** CTF 1D/2D/3D computation with SIMD acceleration.

```rust
use std::simd::{f32x8, SimdFloat};

pub struct CtfCalculator;

impl CtfCalculator {
    /// 1D CTF - SIMD vectorized
    pub fn ctf_1d_simd(params: &CtfParams, length: usize) -> Vec<f32> {
        // Process 8 elements at a time using f32x8
        // λ = 12.2643247 / sqrt(voltage * (1 + voltage * 0.978466e-6))
        // χ = π/2 * (λ³ * Cs * k⁴ + 2λ * Δf * k²) - phase_shift
        // CTF = A * cos(χ) - sqrt(1-A²) * sin(χ)
    }
    
    /// 2D CTF (polar interpolation)
    pub fn ctf_2d(params: &CtfParams, length: usize) -> Array2<f32>;
    
    /// 3D CTF (spherical interpolation)
    pub fn ctf_3d(params: &CtfParams, length: usize) -> Array3<f32>;
    
    /// Wiener filter 1D
    pub fn wiener_1d(
        params: &CtfParams,
        snr_falloff: f32,
        deconv_strength: f32,
        highpass_nyquist: f32,
        phase_flipped: bool,
        length: usize,
    ) -> Vec<f32>;
    
    /// Wiener filter 3D
    pub fn wiener_3d(
        params: &CtfParams,
        snr_falloff: f32,
        deconv_strength: f32,
        highpass_nyquist: f32,
        phase_flipped: bool,
        length: usize,
    ) -> Array3<f32>;
    
    /// Phase flip filter (sign of CTF)
    pub fn phase_flip_filter(params: &CtfParams, length: usize) -> Array3<f32>;
    
    /// B-factor application
    pub fn apply_bfactor(ctf: &mut [f32], bfactor: f32, pixel_size: f32);
}

// SIMD-accelerated wavelength calculation
#[inline(always)]
pub fn electron_wavelength_simd(voltage: f32x8) -> f32x8 {
    let numerator = f32x8::splat(12.2643247);
    let correction = f32x8::splat(0.978466e-6);
    numerator / (voltage * (f32x8::splat(1.0) + voltage * correction)).sqrt()
}
```

---

### 5. `wedge` (SIMD-optimized)

**Purpose:** Missing wedge mask generation.

```rust
pub struct MissingWedge;

impl MissingWedge {
    /// Generate 2D missing wedge mask
    /// missing_angle: [90 + tilt_min, 90 - tilt_max]
    pub fn mask_2d(dim: usize, missing_angle: [Degree; 2]) -> Array2<f32>;
    
    /// Generate 3D missing wedge mask (repeated 2D)
    pub fn mask_3d(dim: usize, missing_angle: [Degree; 2]) -> Array3<f32>;
    
    /// Generate 3D missing wedge with tilt-step correction
    pub fn mask_3d_with_tilt_step(
        dim: usize,
        missing_angle: [Degree; 2],
        tilt_step: f32,
        start_dim: usize,
    ) -> Array3<f32>;
    
    /// Spherical mask (for frequency domain)
    pub fn spherical_mask(dim: usize) -> Array3<f32>;
    
    /// Apply missing wedge to volume in Fourier domain
    pub fn apply_wedge(volume: &mut Array3<f32>, wedge: &Array3<f32>);
    
    /// SIMD-accelerated mask generation
    pub fn mask_2d_simd(dim: usize, missing_angle: [f32; 2]) -> Array2<f32>;
}

// Pre-computed missing wedges cache (tilt_step → mask)
pub struct WedgeCache {
    cache: DashMap<(usize, f32, [Degree; 2]), Array3<f32>>,
}
```

---

### 6. `models` (Burn)

**Purpose:** Neural network architectures.

```rust
use burn::{
    module::Module,
    nn::{conv::{Conv3d, Conv3dConfig}, BatchNorm, LeakyRelu},
    tensor::{backend::Backend, Tensor},
};

// UNet Architecture
#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    encoder: EncoderBlock<B>,
    decoder: DecoderBlock<B>,
    final_conv: Conv3d<B>,
    add_residual: bool,
}

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    convs: Vec<Conv3d<B>>,
    bn: Vec<BatchNorm<B, 3>>,
    activation: LeakyRelu,
}

#[derive(Module, Debug)]
pub struct EncoderBlock<B: Backend> {
    first_conv: Conv3d<B>,
    conv_stacks: Vec<ConvBlock<B>>,
    stride_convs: Vec<ConvBlock<B>>,
    bottleneck: ConvBlock<B>,
}

#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    deconvs: Vec<ConvTranspose3d<B>>,
    activations: Vec<LeakyRelu>,
    conv_stacks: Vec<ConvBlock<B>>,
}

// Architecture configurations
pub struct UNetConfig {
    pub filter_base: usize,  // 16 (small), 32 (medium), 64 (large)
    pub depth: usize,        // 4
    pub n_conv: usize,       // 3
    pub add_residual: bool,  // true
}

impl UNetConfig {
    pub fn small() -> Self { Self { filter_base: 16, .. } }
    pub fn medium() -> Self { Self { filter_base: 32, .. } }
    pub fn large() -> Self { Self { filter_base: 64, .. } }
}

// Model wrapper for save/load
pub struct Network<B: Backend> {
    model: UNet<B>,
    arch: String,
    method: String,
    cube_size: usize,
    ctf_mode: Option<String>,
    do_phaseflip_input: bool,
    metrics: Metrics,
}

impl<B: Backend> Network<B> {
    /// Load from checkpoint (compatible with PyTorch .pt)
    pub fn from_checkpoint(path: &Path) -> Result<Self>;
    
    /// Save checkpoint
    pub fn save_checkpoint(&self, path: &Path) -> Result<()>;
    
    /// Export to Burn's format for inference
    pub fn export_for_inference(&self, path: &Path) -> Result<()>;
}

// SCUNet (if needed - skip for initial implementation)
// pub struct SCUNet<B: Backend> { ... }
```

---

### 7. `train` (Burn)

**Purpose:** Training logic, losses, optimizers.

```rust
use burn::{
    optim::{AdamW, AdamWConfig},
    lr_scheduler::{CosineAnnealingLR, CosineAnnealingLRConfig},
    tensor::backend::AutodiffBackend,
    train::{TrainStep, ValidStep},
};

// Training configuration
pub struct TrainingConfig {
    pub method: Method,  // N2N, IsoNet2, IsoNet2N2N
    pub arch: Architecture,
    pub cube_size: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub steps_per_epoch: usize,
    pub learning_rate: f64,
    pub learning_rate_min: f64,
    pub loss_func: LossFunction,
    pub mixed_precision: bool,
    pub ctf_mode: CtfMode,
    pub mw_weight: f32,  // Missing wedge loss weight
    pub bfactor: f32,
    pub clip_first_peak_mode: u8,
    pub random_rot_weight: f32,
}

pub enum Method {
    N2N,
    IsoNet2,
    IsoNet2N2N,
}

pub enum Architecture {
    UnetSmall,
    UnetMedium,
    UnetLarge,
}

pub enum LossFunction {
    L2,
    L1,
    Huber { delta: f32 },
}

pub enum CtfMode {
    None,
    PhaseOnly,
    Network,
    Wiener,
}

// Loss functions
pub struct MaskedLoss;

impl MaskedLoss {
    /// Compute inside/outside missing wedge loss
    /// Returns (outside_loss, inside_loss)
    pub fn compute<B: Backend>(
        pred: &Tensor<B, 5>,  // [B, C, D, H, W]
        target: &Tensor<B, 5>,
        rotated_mw: &Tensor<B, 5>,  // Missing wedge mask rotated
        original_mw: &Tensor<B, 5>,
        loss_fn: &dyn Fn(&Tensor<B, 5>, &Tensor<B, 5>) -> Tensor<B, 1>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>);
}

// Training step
pub struct TrainStepConfig {
    pub apply_mw_x1: bool,
    pub do_phaseflip: bool,
    pub noise_level: f32,
}

// Trainer
pub struct Trainer<B: AutodiffBackend> {
    model: UNet<B>,
    optimizer: AdamW<B>,
    scheduler: CosineAnnealingLR<B>,
    config: TrainingConfig,
    metrics: Metrics,
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(model: UNet<B>, config: TrainingConfig) -> Self;
    
    pub fn train_epoch<D: Dataset>(&mut self, dataloader: &DataLoader<D>) -> Metrics;
    
    /// Train with distributed data parallel
    pub fn train_ddp<D: Dataset>(
        &mut self,
        world_size: usize,
        rank: usize,
        dataloader: &DataLoader<D>,
    ) -> Metrics;
}

// Metrics tracking
#[derive(Default, Clone)]
pub struct Metrics {
    pub average_loss: Vec<f32>,
    pub inside_loss: Vec<f32>,
    pub outside_loss: Vec<f32>,
}

impl Metrics {
    pub fn plot(&self, path: &Path) -> Result<()>;
    pub fn to_json(&self) -> String;
}
```

---

### 8. `data`

**Purpose:** Data loading, subvolume extraction, augmentation.

```rust
use rayon::prelude::*;

// Dataset trait
pub trait Dataset: Send + Sync {
    type Item;
    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> Self::Item;
}

// Noise2Noise dataset
pub struct N2NDataset {
    star: IsoNetStar,
    tomograms_even: Vec<Mmap>,  // Memory-mapped even tomograms
    tomograms_odd: Vec<Mmap>,   // Memory-mapped odd tomograms
    masks: Vec<Option<Array3<u8>>>,
    coords: Vec<Array2<usize>>, // [N, 3] coordinates per tomogram
    cube_size: usize,
    mw_masks: Vec<Array3<f32>>, // Pre-computed missing wedge masks
    ctf_volumes: Vec<Array3<f32>>, // Pre-computed CTF
    wiener_volumes: Vec<Array3<f32>>, // Pre-computed Wiener filters
}

impl N2NDataset {
    pub fn new(
        star_path: &Path,
        cube_size: usize,
        method: Method,
        ctf_params: &CtfConfig,
    ) -> Result<Self>;
    
    /// Generate random coordinates within mask
    fn generate_coords(
        mask: &Array3<u8>,
        n_samples: usize,
        cube_size: usize,
        split: Split,
    ) -> Array2<usize>;
    
    /// Extract subvolume at coordinates
    fn extract_subvolume(&self, tomo_idx: usize, coord: [usize; 3]) -> Array3<f32>;
    
    /// Normalize subvolume
    fn normalize(&self, subvol: &Array3<f32>, tomo_idx: usize, is_even: bool) -> Array3<f32>;
}

impl Dataset for N2NDataset {
    type Item = (Array4<f32>, Array4<f32>, Array4<f32>, Array4<f32>, Array4<f32>, Array4<f32>);
    // (x1, x2, ground_truth, mw_mask, ctf, wiener, noise)
    
    fn get(&self, idx: usize) -> Self::Item {
        // Multi-threaded extraction using rayon
        // Returns batch of subvolumes
    }
}

// DataLoader with parallel prefetching
pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    num_workers: usize,
    shuffle: bool,
}

impl<D: Dataset> DataLoader<D> {
    pub fn iter(&self) -> DataLoaderIter<D>;
}

// 3D rotation utilities
pub struct Rotation3D;

impl Rotation3D {
    /// 90-degree rotations (axis, k)
    pub fn rotate_90(volume: &Array3<f32>, axis: Axis, k: i32) -> Array3<f32>;
    
    /// Random rotation (axis + angle)
    pub fn random_rotation(volume: &Array3<f32>) -> Array3<f32>;
    
    /// Random axis and angle sampling
    pub fn sample_axis_angle() -> ([f32; 3], Degree);
}

pub enum Split {
    Full,
    Top,
    Bottom,
}
```

---

### 9. `mask` (SIMD-optimized)

**Purpose:** Mask generation using local statistics.

```rust
use std::simd::{f32x8, u8x64};

pub struct MaskGenerator;

impl MaskGenerator {
    /// Generate mask from tomogram
    pub fn generate(
        tomo: &Array3<f32>,
        side: usize,  // patch size (default 4)
        density_percentage: f32,
        std_percentage: f32,
        surface_crop: Option<f32>,  // z_crop/2
    ) -> Array3<u8>;
    
    /// Max filter mask (density-based)
    fn maxmask_simd(
        tomo: &Array3<f32>,
        side: usize,
        percentile: f32,
    ) -> Array3<u8>;
    
    /// Std filter mask (texture-based)
    fn stdmask_simd(
        tomo: &Array3<f32>,
        side: usize,
        threshold: f32,
    ) -> Array3<u8>;
    
    /// Combine masks
    fn combine_masks(m1: &Array3<u8>, m2: &Array3<u8>) -> Array3<u8>;
    
    /// Apply boundary mask
    fn apply_boundary(mask: &mut Array3<u8>, boundary: &Boundary);
}

// SIMD-accelerated local max/std computation
pub fn local_max_simd(window: &[f32]) -> f32;
pub fn local_std_simd(window: &[f32]) -> f32;
```

---

### 10. `deconv`

**Purpose:** CTF deconvolution.

```rust
pub struct Deconvolver;

impl Deconvolver {
    /// Deconvolve single tomogram
    pub fn deconvolve(
        input: &Array3<f32>,
        params: &CtfParams,
        snr_falloff: f32,
        deconv_strength: f32,
        highpass_nyquist: f32,
        phase_flipped: bool,
    ) -> Array3<f32>;
    
    /// Chunked deconvolution for large tomograms
    pub fn deconvolve_chunked(
        input_path: &Path,
        output_path: &Path,
        params: &CtfParams,
        chunk_size: usize,
        overlap: f32,
        ncpu: usize,
    ) -> Result<()>;
}

// Chunk management
pub struct Chunks {
    chunk_size: usize,
    overlap: f32,
    shape: [usize; 3],
    n_chunks: [usize; 3],
}

impl Chunks {
    pub fn new(tomo_shape: [usize; 3], chunk_size: usize, overlap: f32) -> Self;
    pub fn iter(&self) -> ChunkIter;
    pub fn restore(&self, chunks: &[Array3<f32>]) -> Array3<f32>;
}
```

---

### 11. `cli`

**Purpose:** Command-line interface.

```rust
use clap::{Parser, Subcommand, Args};

#[derive(Parser)]
#[command(name = "isonet2")]
#[command(about = "IsoNet2: Deep learning for cryo-ET missing wedge correction")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate STAR file from tomogram folders
    PrepareStar(PrepareStarArgs),
    
    /// Train denoising model (noise2noise)
    Denoise(DenoiseArgs),
    
    /// CTF deconvolution preprocessing
    Deconv(DeconvArgs),
    
    /// Generate masks for tomograms
    MakeMask(MakeMaskArgs),
    
    /// Train missing wedge correction model
    Refine(RefineArgs),
    
    /// Apply trained model to tomograms
    Predict(PredictArgs),
    
    /// Check installation and GPU performance
    Check,
}

#[derive(Args)]
struct PrepareStarArgs {
    #[arg(long)]
    full: Option<PathBuf>,
    
    #[arg(long)]
    even: Option<PathBuf>,
    
    #[arg(long)]
    odd: Option<PathBuf>,
    
    #[arg(long, default_value = "tomograms.star")]
    star_name: PathBuf,
    
    #[arg(long, default_value = "auto")]
    pixel_size: String,
    
    #[arg(long, value_delimiter = ',')]
    defocus: Vec<f32>,
    
    #[arg(long, default_value = "2.7")]
    cs: f32,
    
    #[arg(long, default_value = "300")]
    voltage: f32,
    
    #[arg(long, default_value = "0.1")]
    ac: f32,
    
    #[arg(long, default_value = "-60")]
    tilt_min: f32,
    
    #[arg(long, default_value = "60")]
    tilt_max: f32,
    
    #[arg(long)]
    create_average: bool,
}

#[derive(Args)]
struct DenoiseArgs {
    star_file: PathBuf,
    
    #[arg(long, default_value = "denoise")]
    output_dir: PathBuf,
    
    #[arg(long)]
    gpu_id: Option<String>,
    
    #[arg(long, default_value = "16")]
    ncpus: usize,
    
    #[arg(long, default_value = "unet-medium")]
    arch: String,
    
    #[arg(long)]
    pretrained_model: Option<PathBuf>,
    
    #[arg(long, default_value = "96")]
    cube_size: usize,
    
    #[arg(long, default_value = "50")]
    epochs: usize,
    
    #[arg(long, default_value = "auto")]
    batch_size: String,
    
    #[arg(long, default_value = "L2")]
    loss_func: String,
    
    #[arg(long, default_value = "10")]
    save_interval: usize,
    
    #[arg(long, default_value = "3e-4")]
    learning_rate: f64,
    
    #[arg(long, default_value = "3e-4")]
    learning_rate_min: f64,
    
    #[arg(long, default_value = "true")]
    mixed_precision: bool,
    
    #[arg(long, default_value = "None")]
    ctf_mode: String,
    
    #[arg(long, default_value = "0")]
    bfactor: f32,
    
    #[arg(long, default_value = "1")]
    clip_first_peak_mode: u8,
    
    #[arg(long, default_value = "true")]
    with_preview: bool,
}

#[derive(Args)]
struct DeconvArgs {
    star_file: PathBuf,
    
    #[arg(long, default_value = "./deconv")]
    output_dir: PathBuf,
    
    #[arg(long, default_value = "rlnTomoName")]
    input_column: String,
    
    #[arg(long, default_value = "1.0")]
    snrfalloff: f32,
    
    #[arg(long, default_value = "1.0")]
    deconvstrength: f32,
    
    #[arg(long, default_value = "0.02")]
    highpassnyquist: f32,
    
    #[arg(long)]
    chunk_size: Option<usize>,
    
    #[arg(long, default_value = "0.25")]
    overlap_rate: f32,
    
    #[arg(long, default_value = "4")]
    ncpus: usize,
    
    #[arg(long)]
    phaseflipped: bool,
    
    #[arg(long)]
    tomo_idx: Option<String>,
}

#[derive(Args)]
struct MakeMaskArgs {
    star_file: PathBuf,
    
    #[arg(long, default_value = "mask")]
    output_dir: PathBuf,
    
    #[arg(long, default_value = "rlnDeconvTomoName")]
    input_column: String,
    
    #[arg(long, default_value = "4")]
    patch_size: usize,
    
    #[arg(long, default_value = "50")]
    density_percentage: f32,
    
    #[arg(long, default_value = "50")]
    std_percentage: f32,
    
    #[arg(long, default_value = "0.2")]
    z_crop: f32,
    
    #[arg(long)]
    tomo_idx: Option<String>,
}

#[derive(Args)]
struct RefineArgs {
    star_file: PathBuf,
    
    #[arg(long, default_value = "isonet_maps")]
    output_dir: PathBuf,
    
    #[arg(long)]
    gpu_id: Option<String>,
    
    #[arg(long, default_value = "16")]
    ncpus: usize,
    
    #[arg(long, default_value = "auto")]
    method: String,  // isonet2, n2n
    
    #[arg(long, default_value = "unet-medium")]
    arch: String,
    
    #[arg(long)]
    pretrained_model: Option<PathBuf>,
    
    #[arg(long, default_value = "96")]
    cube_size: usize,
    
    #[arg(long, default_value = "50")]
    epochs: usize,
    
    #[arg(long, default_value = "rlnDeconvTomoName")]
    input_column: String,
    
    #[arg(long, default_value = "auto")]
    batch_size: String,
    
    #[arg(long, default_value = "L2")]
    loss_func: String,
    
    #[arg(long, default_value = "10")]
    save_interval: usize,
    
    #[arg(long, default_value = "-1")]
    mw_weight: f32,
    
    #[arg(long, default_value = "true")]
    apply_mw_x1: bool,
    
    #[arg(long, default_value = "true")]
    mixed_precision: bool,
    
    #[arg(long, default_value = "None")]
    ctf_mode: String,
    
    #[arg(long, default_value = "0")]
    bfactor: f32,
    
    #[arg(long, default_value = "true")]
    with_preview: bool,
}

#[derive(Args)]
struct PredictArgs {
    star_file: PathBuf,
    model: PathBuf,
    
    #[arg(long, default_value = "./corrected_tomos")]
    output_dir: PathBuf,
    
    #[arg(long)]
    gpu_id: Option<String>,
    
    #[arg(long, default_value = "rlnDeconvTomoName")]
    input_column: String,
    
    #[arg(long, default_value = "true")]
    apply_mw_x1: bool,
    
    #[arg(long)]
    is_ctf_flipped: bool,
    
    #[arg(long, default_value = "1.5")]
    padding_factor: f32,
    
    #[arg(long)]
    tomo_idx: Option<String>,
    
    #[arg(long)]
    output_prefix: Option<String>,
}
```

---

## SIMD Optimization Strategy

### 1. CTF Calculation (`ctf`)

```rust
// Vectorized 1D CTF using f32x8
pub fn ctf_1d_simd(params: &CtfParams, length: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; length];
    let chunk_size = 8;
    
    // Constants as SIMD vectors
    let lambda = f32x8::splat(electron_wavelength(params.voltage));
    let lambda2 = lambda * f32x8::splat(2.0);
    let lambda3_cs = lambda * lambda * lambda * f32x8::splat(params.cs.0 * 1e-3);
    let defocus = f32x8::splat(-params.defocus.0 * 1e-6);
    
    result.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(i, chunk)| {
            let idx = i * chunk_size;
            let k2 = compute_k2_simd(idx, length, params.angpix.0);
            
            // χ = π/2 * (λ³ * Cs * k⁴ + 2λ * Δf * k²)
            let chi = f32x8::splat(PI / 2.0) * 
                      (lambda3_cs * k2 * k2 + lambda2 * defocus * k2);
            
            // CTF = -√(1-A²) * sin(χ) + A * cos(χ)
            let amplitude = f32x8::splat(params.amplitude);
            let sin_chi = chi.sin();
            let cos_chi = chi.cos();
            
            let ctf = sin_chi * -f32x8::splat((1.0 - params.amplitude.powi(2)).sqrt())
                    + cos_chi * amplitude;
            
            // Copy to output
            ctf.copy_to_slice(chunk);
        });
    
    result
}
```

### 2. Missing Wedge Generation (`wedge`)

```rust
// SIMD-accelerated 2D wedge mask
pub fn mask_2d_simd(dim: usize, missing_angle: [f32; 2]) -> Array2<f32> {
    let mut mask = Array2::zeros((dim, dim));
    let center = dim as f32 / 2.0;
    let missing_rad = [missing_angle[0].to_radians(), missing_angle[1].to_radians()];
    
    mask.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let y = f32x8::splat(i as f32 - center);
            
            for (j, cell) in row.iter_mut().enumerate().step_by(8) {
                let x = (j..j+8).map(|k| k as f32 - center).collect::<Vec<_>>();
                let x_vec = f32x8::from_slice(&x);
                
                // θ = atan2(y, x)
                let theta = y.atan2(x_vec).abs();
                
                // Check if within missing wedge
                let in_wedge = theta.simd_lt(f32x8::splat(missing_rad[0]))
                             | theta.simd_lt(f32x8::splat(missing_rad[1]));
                
                // Store results
                // ...
            }
        });
    
    mask
}
```

### 3. Mask Generation (`mask`)

```rust
// SIMD local statistics for mask generation
pub fn local_max_std_simd(
    tomo: &ArrayView3<f32>,
    side: usize,
) -> (Array3<f32>, Array3<f32>) {
    // Use rayon for parallel slice processing
    // Within each slice, use SIMD for local window aggregation
    todo!()
}
```

---

## Zero-Copy I/O Strategy

### MRC Memory Mapping

```rust
use memmap2::Mmap;

pub struct MrcMmap {
    mmap: Mmap,
    header: MrcHeader,
    shape: [usize; 3],
    dtype: MrcDtype,
}

impl MrcMmap {
    /// Zero-copy view of data
    pub fn as_slice(&self) -> &[f32] {
        let offset = 1024 + self.header.extended_bytes;
        let data = &self.mmap[offset..];
        bytemuck::cast_slice(data)
    }
    
    /// Get ndarray view without copying
    pub fn as_array(&self) -> ArrayView3<f32> {
        ArrayView3::from_shape(self.shape, self.as_slice())
            .expect("Shape mismatch")
    }
    
    /// Read subvolume (only this copies)
    pub fn read_subvolume(
        &self,
        z: Range<usize>,
        y: Range<usize>,
        x: Range<usize>,
    ) -> Array3<f32> {
        // Seek and read only required data
        // ...
    }
}
```

### Parallel I/O with Rayon

```rust
// Parallel tomogram loading
pub fn load_tomograms_parallel(
    paths: &[PathBuf],
) -> Vec<MrcMmap> {
    paths.par_iter()
        .map(|p| MrcMmap::open(p).expect("Failed to open MRC"))
        .collect()
}
```

---

## Testing Strategy

### 1. Unit Tests (per crate)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ctf_1d_matches_python() {
        let params = CtfParams { ... };
        let rust_result = CtfCalculator::ctf_1d(&params, 256);
        let python_result = load_numpy_array("ctf_1d_ref.npy");
        assert_array_near(&rust_result, &python_result, 1e-5);
    }
    
    #[test]
    fn test_missing_wedge_matches_python() {
        let rust_mask = MissingWedge::mask_2d(128, [Degree(30.0), Degree(30.0)]);
        let python_mask = load_numpy_array("mw_2d_ref.npy");
        assert_array_near(&rust_mask, &python_mask, 1e-5);
    }
}
```

### 2. Integration Tests

```rust
// tests/integration_test.rs
#[test]
fn test_end_to_end_denoise() {
    // Run Python implementation
    let py_output = run_python_isonet("denoise", &args);
    
    // Run Rust implementation
    let rs_output = run_rust_isonet("denoise", &args);
    
    // Compare outputs
    assert_mrc_near(&py_output, &rs_output, 1e-4);
}
```

### 3. Numerical Accuracy Tests

- CTF 1D/2D/3D: < 1e-5 relative error vs Python
- Missing wedge masks: < 1e-5 relative error
- Network inference: < 1e-4 relative error (due to different BLAS backends)
- Deconvolution: < 1e-5 relative error

---

## Build Configuration

```toml
# Cargo.toml (workspace)
[workspace]
members = ["crates/*"]
resolver = "2"

[workspace.dependencies]
# Core
ndarray = { version = "0.16", features = ["rayon", "serde"] }
ndarray-npy = "0.9"
num-complex = "0.4"
bytemuck = "1.16"

# Parallelism
rayon = "1.10"
dashmap = "6.0"

# I/O
memmap2 = "0.9"
mrc = "0.4"
emstar = "0.2"

# Math/FFT
rustfft = "6.2"
realfft = "3.3"

# Burn (DL)
burn = { version = "0.16", features = ["ndarray", "wgpu", "candle"] }
burn-ndarray = "0.16"

# CLI
clap = { version = "4.5", features = ["derive"] }
indicatif = "0.17"
tracing = "0.1"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
toml = "0.8"

# Testing
approx = "0.5"
criterion = "0.5"

# SIMD
portable-simd = "0.2"
```

---

## Performance Targets

| Operation | Python (PyTorch) | Rust Target | Speedup |
|-----------|-----------------|-------------|---------|
| CTF 3D (128³) | ~50ms | ~10ms | 5x |
| Missing Wedge (128³) | ~20ms | ~5ms | 4x |
| Subvolume Extract | ~5ms | ~1ms | 5x |
| Mask Generation | ~500ms | ~100ms | 5x |
| Deconvolution | ~2s | ~0.5s | 4x |
| Training (1 epoch) | ~60s | ~45s | 1.3x |
| Inference (1 tomo) | ~30s | ~25s | 1.2x |

*Note: Training/inference speedup limited by GPU computation.*

---

## Implementation Phases

### Phase 1: Core Infrastructure

1. `core` - Types and traits
2. `io` - MRC/STAR I/O
3. `fft` - FFT wrappers

### Phase 2: Numerical Computing

1. `ctf` - CTF computation (SIMD)
2. `wedge` - Missing wedge (SIMD)
3. `mask` - Mask generation (SIMD)
4. `deconv` - Deconvolution

### Phase 3: Deep Learning

1. `models` - UNet architectures (Burn)
2. `train` - Training logic
3. `data` - Data loading

### Phase 4: CLI and Integration

1. `cli` - Command interface
2. Integration tests
3. Documentation

---

## Compatibility Notes

### PyTorch Model Loading

- Use `burn-import` to convert PyTorch `.pt` files
- Or implement custom loader for IsoNet2 checkpoints
- Map PyTorch tensor names to Burn module names

### Python Interop (for testing)

- Generate reference outputs with Python
- Store as `.npy` files
- Load in Rust tests with `ndarray-npy`

---

## Future Enhancements

1. **GPU Kernels**: Custom wgpu kernels for CTF/mask generation
2. **Distributed Training**: Multi-node training with `burn-train`
3. **ONNX Export**: Export trained models to ONNX
4. **Web Interface**: WASM build for browser-based visualization
