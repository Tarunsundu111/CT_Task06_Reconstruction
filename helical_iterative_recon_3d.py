#!/usr/bin/env python3
"""
=============================================================================
3D Iterative Helical Cone-Beam CT Reconstruction from Scratch (GPU PyTorch)
=============================================================================

Specialized Scanner Hardware & Helical Geometry:
  - Mode: Helical Cone-Beam CT (120.0 kVp, 200.0 mA, 1.0 s exposure time / 200.0 mAs total)
  - Source-to-Isocenter Distance (SID): 541.0 mm
  - Source-to-Detector Distance (SDD): 949.0 mm
  - Detector Radius from Isocenter: 408.0 mm (SDD - SID)
  - Optical Magnification: SDD / SID = 1.75415896...
  - Beam Collimation at Isocenter: 40.0 mm | Pitch: 0.516
  - Table Feed per Rotation: 20.64 mm | Table Speed: 73.714 mm/s
  - Rotation Time: 0.28 s per 360.0 deg rotation
  - Views per Rotation: 984 views
  - Angular Increment (Delta Theta): 0.36585366 deg/view (2*pi / 984 rad)
  - Total Rotations & Views: 17 rotations totaling 16,728 views
  - Table Increment per View: Delta Y = 20.64 / 984 = 0.02097561 mm/view
  - Rotation Plane: XZ plane | Longitudinal Translation: Y-axis
  - Display Field of View (FOV): 500.0 mm in axial XZ plane
  - Longitudinal Scan Range: 350.0 mm along Y-axis
  - Detector Matrix: 64 rows (V) x 888 cols (U)
      Row pitch: Delta V = Collimation * Mag / 64 = 1.096349 mm
      Col pitch: Delta U = 2 * SDD * tan(HalfFan) / 888 = 0.987702 mm
  - Reconstruction Volume Grid: (NZ, NY, NX) = (142, 232, 194)
      X in [-250.0, 250.0] mm, Y in [-175.0, 175.0] mm, Z in [-250.0, 250.0] mm
  - Calibrated Linear Attenuation Baselines (mm^-1) at 120 kVp:
      Water: 0.0200, Air: 0.0000, Bone: 0.0400, Lung: 0.0050, Soft Tissue: 0.0205

Strict Implementation Constraints:
  1. Zero External CT Libraries:
     Strictly NO ASTRA-Toolbox, TIGRE, RTK, TomoPy, ODL, or skimage.transform.
     3D cone-beam forward projector (A * f) and 3D cone-beam backprojector (A^T * p)
     are implemented from first principles via PyTorch CUDA tensor operations and grid_sample.
  2. Pure GPU Execution:
     All preprocessing, ray-tracing, line integration, voxel backprojection, subset updating,
     and 3D Total Variation regularization execute in GPU VRAM (cuda:0).
  3. Single Image Per File Output Rule (Strict):
     Strictly NO composite multi-panel subplots or GridSpec.
     Every exported visualization is saved to its own dedicated file containing ONE image.
=============================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import time
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# GPU Device Initialization & Hardware Check
# ============================================================================

if not torch.cuda.is_available():
    print("\n[FATAL ERROR] No CUDA-capable NVIDIA GPU detected.")
    print("This reconstruction pipeline strictly requires an NVIDIA GPU with PyTorch CUDA support.")
    print("Exiting without CPU fallback.")
    sys.exit(1)

DEVICE = torch.device("cuda:0")
GPU_NAME = torch.cuda.get_device_name(0)
TOTAL_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 3)


# ============================================================================
# CT Scanner Hardware & Helical Geometric Constants
# ============================================================================

SID = 541.0                   # Source-to-Isocenter Distance (mm)
SDD = 949.0                   # Source-to-Detector Distance (mm)
DET_RADIUS = 408.0            # Detector radius from isocenter: SDD - SID (mm)
MAGNIFICATION = SDD / SID     # ~1.75415896...

COLLIMATION = 40.0            # Beam collimation at isocenter (mm)
PITCH = 0.516                 # Table feed per rot / Collimation
TABLE_FEED_PER_ROT = 20.64    # mm per 360 deg rotation
ROTATION_TIME_S = 0.28        # seconds per 360 deg rotation
VIEWS_PER_ROT = 984           # views per rotation
DELTA_THETA_DEG = 0.36585366  # angular increment per view (deg)
DELTA_THETA_RAD = float(np.deg2rad(DELTA_THETA_DEG))
DELTA_Y = TABLE_FEED_PER_ROT / VIEWS_PER_ROT  # ~0.02097561 mm/view

TOTAL_VIEWS = 16728           # 17 rotations * 984 views
TOTAL_ROTATIONS = 17
SCAN_RANGE_Y = 350.0          # mm along longitudinal translation axis (Y)
DISPLAY_FOV_XZ = 500.0        # mm in axial imaging plane (X and Z)

# Detector specifications
DET_ROWS = 64                 # elevation / along translation axis (Y)
DET_COLS = 888                # transverse / across rotation plane (XZ)

# Detector pixel pitch at detector plane
DET_PIXEL_V = COLLIMATION * MAGNIFICATION / DET_ROWS          # ~1.09635 mm
HALF_FAN_RAD = float(np.arctan((DISPLAY_FOV_XZ / 2.0) / SID))
DET_WIDTH_MM = 2.0 * SDD * float(np.tan(HALF_FAN_RAD))        # ~877.08 mm
DET_PIXEL_U = DET_WIDTH_MM / DET_COLS                         # ~0.98770 mm

# Volume grid dimensions matching ground truth volume (142, 232, 194)
# Coordinate indexing: (NZ, NY, NX)
NZ = 142                      # Axial depth dimension (Z-axis)
NY = 232                      # Longitudinal translation dimension (Y-axis)
NX = 194                      # Axial lateral dimension (X-axis)

# Physical coordinate bounds (mm)
MIN_X, MAX_X = -DISPLAY_FOV_XZ / 2.0, DISPLAY_FOV_XZ / 2.0    # [-250.0, 250.0] mm
MIN_Y, MAX_Y = -SCAN_RANGE_Y / 2.0, SCAN_RANGE_Y / 2.0        # [-175.0, 175.0] mm
MIN_Z, MAX_Z = -DISPLAY_FOV_XZ / 2.0, DISPLAY_FOV_XZ / 2.0    # [-250.0, 250.0] mm

# Calibrated linear attenuation reference baselines (mm^-1) for 120 kVp beam
MU_WATER = 0.0200
MU_AIR = 0.0000
MU_BONE = 0.0400
MU_LUNG = 0.0050
MU_SOFT_TISSUE = 0.0205


# ============================================================================
# 1. GPU Preprocessing (Beer-Lambert Law & Row 32 Validation)
# ============================================================================

def preprocess_projections_gpu(
    raw_projections: np.ndarray,
    air_reference_i0: np.ndarray,
    num_views: int,
    batch_size: int = 512,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Applies the Beer-Lambert log transform on GPU memory in streamed batches:
      p = -ln((I + eps) / (I0 + eps))
    Clamps negative values to 0, replaces NaNs/Infs with 0, and inpaints
    defective hardware detector columns (channels 31 and 196) via nearest neighbor.
    
    Returns:
      sinogram: Preprocessed attenuation sinogram array of shape (num_views, DET_ROWS, DET_COLS)
    """
    sinogram = np.zeros((num_views, DET_ROWS, DET_COLS), dtype=np.float32)

    # Transfer air reference I0 to GPU
    i0_gpu = torch.from_numpy(air_reference_i0.astype(np.float32)).to(DEVICE)
    if i0_gpu.ndim == 1:
        i0_gpu = i0_gpu.view(1, 1, DET_COLS)
    elif i0_gpu.ndim == 2:
        i0_gpu = i0_gpu.unsqueeze(0)  # (1, 64, 888)

    # Safe denominator avoiding zero-division on uncalibrated dead elements
    i0_safe = torch.where(i0_gpu > 0, i0_gpu, torch.ones_like(i0_gpu))

    for start_idx in range(0, num_views, batch_size):
        end_idx = min(start_idx + batch_size, num_views)
        cur_batch = end_idx - start_idx

        raw_batch_host = np.ascontiguousarray(raw_projections[start_idx:end_idx], dtype=np.float32).copy()
        raw_batch_gpu = torch.from_numpy(raw_batch_host).to(DEVICE)

        # Transmission ratio with clamping
        ratio = torch.clamp((raw_batch_gpu + eps) / (i0_safe + eps), min=eps, max=1.0)
        p_batch = -torch.log(ratio)
        p_batch = torch.clamp(p_batch, min=0.0, max=23.02585)
        p_batch = torch.where(torch.isfinite(p_batch), p_batch, torch.zeros_like(p_batch))

        # Hardware defective detector channel inpainting (columns 31 and 196)
        p_batch[:, :, 31] = p_batch[:, :, 30]
        p_batch[:, :, 196] = p_batch[:, :, 195]

        sinogram[start_idx:end_idx] = p_batch.cpu().numpy()

        del raw_batch_gpu, p_batch, ratio
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return sinogram


def validate_sinogram_preprocessing(
    sinogram: np.ndarray,
    reference_sinogram_slice32: np.ndarray
) -> float:
    """
    Validates preprocessed detector row 32 against helical_sinogram_16728_slice32.npy.
    Strictly asserts max absolute error < 1e-4.
    """
    num_views = sinogram.shape[0]
    processed_row32 = sinogram[:, 32, :]
    ref_row32 = reference_sinogram_slice32[:num_views, :]
    max_err = float(np.max(np.abs(processed_row32 - ref_row32)))
    mean_err = float(np.mean(np.abs(processed_row32 - ref_row32)))
    print(f"  [Validation] Row 32 vs reference slice32: max abs error = {max_err:.6e}, mean = {mean_err:.6e}")
    assert max_err < 1e-4, f"Preprocessed sinogram mismatch! Max error {max_err:.6e} >= 1e-4"
    print("  [Validation PASSED] Detector row 32 strictly matches reference (< 1e-4).")
    return max_err


# ============================================================================
# 2. From-Scratch 3D Cone-Beam Forward Projector (Ray-Driven)
# ============================================================================

def forward_project_cone3d_gpu(
    vol: torch.Tensor,
    view_indices: torch.Tensor,
    batch_views: int = 16,
    ds: float = 1.5,
    chunk_t: int = 32
) -> torch.Tensor:
    """
    From-Scratch 3D Cone-Beam Forward Projector (A * f) implemented via PyTorch CUDA.
    
    Ray Parameterization:
      s[k]   = [SID * cos(theta_k), k * Delta Y, SID * sin(theta_k)]
      d_c[k] = [-DET_RADIUS * cos(theta_k), k * Delta Y, -DET_RADIUS * sin(theta_k)]
      u_hat  = [-sin(theta_k), 0, cos(theta_k)]
      v_hat  = [0, 1, 0]
      d_{k, v, u} = d_c[k] + (u - (N_cols-1)/2) * Delta_U * u_hat + (v - (N_rows-1)/2) * Delta_V * v_hat
      r_hat  = (d - s) / ||d - s||_2
      p_{calc} = sum_i mu(s + t_i * r_hat) * ds
    Evaluated with 5D grid_sample on GPU.
    """
    device = vol.device
    if not isinstance(view_indices, torch.Tensor):
        view_indices = torch.as_tensor(view_indices, device=device, dtype=torch.float32)
    elif view_indices.device != device:
        view_indices = view_indices.to(device=device, dtype=torch.float32)
    else:
        view_indices = view_indices.to(dtype=torch.float32)

    vol_5d = vol.view(1, 1, NZ, NY, NX)
    n_views = len(view_indices)
    out_sino = torch.zeros((n_views, DET_ROWS, DET_COLS), device=device, dtype=torch.float32)

    # Detector pixel offsets from center (mm)
    u_vals = (torch.arange(DET_COLS, device=device, dtype=torch.float32) - (DET_COLS - 1) * 0.5) * DET_PIXEL_U
    v_vals = (torch.arange(DET_ROWS, device=device, dtype=torch.float32) - (DET_ROWS - 1) * 0.5) * DET_PIXEL_V
    v_grid, u_grid = torch.meshgrid(v_vals, u_vals, indexing="ij")

    # Volume bounds inverse scale for [-1, 1]^3 mapping: x/250, y/175, z/250
    inv_scale = torch.tensor([1.0 / 250.0, 1.0 / 175.0, 1.0 / 250.0], device=device, dtype=torch.float32).view(1, 1, 1, 3)
    v_vec = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32).view(1, 1, 1, 3)

    # Active ray segment through cylinder bounding volume: [187.5 mm, 895.0 mm]
    t_steps = torch.arange(187.5, 895.0, ds, device=device, dtype=torch.float32)

    for b_start in range(0, n_views, batch_views):
        b_end = min(b_start + batch_views, n_views)
        b_indices = view_indices[b_start:b_end]
        B = len(b_indices)

        theta = b_indices * DELTA_THETA_RAD
        cos_t = torch.cos(theta).view(B, 1, 1, 1)
        sin_t = torch.sin(theta).view(B, 1, 1, 1)
        table_y = (b_indices * DELTA_Y).view(B, 1, 1, 1)

        # Source coordinates
        src = torch.cat([SID * cos_t, table_y, SID * sin_t], dim=-1)
        # Detector center coordinates
        det_c = torch.cat([-DET_RADIUS * cos_t, table_y, -DET_RADIUS * sin_t], dim=-1)
        # Transverse detector axis
        u_vec = torch.cat([-sin_t, torch.zeros_like(sin_t), cos_t], dim=-1)

        # Target detector pixel coordinates
        det_pts = det_c + u_grid.view(1, DET_ROWS, DET_COLS, 1) * u_vec + v_grid.view(1, DET_ROWS, DET_COLS, 1) * v_vec
        ray_dirs = det_pts - src
        ray_unit = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)

        # Normalized coordinates for 5D grid_sample
        src_norm = (src * inv_scale).unsqueeze(1)                   # (B, 1, 1, 1, 3)
        ray_norm = (ray_unit * inv_scale).unsqueeze(1)              # (B, 1, DET_ROWS, DET_COLS, 3)

        vol_b = vol_5d.expand(B, 1, NZ, NY, NX)
        p_batch = torch.zeros(B, DET_ROWS, DET_COLS, device=device, dtype=torch.float32)

        for t_idx in range(0, len(t_steps), chunk_t):
            t_chunk = t_steps[t_idx:t_idx + chunk_t].view(1, -1, 1, 1, 1)
            grid = src_norm + t_chunk * ray_norm                     # (B, K, DET_ROWS, DET_COLS, 3)
            samples = F.grid_sample(vol_b, grid, mode="bilinear", align_corners=True, padding_mode="zeros")
            p_batch += samples.squeeze(1).sum(dim=1) * ds

        out_sino[b_start:b_end] = p_batch

    return out_sino


# ============================================================================
# 3. From-Scratch 3D Voxel-Driven Backprojector (Cone-Beam)
# ============================================================================

def backproject_cone3d_gpu(
    residual_sino: torch.Tensor,
    view_indices: torch.Tensor,
    batch_views: int = 8
) -> torch.Tensor:
    """
    From-Scratch 3D Voxel-Driven Backprojector (A^T * p) implemented via PyTorch CUDA.
    
    For each voxel center (x, y, z):
      1. Source-to-voxel central ray distance:
           L(x, z, theta_k) = SID - (x * cos theta_k + z * sin theta_k)
      2. Perspective projection mapping onto virtual detector plane:
           u' = (SDD / L) * (-x * sin theta_k + z * cos theta_k)
           v' = (SDD / L) * (y - k * Delta Y)
      3. Normalize (u', v') to [-1, 1]^2 and sample 2D residual sinogram via grid_sample.
      4. Distance-squared cone-beam weighting: (SID / L)^2.
      5. Accumulate backprojected updates across views directly into 3D volume grid.
    """
    device = residual_sino.device
    if not isinstance(view_indices, torch.Tensor):
        view_indices = torch.as_tensor(view_indices, device=device, dtype=torch.float32)
    elif view_indices.device != device:
        view_indices = view_indices.to(device=device, dtype=torch.float32)
    else:
        view_indices = view_indices.to(dtype=torch.float32)

    n_views = len(view_indices)
    vol_accum = torch.zeros((NZ, NY, NX), device=device, dtype=torch.float32)

    # Physical voxel center coordinate grids (mm)
    z_coords = torch.linspace(MIN_Z, MAX_Z, NZ, device=device, dtype=torch.float32).view(1, NZ, 1, 1)
    y_coords = torch.linspace(MIN_Y, MAX_Y, NY, device=device, dtype=torch.float32).view(1, 1, NY, 1)
    x_coords = torch.linspace(MIN_X, MAX_X, NX, device=device, dtype=torch.float32).view(1, 1, 1, NX)

    # Maximum detector half-width and half-height for [-1, 1] normalization
    U_max = ((DET_COLS - 1) * 0.5) * DET_PIXEL_U
    V_max = ((DET_ROWS - 1) * 0.5) * DET_PIXEL_V

    for b_start in range(0, n_views, batch_views):
        b_end = min(b_start + batch_views, n_views)
        b_indices = view_indices[b_start:b_end]
        B = len(b_indices)

        theta = b_indices * DELTA_THETA_RAD
        cos_t = torch.cos(theta).view(B, 1, 1, 1)
        sin_t = torch.sin(theta).view(B, 1, 1, 1)
        table_y = (b_indices * DELTA_Y).view(B, 1, 1, 1)

        # 1. Source-to-voxel central ray distance
        L = SID - (x_coords * cos_t + z_coords * sin_t)             # (B, NZ, 1, NX)
        inv_L = 1.0 / torch.clamp(L, min=1.0)

        # 2. Perspective projection onto virtual detector plane (u', v')
        u_prime = (SDD * inv_L) * (-x_coords * sin_t + z_coords * cos_t)  # (B, NZ, 1, NX)
        v_prime = (SDD * inv_L) * (y_coords - table_y)                    # (B, NZ, NY, NX)

        # 3. Normalize into [-1, 1]^2 for 2D grid_sample
        u_norm = (u_prime / U_max).expand(B, NZ, NY, NX)
        v_norm = v_prime / V_max

        grid = torch.stack([u_norm, v_norm], dim=-1).view(B, NZ * NY, NX, 2)
        sino_b = residual_sino[b_start:b_end].unsqueeze(1)

        sampled = F.grid_sample(sino_b, grid, mode="bilinear", align_corners=True, padding_mode="zeros")
        sampled = sampled.view(B, NZ, NY, NX)

        # 4. Distance-squared cone-beam weighting: (SID / L)^2
        w = (SID * inv_L) ** 2                                       # (B, NZ, 1, NX)
        vol_accum += (sampled * w).sum(dim=0)

    return vol_accum


# ============================================================================
# 4. Native PyTorch CUDA 3D Total Variation Regularizer (Chambolle Dual)
# ============================================================================

def tv3d_proximal_chambolle_gpu(
    vol: torch.Tensor,
    weight: float,
    n_iter: int = 6,
    tau: float = 1.0 / 6.0
) -> torch.Tensor:
    """
    Solves the 3D Total Variation denoising problem:
      min_f 0.5 * ||f - vol||_2^2 + weight * TV(f)
    using Chambolle's dual projection algorithm with Neumann boundary conditions.
    """
    if weight <= 0.0:
        return vol

    pz = torch.zeros_like(vol)
    py = torch.zeros_like(vol)
    px = torch.zeros_like(vol)

    inv_weight = 1.0 / weight

    for _ in range(n_iter):
        # 3D Divergence with Neumann boundary conditions
        div = torch.zeros_like(vol)
        div[0, :, :] += pz[0, :, :]
        div[1:-1, :, :] += pz[1:-1, :, :] - pz[:-2, :, :]
        div[-1, :, :] -= pz[-2, :, :]

        div[:, 0, :] += py[:, 0, :]
        div[:, 1:-1, :] += py[:, 1:-1, :] - py[:, :-2, :]
        div[:, -1, :] -= py[:, -2, :]

        div[:, :, 0] += px[:, :, 0]
        div[:, :, 1:-1] += px[:, :, 1:-1] - px[:, :, :-2]
        div[:, :, -1] -= px[:, :, -2]

        v = div - vol * inv_weight

        # 3D Forward difference gradient
        gz = torch.zeros_like(vol)
        gy = torch.zeros_like(vol)
        gx = torch.zeros_like(vol)

        gz[:-1, :, :] = v[1:, :, :] - v[:-1, :, :]
        gy[:, :-1, :] = v[:, 1:, :] - v[:, :-1, :]
        gx[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]

        # Projection onto Euclidean dual unit ball
        norm_g = torch.sqrt(gz ** 2 + gy ** 2 + gx ** 2 + 1e-12)
        denom = 1.0 + tau * norm_g

        pz = (pz + tau * gz) / denom
        py = (py + tau * gy) / denom
        px = (px + tau * gx) / denom

    # Final divergence reconstruction
    div = torch.zeros_like(vol)
    div[0, :, :] += pz[0, :, :]
    div[1:-1, :, :] += pz[1:-1, :, :] - pz[:-2, :, :]
    div[-1, :, :] -= pz[-2, :, :]

    div[:, 0, :] += py[:, 0, :]
    div[:, 1:-1, :] += py[:, 1:-1, :] - py[:, :-2, :]
    div[:, -1, :] -= py[:, -2, :]

    div[:, :, 0] += px[:, :, 0]
    div[:, :, 1:-1] += px[:, :, 1:-1] - px[:, :, :-2]
    div[:, :, -1] -= px[:, :, -2]

    f_out = vol - weight * div
    return f_out


# ============================================================================
# 5. OS-SART Iterative Reconstruction Loop
# ============================================================================

def reconstruct_os_sart(
    sinogram_host: np.ndarray,
    view_indices: np.ndarray,
    num_iterations: int = 8,
    num_subsets: int = 16,
    relaxation_lambda: float = 0.6,
    tv_weight: float = 2e-5,
    ds: float = 1.5,
    non_negative: bool = True
) -> Tuple[np.ndarray, List[float]]:
    """
    Explicit Ordered Subsets SART (OS-SART) reconstruction loop on GPU:
      f^(k+1) = max(0, f^(k) + lambda * C_s * A_s^T ( R_s * (p_s - A_s f^(k)) ) )
    with Chambolle 3D Total Variation regularization after each full iteration.
    
    Streams subset projections and normalizations from CPU to GPU dynamically,
    guaranteeing safe execution without CUDA Out-Of-Memory errors even on 4 GB GPUs.
    """
    total_views = len(view_indices)
    n_subsets = min(num_subsets, total_views)

    print("\n" + "=" * 70)
    print("STEP 3: Explicit OS-SART Iterative Reconstruction on GPU")
    print(f"  Views: {total_views} | Subsets: {n_subsets} | Iterations: {num_iterations}")
    print(f"  Relaxation (lambda): {relaxation_lambda} | TV Weight: {tv_weight} | Ray Step: {ds} mm")
    print("=" * 70)

    t0_pre = time.time()
    sino_cpu = torch.from_numpy(sinogram_host)
    view_indices_gpu = torch.from_numpy(view_indices).to(device=DEVICE, dtype=torch.float32)

    p_norm = float(np.linalg.norm(sinogram_host))
    if p_norm == 0.0:
        p_norm = 1.0

    print("  Precomputing Row Normalization (R) on GPU...")
    ones_vol = torch.ones((NZ, NY, NX), device=DEVICE, dtype=torch.float32)
    r_full = forward_project_cone3d_gpu(ones_vol, view_indices_gpu, batch_views=16, ds=ds)
    R_full_cpu = (1.0 / torch.clamp(r_full, min=1e-4)).cpu()
    del ones_vol, r_full
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Precomputing {n_subsets} Subset Column Normalizations (C_s) on GPU...")
    C_list_cpu = []
    for s in range(n_subsets):
        sub_views = view_indices_gpu[s::n_subsets]
        ones_sub = torch.ones((len(sub_views), DET_ROWS, DET_COLS), device=DEVICE, dtype=torch.float32)
        c = backproject_cone3d_gpu(ones_sub, sub_views, batch_views=8)
        c_s = 1.0 / torch.clamp(c, min=1e-4)
        c_s[c < 1e-3] = 0.0
        C_list_cpu.append(c_s.cpu())
        del ones_sub, c, c_s
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"  Normalization precomputation complete in {time.time() - t0_pre:.2f} s")

    # Initial volume estimate f_0 = 0 in GPU VRAM
    f_gpu = torch.zeros((NZ, NY, NX), device=DEVICE, dtype=torch.float32)
    residuals = []

    for it in range(1, num_iterations + 1):
        t_it_start = time.time()
        res_sq_sum = 0.0

        for s in range(n_subsets):
            sub_rel_idx = np.arange(s, total_views, n_subsets)
            sub_views = view_indices_gpu[s::n_subsets]

            # Stream subset projection and normalizers to GPU
            p_sub = sino_cpu[sub_rel_idx].to(DEVICE)
            R_sub = R_full_cpu[sub_rel_idx].to(DEVICE)
            C_sub = C_list_cpu[s].to(DEVICE)

            # 1. Forward project current volume estimate: af = A_s f
            af_sub = forward_project_cone3d_gpu(f_gpu, sub_views, batch_views=16, ds=ds)

            # 2. Subset residual weighted by row normalization: Delta_p = R_s * (p_s - A_s f)
            diff = p_sub - af_sub
            res_sq_sum += float(torch.sum(diff ** 2).item())

            # 3. Backproject normalized residual: Delta_f = A_s^T (R_s * diff)
            delta_f = backproject_cone3d_gpu(diff * R_sub, sub_views, batch_views=8)

            # 4. SART update step: f <- f + lambda * (C_s * Delta_f)
            f_gpu = f_gpu + relaxation_lambda * (delta_f * C_sub)

            # 5. Non-negativity constraint f >= 0
            if non_negative:
                f_gpu = torch.clamp(f_gpu, min=0.0)

            del p_sub, R_sub, C_sub, af_sub, diff, delta_f

        # 3D Total Variation Regularization after each full iteration
        if tv_weight > 0.0:
            f_gpu = tv3d_proximal_chambolle_gpu(f_gpu, weight=tv_weight, n_iter=6, tau=1.0 / 6.0)
            if non_negative:
                f_gpu = torch.clamp(f_gpu, min=0.0)

        # Track relative data discrepancy ||A f^(k) - p||_2 / ||p||_2
        rel_residual = float(np.sqrt(res_sq_sum) / p_norm)
        residuals.append(rel_residual)

        it_time = time.time() - t_it_start
        print(f"  Iteration {it:2d}/{num_iterations:2d} | Relative Residual: {rel_residual:.6e} | Time: {it_time:.2f} s | Voxel Max: {float(f_gpu.max().item()):.5f}")

    recon_host = f_gpu.cpu().numpy()
    return recon_host, residuals


# ============================================================================
# 6. From-Scratch Metrics & Hounsfield Units Conversion
# ============================================================================

def convert_to_hounsfield_units(mu_volume: np.ndarray) -> np.ndarray:
    """
    Converts linear attenuation coefficients mu (mm^-1) to Hounsfield Units (HU):
      HU = 1000 * (mu - mu_water) / (mu_water - mu_air)
    where mu_water = 0.020 mm^-1, mu_air = 0.0 mm^-1.
    """
    return 1000.0 * (mu_volume - MU_WATER) / (MU_WATER - MU_AIR)


def convert_density_to_attenuation(density: np.ndarray, material: np.ndarray) -> np.ndarray:
    """
    Converts ground truth density volume to linear attenuation coefficients mu (mm^-1):
      - Air (mat 0): 0.0000 mm^-1
      - Soft Tissue (mat 1): 0.0205 mm^-1
      - Bone (mat 2): 0.0400 mm^-1
      - Lung (mat 3): 0.0050 mm^-1
    """
    gt_mu = np.zeros_like(density, dtype=np.float32)
    gt_mu[material == 0] = MU_AIR
    gt_mu[material == 1] = MU_SOFT_TISSUE
    gt_mu[material == 2] = MU_BONE
    gt_mu[material == 3] = MU_LUNG
    return gt_mu


def conv2d_separable_numpy(img: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    """Performs separable 2D convolution with reflect padding in pure NumPy."""
    k = len(kernel_1d)
    pad = k // 2
    padded = np.pad(img, pad, mode="reflect")
    temp = np.zeros((img.shape[0], padded.shape[1]), dtype=np.float32)
    for i in range(k):
        temp += kernel_1d[i] * padded[i:i + img.shape[0], :]
    out = np.zeros_like(img, dtype=np.float32)
    for j in range(k):
        out += kernel_1d[j] * temp[:, j:j + img.shape[1]]
    return out


def compute_ssim_2d(
    img1: np.ndarray,
    img2: np.ndarray,
    data_range: float,
    win_size: int = 11,
    sigma: float = 1.5
) -> float:
    """
    Computes Structural Similarity Index (SSIM) between two 2D images
    from first principles using an 11x11 separable Gaussian window (strictly no skimage).
    """
    coords = np.arange(win_size) - (win_size - 1) / 2.0
    kernel_1d = np.exp(-(coords ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    kernel_1d = kernel_1d / np.sum(kernel_1d)

    k1, k2 = 0.01, 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu1 = conv2d_separable_numpy(img1, kernel_1d)
    mu2 = conv2d_separable_numpy(img2, kernel_1d)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d_separable_numpy(img1 * img1, kernel_1d) - mu1_sq
    sigma2_sq = conv2d_separable_numpy(img2 * img2, kernel_1d) - mu2_sq
    sigma12 = conv2d_separable_numpy(img1 * img2, kernel_1d) - mu1_mu2

    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / (den + 1e-12)
    return float(np.mean(ssim_map))


def compute_reconstruction_metrics(gt_mu: np.ndarray, recon_mu: np.ndarray) -> Dict[str, float]:
    """Computes PSNR, RMSE, axial slice SSIM, and volume-average SSIM from scratch."""
    mse = float(np.mean((gt_mu - recon_mu) ** 2))
    rmse = float(np.sqrt(mse))

    data_max = float(np.max(gt_mu))
    data_min = float(np.min(gt_mu))
    data_range = max(data_max - data_min, 1e-6)

    psnr = float(20.0 * np.log10(data_max / (rmse + 1e-12))) if rmse > 0 else float("inf")

    mid_y = gt_mu.shape[1] // 2
    axial_ssim = compute_ssim_2d(gt_mu[:, mid_y, :], recon_mu[:, mid_y, :], data_range=data_range)

    ssim_list = []
    step = max(1, gt_mu.shape[1] // 16)
    for iy in range(0, gt_mu.shape[1], step):
        s_gt = gt_mu[:, iy, :]
        s_rc = recon_mu[:, iy, :]
        if s_gt.max() > 1e-5:
            ssim_list.append(compute_ssim_2d(s_gt, s_rc, data_range=data_range))
    vol_ssim = float(np.mean(ssim_list)) if ssim_list else axial_ssim

    return {
        "RMSE": rmse,
        "PSNR (dB)": psnr,
        "SSIM (Axial Slice)": axial_ssim,
        "SSIM (Volume Average)": vol_ssim,
        "MSE": mse
    }


# ============================================================================
# 7. Output Deliverables: Strict Single-Image-Per-File Visualization
# ============================================================================

def export_standalone_image(
    data: np.ndarray,
    output_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    extent: Optional[List[float]] = None,
    aspect: str = "equal",
    cbar_label: str = ""
) -> None:
    """
    Exports exactly ONE standalone image per file (strictly NO composite subplots or GridSpec).
    """
    fig = plt.figure(figsize=(8, 7), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, aspect=aspect)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved Standalone Deliverable] {output_path}")


def export_all_deliverables(
    recon_mu: np.ndarray,
    gt_mu: np.ndarray,
    residuals: List[float],
    output_dir: Union[str, Path] = "."
) -> List[str]:
    """
    Exports all 8 standalone image deliverables and reconstructed volume array:
      1. reconstructed_chest_volume.npy
      2. 01_reconstructed_axial_slice.png
      3. 02_ground_truth_axial_slice.png
      4. 03_absolute_error_axial_map.png
      5. 04_reconstructed_coronal_slice.png
      6. 05_reconstructed_sagittal_slice.png
      7. 06_3d_mip_coronal_view.png
      8. 07_3d_mip_axial_view.png
      9. 08_convergence_loss_curve.png
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    print("\n" + "=" * 70)
    print("STEP 5: Exporting Deliverables (Strict Single-Image-Per-File Rule)")
    print("=" * 70)

    # 1. Save 3D reconstructed volume array
    out_vol_path = str(out_dir / "reconstructed_chest_volume.npy")
    np.save(out_vol_path, recon_mu)
    print(f"  [Saved Deliverable] {out_vol_path} ({os.path.getsize(out_vol_path) / 1e6:.2f} MB)")
    generated_files.append(out_vol_path)

    # Convert linear attenuation mu (mm^-1) to Hounsfield Units (HU)
    recon_hu = convert_to_hounsfield_units(recon_mu)
    gt_hu = convert_to_hounsfield_units(gt_mu)

    nz, ny, nx = recon_mu.shape
    mid_z = nz // 2  # 71
    mid_y = ny // 2  # 116
    mid_x = nx // 2  # 97

    # 01. Central reconstructed axial 2D slice (XZ plane, HU)
    f01_path = str(out_dir / "01_reconstructed_axial_slice.png")
    export_standalone_image(
        data=recon_hu[:, mid_y, :],
        output_path=f01_path,
        title="01: Reconstructed Central Axial Slice (XZ Plane, HU)",
        xlabel="X (mm)",
        ylabel="Z (mm)",
        cmap="gray",
        vmin=-1000, vmax=1000,
        extent=[MIN_X, MAX_X, MIN_Z, MAX_Z],
        aspect="equal",
        cbar_label="Hounsfield Units (HU)"
    )
    generated_files.append(f01_path)

    # 02. Ground truth reference axial slice (HU)
    f02_path = str(out_dir / "02_ground_truth_axial_slice.png")
    export_standalone_image(
        data=gt_hu[:, mid_y, :],
        output_path=f02_path,
        title="02: Ground Truth Central Axial Slice (XZ Plane, HU)",
        xlabel="X (mm)",
        ylabel="Z (mm)",
        cmap="gray",
        vmin=-1000, vmax=1000,
        extent=[MIN_X, MAX_X, MIN_Z, MAX_Z],
        aspect="equal",
        cbar_label="Hounsfield Units (HU)"
    )
    generated_files.append(f02_path)

    # 03. Absolute difference error map (|Reconstruction - Ground Truth|)
    abs_error_hu = np.abs(recon_hu[:, mid_y, :] - gt_hu[:, mid_y, :])
    f03_path = str(out_dir / "03_absolute_error_axial_map.png")
    export_standalone_image(
        data=abs_error_hu,
        output_path=f03_path,
        title=f"03: Absolute Difference Error Map |Recon - GT| (Axial Slice, HU)\nMax Error: {abs_error_hu.max():.1f} HU | Mean: {abs_error_hu.mean():.1f} HU",
        xlabel="X (mm)",
        ylabel="Z (mm)",
        cmap="inferno",
        vmin=0.0,
        vmax=float(np.percentile(abs_error_hu, 99.5)),
        extent=[MIN_X, MAX_X, MIN_Z, MAX_Z],
        aspect="equal",
        cbar_label="Absolute Error (HU)"
    )
    generated_files.append(f03_path)

    # 04. Central coronal reconstructed 2D slice (XY plane, HU)
    f04_path = str(out_dir / "04_reconstructed_coronal_slice.png")
    export_standalone_image(
        data=recon_hu[mid_z, :, :],
        output_path=f04_path,
        title="04: Reconstructed Central Coronal Slice (XY Plane, HU)",
        xlabel="X (mm)",
        ylabel="Y - Translation Axis (mm)",
        cmap="gray",
        vmin=-1000, vmax=1000,
        extent=[MIN_X, MAX_X, MIN_Y, MAX_Y],
        aspect="auto",
        cbar_label="Hounsfield Units (HU)"
    )
    generated_files.append(f04_path)

    # 05. Central sagittal reconstructed 2D slice (YZ plane, HU)
    f05_path = str(out_dir / "05_reconstructed_sagittal_slice.png")
    export_standalone_image(
        data=recon_hu[:, :, mid_x].T,
        output_path=f05_path,
        title="05: Reconstructed Central Sagittal Slice (YZ Plane, HU)",
        xlabel="Z (mm)",
        ylabel="Y - Translation Axis (mm)",
        cmap="gray",
        vmin=-1000, vmax=1000,
        extent=[MIN_Z, MAX_Z, MIN_Y, MAX_Y],
        aspect="auto",
        cbar_label="Hounsfield Units (HU)"
    )
    generated_files.append(f05_path)

    # 06. 3D Maximum Intensity Projection (MIP) along Coronal Axis (Z projection)
    mip_coronal = np.max(recon_hu, axis=0)  # Shape (NY, NX)
    f06_path = str(out_dir / "06_3d_mip_coronal_view.png")
    export_standalone_image(
        data=mip_coronal,
        output_path=f06_path,
        title="06: 3D Maximum Intensity Projection (MIP) - Coronal View (HU)",
        xlabel="X (mm)",
        ylabel="Y - Translation Axis (mm)",
        cmap="gray",
        vmin=-1000, vmax=1000,
        extent=[MIN_X, MAX_X, MIN_Y, MAX_Y],
        aspect="auto",
        cbar_label="Hounsfield Units (HU)"
    )
    generated_files.append(f06_path)

    # 07. 3D Maximum Intensity Projection (MIP) along Axial Axis (Y projection)
    mip_axial = np.max(recon_hu, axis=1)    # Shape (NZ, NX)
    f07_path = str(out_dir / "07_3d_mip_axial_view.png")
    export_standalone_image(
        data=mip_axial,
        output_path=f07_path,
        title="07: 3D Maximum Intensity Projection (MIP) - Axial View (HU)",
        xlabel="X (mm)",
        ylabel="Z (mm)",
        cmap="gray",
        vmin=-1000, vmax=1000,
        extent=[MIN_X, MAX_X, MIN_Z, MAX_Z],
        aspect="equal",
        cbar_label="Hounsfield Units (HU)"
    )
    generated_files.append(f07_path)

    # 08. Standalone convergence curve plotting relative data discrepancy vs iteration
    f08_path = str(out_dir / "08_convergence_loss_curve.png")
    fig = plt.figure(figsize=(8, 6), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    if residuals:
        iters = np.arange(1, len(residuals) + 1)
        ax.semilogy(iters, residuals, "b-o", markersize=6, linewidth=1.8, label=r"Relative Discrepancy $\|Af^{(k)} - p\|_2 / \|p\|_2$")
        ax.set_xlabel("OS-SART Iteration Number", fontsize=11)
        ax.set_ylabel(r"Data Discrepancy ($\log_{10}$ scale)", fontsize=11)
        ax.set_title("08: OS-SART Iterative Convergence Curve", fontsize=12, fontweight="bold", pad=12)
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout()
    fig.savefig(f08_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved Standalone Deliverable] {f08_path}")
    generated_files.append(f08_path)

    return generated_files


# ============================================================================
# Main Execution Entry Point & CLI Parser
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="3D Iterative Helical Cone-Beam CT Reconstruction (Pure PyTorch CUDA)"
    )
    parser.add_argument(
        "--prototype", action="store_true", default=False,
        help="Reconstruct 1 rotation (984 views) for fast prototyping without VRAM exhaustion"
    )
    parser.add_argument(
        "--views", type=int, default=None,
        help="Explicit number of projection views to run (default: 984 for prototype, or full dataset)"
    )
    parser.add_argument(
        "--iterations", type=int, default=8,
        help="Number of OS-SART iterations (default: 8)"
    )
    parser.add_argument(
        "--subsets", type=int, default=16,
        help="Number of ordered subsets (default: 16)"
    )
    parser.add_argument(
        "--relaxation", type=float, default=0.6,
        help="OS-SART relaxation parameter lambda (default: 0.6)"
    )
    parser.add_argument(
        "--tv_weight", type=float, default=2e-5,
        help="3D TV regularization weight (default: 2e-5)"
    )
    parser.add_argument(
        "--step", type=float, default=1.5,
        help="Ray step size in mm for forward projector (default: 1.5)"
    )
    parser.add_argument(
        "--data_dir", type=str, default=".",
        help="Directory containing input .npy files (default: current directory)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=".",
        help="Directory to save output deliverables (default: current directory)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("3D ITERATIVE HELICAL CONE-BEAM CT RECONSTRUCTION (FROM SCRATCH)")
    print("Pure PyTorch CUDA Operations | Zero External CT Toolboxes")
    print("=" * 70)
    print(f"  GPU Hardware: {GPU_NAME} ({TOTAL_VRAM_GB:.2f} GB VRAM)")

    data_dir = Path(args.data_dir)

    # 1. Verify and Load Input Data Files
    print("\n" + "=" * 70)
    print("STEP 1: Loading Scanner Input Datasets (.npy)")
    print("=" * 70)
    req_files = [
        "helical_projections_3d_16728.npy",
        "air_reference_I0.npy",
        "helical_sinogram_raw_16728_slice32.npy",
        "helical_sinogram_16728_slice32.npy",
        "voxelized_chest_density.npy",
        "voxelized_chest_material.npy"
    ]
    for rf in req_files:
        p = data_dir / rf
        if not p.exists():
            raise FileNotFoundError(f"Required input file missing: {p.resolve()}")
        print(f"  Found: {rf:<38} ({p.stat().st_size / 1e6:8.2f} MB)")

    print("\n  Loading air_reference_I0.npy...")
    i0 = np.load(data_dir / "air_reference_I0.npy").astype(np.float32)

    print("  Loading helical_sinogram_16728_slice32.npy...")
    sino32_ref = np.load(data_dir / "helical_sinogram_16728_slice32.npy").astype(np.float32)

    print("  Loading helical_projections_3d_16728.npy (memory-mapped)...")
    raw_projections = np.load(data_dir / "helical_projections_3d_16728.npy", mmap_mode="r")
    total_dataset_views = raw_projections.shape[0]

    # Determine view range
    if args.prototype:
        num_views_to_run = VIEWS_PER_ROT
    elif args.views is not None:
        num_views_to_run = args.views
    else:
        num_views_to_run = VIEWS_PER_ROT

    num_views_to_run = min(num_views_to_run, total_dataset_views)
    print(f"\n  [Execution Mode] Reconstructing {num_views_to_run} views ({num_views_to_run / VIEWS_PER_ROT:.2f} rotations)")
    view_indices = np.arange(num_views_to_run, dtype=np.int32)

    # 2. GPU Preprocessing (Beer-Lambert Conversion)
    print("\n" + "=" * 70)
    print("STEP 2: GPU Beer-Lambert Preprocessing (Transmission to Attenuation)")
    print("=" * 70)
    t0_prep = time.time()
    sinogram = preprocess_projections_gpu(
        raw_projections=raw_projections,
        air_reference_i0=i0,
        num_views=num_views_to_run,
        batch_size=512,
        eps=1e-6
    )
    print(f"  Beer-Lambert preprocessing complete in {time.time() - t0_prep:.2f} s")
    print(f"  Attenuation sinogram shape: {sinogram.shape}, range: [{sinogram.min():.4f}, {sinogram.max():.4f}]")

    validate_sinogram_preprocessing(sinogram, sino32_ref)

    # 3. Explicit OS-SART Reconstruction Loop
    t0_recon = time.time()
    recon_volume, residuals = reconstruct_os_sart(
        sinogram_host=sinogram,
        view_indices=view_indices,
        num_iterations=args.iterations,
        num_subsets=args.subsets,
        relaxation_lambda=args.relaxation,
        tv_weight=args.tv_weight,
        ds=args.step,
        non_negative=True
    )
    total_recon_time = time.time() - t0_recon
    print(f"\n  Reconstruction completed in {total_recon_time:.2f} s ({total_recon_time / 60.0:.2f} min)")
    print(f"  Reconstructed volume shape: {recon_volume.shape}, range: [{recon_volume.min():.6f}, {recon_volume.max():.6f}] mm^-1")

    # 4. Ground-Truth Benchmarking & Quantitative Metrics
    print("\n" + "=" * 70)
    print("STEP 4: Ground-Truth Phantom Benchmarking & Quantitative Metrics")
    print("=" * 70)
    gt_density = np.load(data_dir / "voxelized_chest_density.npy").astype(np.float32)
    gt_material = np.load(data_dir / "voxelized_chest_material.npy").astype(np.int32)
    gt_mu = convert_density_to_attenuation(gt_density, gt_material)

    metrics = compute_reconstruction_metrics(gt_mu, recon_volume)
    print(f"  +------------------------------------------------+")
    print(f"  |  Quantitative Benchmarking Metrics             |")
    print(f"  +------------------------------------------------+")
    print(f"  |  PSNR:                  {metrics['PSNR (dB)']:>10.2f} dB        |")
    print(f"  |  SSIM (Axial Slice):    {metrics['SSIM (Axial Slice)']:>10.4f}           |")
    print(f"  |  SSIM (Volume Average): {metrics['SSIM (Volume Average)']:>10.4f}           |")
    print(f"  |  RMSE:                  {metrics['RMSE']:>10.6f} mm^-1    |")
    print(f"  +------------------------------------------------+")

    recon_hu = convert_to_hounsfield_units(recon_volume)
    print(f"  Reconstructed HU range: [{recon_hu.min():.1f}, {recon_hu.max():.1f}] HU")

    # 5. Export Deliverables (Strictly One Image Per File)
    generated_deliverables = export_all_deliverables(
        recon_mu=recon_volume,
        gt_mu=gt_mu,
        residuals=residuals,
        output_dir=args.output_dir
    )

    print("\n" + "=" * 70)
    print("ALL RECONSTRUCTION PIPELINE TASKS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("Generated Deliverables:")
    for gf in generated_deliverables:
        print(f"  - {gf}")


if __name__ == "__main__":
    main()
