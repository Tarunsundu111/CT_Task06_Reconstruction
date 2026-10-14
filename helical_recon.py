#!/usr/bin/env python3
"""
=============================================================================
End-to-End GPU-Accelerated 3D Helical Cone-Beam CT Reconstruction Pipeline
=============================================================================

Specialized for:
  - Scanner: 120 kVp, 200 mA, 1.0 s exposure (200 mAs)
  - SID: 541.0 mm | SDD: 949.0 mm (Detector radius: 408.0 mm) | Mag: 1.754159
  - Collimation: 40.0 mm | Pitch: 0.516 | Table Feed/Rot: 20.64 mm
  - Views/Rot: 984 | Increment: 0.36585366 deg | Total Views: 16,728 (17 rot)
  - Translation: Y-axis (Ys[k] = k * 0.0209756 mm) | Rotation: XZ plane
  - Display FOV: 500.0 mm (XZ plane) | Scan Range: 350.0 mm (Y-axis)

Solvers implemented:
  - OS-SART (Ordered Subsets Simultaneous Algebraic Reconstruction Technique)
  - CGLS (Conjugate Gradient Least Squares)
  - PyTorch CUDA 3D Total Variation (TV) Regularization
  - Strict Non-Negativity Constraints

Outputs:
  - reconstructed_chest_volume.npy
  - reconstruction_analysis.png
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
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import torch
import astra
from skimage.metrics import structural_similarity as ssim_fn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================================
# Hardware & Scanner Trajectory Constants
# ============================================================================

SID = 541.0                   # Source-to-Isocenter Distance (mm)
SDD = 949.0                   # Source-to-Detector Distance (mm)
DET_RADIUS = 408.0            # Detector radius from isocenter: SDD - SID (mm)
MAGNIFICATION = SDD / SID     # 1.75415896...

COLLIMATION = 40.0            # Collimation at isocenter (mm)
PITCH = 0.516                 # Table feed per rot / Collimation
TABLE_FEED_PER_ROT = 20.64    # mm per 360 deg rotation
ROTATION_TIME_S = 0.28        # seconds per 360 deg rotation
VIEWS_PER_ROT = 984           # views per rotation
DELTA_THETA_DEG = 0.36585366  # angular increment per view (deg)
DELTA_THETA_RAD = np.deg2rad(DELTA_THETA_DEG)
DELTA_Y = TABLE_FEED_PER_ROT / VIEWS_PER_ROT  # 0.020975609756... mm/view

TOTAL_VIEWS = 16728           # 17 rotations * 984 views
TOTAL_ROTATIONS = 17
SCAN_RANGE_Y = 350.0          # mm along longitudinal translation axis
DISPLAY_FOV_XZ = 500.0        # mm in axial imaging plane

# Detector configuration
DET_ROWS = 64                 # elevation / along translation axis (Y)
DET_COLS = 888                # transverse / across rotation plane (XZ)

# Detector pixel pitch at detector plane
DET_PIXEL_V = COLLIMATION * MAGNIFICATION / DET_ROWS  # ~1.09635 mm
HALF_FAN_RAD = np.arctan((DISPLAY_FOV_XZ / 2.0) / SID)
DET_WIDTH_MM = 2.0 * SDD * np.tan(HALF_FAN_RAD)        # ~877.08 mm
DET_PIXEL_U = DET_WIDTH_MM / DET_COLS                 # ~0.98770 mm

# Volume grid dimensions matching ground truth volume (142, 232, 194)
# ASTRA convention: (GridSliceCount, GridRowCount, GridColCount) = (NZ, NY, NX)
NZ = 142                      # Axial depth dimension
NY = 232                      # Longitudinal translation dimension (Y-axis)
NX = 194                      # Axial width dimension

# Physical coordinate bounds (mm)
MIN_X, MAX_X = -DISPLAY_FOV_XZ / 2.0, DISPLAY_FOV_XZ / 2.0  # [-250, 250] mm
MIN_Y, MAX_Y = -SCAN_RANGE_Y / 2.0, SCAN_RANGE_Y / 2.0      # [-175, 175] mm
MIN_Z, MAX_Z = -DISPLAY_FOV_XZ / 2.0, DISPLAY_FOV_XZ / 2.0  # [-250, 250] mm

# Linear attenuation reference baselines at 120 kVp
MU_WATER = 0.0200             # mm^-1
MU_AIR = 0.0000               # mm^-1
MU_BONE = 0.0400              # mm^-1
MU_LUNG = 0.0050              # mm^-1


# ============================================================================
# PyTorch CUDA 3D Total Variation Regularizer
# ============================================================================

def tv3d_proximal_chambolle_gpu(vol_np: np.ndarray, weight: float, n_iter: int = 6) -> np.ndarray:
    """
    Chambolle's dual projection algorithm for 3D Total Variation denoising
    accelerated natively on CUDA with PyTorch.
    Solves: min_x 0.5 * ||x - vol||_2^2 + weight * TV(x)
    """
    if weight <= 0:
        return vol_np

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vol = torch.from_numpy(vol_np).to(device)

    pz = torch.zeros_like(vol)
    py = torch.zeros_like(vol)
    px = torch.zeros_like(vol)
    tau = 1.0 / 6.0

    for _ in range(n_iter):
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

        v = div - vol / weight

        gz = torch.zeros_like(vol)
        gy = torch.zeros_like(vol)
        gx = torch.zeros_like(vol)

        gz[:-1, :, :] = v[1:, :, :] - v[:-1, :, :]
        gy[:, :-1, :] = v[:, 1:, :] - v[:, :-1, :]
        gx[:, :, :-1] = v[:, :, 1:] - v[:, :, :-1]

        denom = 1.0 + tau * torch.sqrt(gz**2 + gy**2 + gx**2 + 1e-12)
        pz = (pz + tau * gz) / denom
        py = (py + tau * gy) / denom
        px = (px + tau * gx) / denom

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

    denoised = vol - weight * div
    return denoised.cpu().numpy()


# ============================================================================
# 1. Preprocessing & Transmission-to-Attenuation Conversion
# ============================================================================

def preprocess_projections(
    raw_projections: np.ndarray,
    air_reference_i0: np.ndarray,
    eps: float = 1e-6,
    batch_size: int = 512
) -> np.ndarray:
    """
    Applies Beer-Lambert log transform on GPU in batches to prevent OOM:
      p = -ln((I + eps) / (I0 + eps))
    Clamps negative values to 0 and replaces NaNs/Infs with 0.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_views = raw_projections.shape[0]
    sinogram = np.zeros(raw_projections.shape, dtype=np.float32)

    i0_torch = torch.from_numpy(air_reference_i0.copy()).to(device)

    for start_idx in range(0, num_views, batch_size):
        end_idx = min(start_idx + batch_size, num_views)
        i_batch = torch.from_numpy(raw_projections[start_idx:end_idx].copy()).to(device)

        # Transmission intensity to line integral attenuation
        ratio = (torch.clamp(i_batch, min=0.0) + eps) / (i0_torch.unsqueeze(0) + eps)
        p = -torch.log(ratio)
        p = torch.clamp(p, min=0.0, max=23.02585)
        p = torch.where(torch.isfinite(p), p, torch.zeros_like(p))

        sinogram[start_idx:end_idx] = p.cpu().numpy()

        del i_batch, p, ratio
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return sinogram


def validate_sinogram_preprocessing(
    sinogram: np.ndarray,
    reference_sinogram_slice32: np.ndarray
) -> None:
    """Validates processed detector row 32 against helical_sinogram_16728_slice32.npy."""
    num_views = sinogram.shape[0]
    processed_row32 = sinogram[:, 32, :]
    ref_row32 = reference_sinogram_slice32[:num_views, :]
    max_err = float(np.max(np.abs(processed_row32 - ref_row32)))
    print(f"  [Validation] Preprocessed row 32 vs reference slice32 max abs error: {max_err:.6e}")
    assert max_err < 1e-4, f"Preprocessed sinogram mismatch! Max error: {max_err}"


# ============================================================================
# 2. Helical Cone-Beam Trajectory & Geometries
# ============================================================================

def build_helical_trajectory_vectors(view_indices: np.ndarray) -> np.ndarray:
    """
    Builds the 12-element ASTRA vector representation for specified view indices:
      [srcX, srcY, srcZ, detX, detY, detZ, uX, uY, uZ, vX, vY, vZ]
    
    Coordinate System:
      - Rotation Plane: XZ plane
        * Xs = SID * cos(theta), Zs = SID * sin(theta)
        * Xd = -DET_RADIUS * cos(theta), Zd = -DET_RADIUS * sin(theta)
      - Translation Axis: Y-axis
        * Ys = Yd = view_index * DELTA_Y
      - Detector Column Vector u (in XZ plane, orthogonal to central ray):
        * Central ray points in (-cos_t, 0, -sin_t)
        * Orthogonal u points in (-sin_t, 0, cos_t) scaled by DET_PIXEL_U
      - Detector Row Vector v (along Y axis):
        * (0, DET_PIXEL_V, 0)
    """
    n_views = len(view_indices)
    vectors = np.zeros((n_views, 12), dtype=np.float32)

    theta = view_indices * DELTA_THETA_RAD
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    table_y = view_indices * DELTA_Y

    # Source position
    vectors[:, 0] = SID * cos_t
    vectors[:, 1] = table_y
    vectors[:, 2] = SID * sin_t

    # Detector center position
    vectors[:, 3] = -DET_RADIUS * cos_t
    vectors[:, 4] = table_y
    vectors[:, 5] = -DET_RADIUS * sin_t

    # Detector column vector u
    vectors[:, 6] = -sin_t * DET_PIXEL_U
    vectors[:, 7] = 0.0
    vectors[:, 8] = cos_t * DET_PIXEL_U

    # Detector row vector v
    vectors[:, 9] = 0.0
    vectors[:, 10] = DET_PIXEL_V
    vectors[:, 11] = 0.0

    return vectors


def create_astra_cone_geometries(view_indices: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Creates ASTRA volume and projection geometries."""
    vol_geom = astra.create_vol_geom(
        NY, NX, NZ,
        MIN_X, MAX_X,
        MIN_Y, MAX_Y,
        MIN_Z, MAX_Z
    )

    vectors = build_helical_trajectory_vectors(view_indices)
    proj_geom = astra.create_proj_geom(
        'cone_vec',
        DET_ROWS,
        DET_COLS,
        vectors
    )

    return vol_geom, proj_geom


# ============================================================================
# 3. Iterative Reconstruction Solver: OS-SART with 3D TV Regularization
# ============================================================================

def reconstruct_os_sart(
    sinogram: np.ndarray,
    view_indices: np.ndarray,
    num_iterations: int = 8,
    num_subsets: int = 16,
    relaxation_lambda: float = 0.6,
    tv_weight: float = 2e-5,
    non_negative: bool = True
) -> Tuple[np.ndarray, List[float]]:
    """
    Performs Ordered Subsets Simultaneous Algebraic Reconstruction Technique (OS-SART):
      f^(k+1) = P_+( f^(k) + lambda * (A_m^T (p_m - A_m f^(k))) / (A_m^T 1) )
    with PyTorch CUDA 3D TV regularization and non-negativity enforcement.
    """
    total_views = len(view_indices)
    n_subsets = min(num_subsets, total_views)

    vol_geom, proj_geom = create_astra_cone_geometries(view_indices)

    # Convert sinogram to ASTRA order: (Rows, Views, Cols)
    sino_astra = np.ascontiguousarray(sinogram.transpose(1, 0, 2))
    p_norm = float(np.linalg.norm(sino_astra))
    if p_norm == 0.0:
        p_norm = 1.0

    print("=" * 70)
    print("STEP 3: GPU OS-SART Reconstruction")
    print(f"  Views: {total_views} | Subsets: {n_subsets} | Iterations: {num_iterations}")
    print(f"  Relaxation lambda: {relaxation_lambda} | TV weight: {tv_weight} | Non-neg: {non_negative}")
    print("=" * 70)

    # Precompute subset geometries and normalization matrices
    t0_pre = time.time()
    subsets_data = []
    ones_vol = np.ones((NZ, NY, NX), dtype=np.float32)

    for s in range(n_subsets):
        sub_rel_idx = np.arange(s, total_views, n_subsets)
        sub_view_indices = view_indices[sub_rel_idx]
        sub_vecs = build_helical_trajectory_vectors(sub_view_indices)
        sub_pg = astra.create_proj_geom('cone_vec', DET_ROWS, DET_COLS, sub_vecs)

        # Row normalization: R_s = 1 / (A_s 1_vol)
        s_id, r = astra.create_sino3d_gpu(ones_vol, sub_pg, vol_geom)
        astra.data3d.delete(s_id)

        # Column normalization: C_s = 1 / (A_s^T 1_sino)
        ones_sub_sino = np.ones((DET_ROWS, len(sub_rel_idx), DET_COLS), dtype=np.float32)
        b_id, c = astra.create_backprojection3d_gpu(ones_sub_sino, sub_pg, vol_geom)
        astra.data3d.delete(b_id)

        R_s = 1.0 / np.maximum(r, 1e-4)
        C_s = 1.0 / np.maximum(c, 1e-4)
        C_s[c < 1e-3] = 0.0

        sub_p = sino_astra[:, sub_rel_idx, :]
        subsets_data.append({
            'pg': sub_pg,
            'rel_idx': sub_rel_idx,
            'p': sub_p,
            'R': R_s,
            'C': C_s
        })

    print(f"  Precomputed {n_subsets} ordered subsets in {time.time() - t0_pre:.2f} s")

    # Initial volume estimate f_0
    f = np.zeros((NZ, NY, NX), dtype=np.float32)
    residuals = []

    for it in range(1, num_iterations + 1):
        t_it_start = time.time()

        for s in range(n_subsets):
            sub = subsets_data[s]

            # Forward project current volume estimate
            s_id, af_sub = astra.create_sino3d_gpu(f, sub['pg'], vol_geom)
            astra.data3d.delete(s_id)

            # Subset residual
            res_sub = sub['p'] - af_sub

            # Backproject normalized residual
            b_id, delta = astra.create_backprojection3d_gpu(res_sub * sub['R'], sub['pg'], vol_geom)
            astra.data3d.delete(b_id)

            # SART update step
            f = f + relaxation_lambda * (delta * sub['C'])

            # Non-negativity constraint
            if non_negative:
                np.maximum(f, 0.0, out=f)

        # PyTorch CUDA 3D Total Variation Regularization
        if tv_weight > 0:
            f = tv3d_proximal_chambolle_gpu(f, weight=tv_weight, n_iter=6)
            if non_negative:
                np.maximum(f, 0.0, out=f)

        # Relative data discrepancy tracking: ||A f^(k) - p||_2 / ||p||_2
        full_s_id, af_full = astra.create_sino3d_gpu(f, proj_geom, vol_geom)
        astra.data3d.delete(full_s_id)

        full_res = sino_astra - af_full
        rel_res = float(np.linalg.norm(full_res) / p_norm)
        residuals.append(rel_res)

        it_time = time.time() - t_it_start
        print(f"  Iteration {it:2d}/{num_iterations:2d} | Relative Residual: {rel_res:.6e} | Time: {it_time:.2f} s | Voxel Max: {f.max():.5f}")

    # Cleanup ASTRA resources
    for sub in subsets_data:
        if 'id' in sub['pg']:
            astra.projector.delete(sub['pg']['id'])
    if 'id' in proj_geom:
        astra.projector.delete(proj_geom['id'])

    return f, residuals


# ============================================================================
# 4. Ground-Truth Benchmarking & Hounsfield Units Conversion
# ============================================================================

def convert_density_to_attenuation(density: np.ndarray, material: np.ndarray) -> np.ndarray:
    """
    Converts ground truth density volume (g/cm^3) to linear attenuation coefficients mu (mm^-1)
    based on calibrated effective energies for 120 kVp beam:
      - Air (mat 0): mu = 0.0000 mm^-1
      - Soft Tissue (mat 1, 1.06 g/cm^3): mu = 0.0205 mm^-1
      - Bone (mat 2, 1.92 g/cm^3): mu = 0.0400 mm^-1
      - Lung (mat 3, 0.26 g/cm^3): mu = 0.0050 mm^-1
    """
    gt_mu = np.zeros_like(density, dtype=np.float32)
    gt_mu[material == 0] = MU_AIR
    gt_mu[material == 1] = 0.0205
    gt_mu[material == 2] = MU_BONE
    gt_mu[material == 3] = MU_LUNG
    return gt_mu


def convert_to_hounsfield_units(mu_volume: np.ndarray) -> np.ndarray:
    """
    Converts linear attenuation coefficients mu (mm^-1) to Hounsfield Units (HU):
      HU = 1000 * (mu - mu_water) / (mu_water - mu_air)
    where mu_water = 0.020 mm^-1, mu_air = 0.0 mm^-1.
    """
    hu = 1000.0 * (mu_volume - MU_WATER) / (MU_WATER - MU_AIR + 1e-12)
    return hu


def compute_reconstruction_metrics(gt_mu: np.ndarray, recon_mu: np.ndarray) -> Dict[str, float]:
    """Computes PSNR, SSIM, and RMSE between ground truth and reconstruction."""
    # RMSE
    mse = float(np.mean((gt_mu - recon_mu) ** 2))
    rmse = float(np.sqrt(mse))

    # PSNR
    data_max = float(np.max(gt_mu))
    data_min = float(np.min(gt_mu))
    data_range = max(data_max - data_min, 1e-6)
    psnr = float(20.0 * np.log10(data_max / (rmse + 1e-12))) if rmse > 0 else float("inf")

    # SSIM on central axial slice
    mid_y = gt_mu.shape[1] // 2
    gt_axial = gt_mu[:, mid_y, :]
    rc_axial = recon_mu[:, mid_y, :]
    axial_ssim = float(ssim_fn(gt_axial, rc_axial, data_range=data_range))

    # Volume-averaged SSIM across sampled slices
    ssim_list = []
    for iy in range(0, gt_mu.shape[1], max(1, gt_mu.shape[1] // 16)):
        s_gt = gt_mu[:, iy, :]
        s_rc = recon_mu[:, iy, :]
        if s_gt.max() > 1e-5:
            ssim_list.append(ssim_fn(s_gt, s_rc, data_range=data_range))
    vol_ssim = float(np.mean(ssim_list)) if ssim_list else axial_ssim

    return {
        "RMSE": rmse,
        "PSNR (dB)": psnr,
        "SSIM (Axial Slice)": axial_ssim,
        "SSIM (Volume Average)": vol_ssim,
        "MSE": mse
    }


# ============================================================================
# 5. Output Deliverables & Comprehensive Visualization
# ============================================================================

def export_reconstruction_analysis_figure(
    recon_mu: np.ndarray,
    gt_mu: np.ndarray,
    gt_axial_ref: np.ndarray,
    residuals: List[float],
    metrics: Dict[str, float],
    output_path: str = "reconstruction_analysis.png"
) -> str:
    """
    Exports a comprehensive multi-panel publication-grade figure containing:
      1. Central axial slice (XZ) reconstruction next to voxelized_chest_axial_slice.npy
      2. Orthogonal coronal (XY) and sagittal (YZ) cross-sections
      3. Absolute error map (|Reconstruction - Ground Truth|)
      4. 1D profile transect across the lung/soft-tissue/bone boundary
      5. Cost function convergence plot across iterations
    """
    nz, ny, nx = recon_mu.shape
    mid_z = nz // 2
    mid_y = ny // 2
    mid_x = nx // 2

    recon_hu = convert_to_hounsfield_units(recon_mu)
    gt_hu = convert_to_hounsfield_units(gt_mu)

    # 1. Central cross sections
    rc_axial_hu = recon_hu[:, mid_y, :]      # (NZ, NX) = (142, 194)
    gt_axial_hu = gt_hu[:, mid_y, :]        # (NZ, NX) = (142, 194)
    rc_coronal_hu = recon_hu[mid_z, :, :]    # (NY, NX) = (232, 194)
    rc_sagittal_hu = recon_hu[:, :, mid_x]   # (NZ, NY) = (142, 232)

    # Absolute error on central axial slice
    abs_error_axial = np.abs(recon_mu[:, mid_y, :] - gt_mu[:, mid_y, :])

    # 1D profile transect crossing Lung -> Soft Tissue -> Bone
    # Piecewise line: Lung at (row 60, col 40) -> Tissue at (row 60, col 100) -> Bone at (row 110, col 100)
    p1_coords = [(60, c) for c in range(40, 101)]
    p2_coords = [(r, 100) for r in range(61, 115)]
    transect_coords = p1_coords + p2_coords

    profile_gt_mu = np.array([gt_mu[r, mid_y, c] for r, c in transect_coords])
    profile_rc_mu = np.array([recon_mu[r, mid_y, c] for r, c in transect_coords])
    profile_dist_mm = np.arange(len(transect_coords)) * 2.5773  # mm approximate pitch

    fig = plt.figure(figsize=(22, 15), dpi=200)
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.30, left=0.06, right=0.95, top=0.92, bottom=0.06)

    # Panel 1: Reconstructed Axial Slice (HU)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(rc_axial_hu, origin="lower", cmap="gray", vmin=-1000, vmax=1000, extent=[-250, 250, -250, 250])
    ax1.set_title("1. Reconstructed Central Axial Slice (XZ, HU)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X (mm)", fontsize=10)
    ax1.set_ylabel("Z (mm)", fontsize=10)
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label("HU", fontsize=10)

    # Panel 2: Ground Truth Axial Reference (Density / HU)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(gt_axial_hu, origin="lower", cmap="gray", vmin=-1000, vmax=1000, extent=[-250, 250, -250, 250])
    ax2.set_title("2. Ground Truth Reference Axial (voxelized_chest)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("X (mm)", fontsize=10)
    ax2.set_ylabel("Z (mm)", fontsize=10)
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label("HU", fontsize=10)

    # Panel 3: Absolute Error Map
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(abs_error_axial, origin="lower", cmap="hot", vmin=0.0, vmax=0.015, extent=[-250, 250, -250, 250])
    ax3.set_title(f"3. Absolute Error Map |Recon - GT|\n(Max Diff: {abs_error_axial.max():.4e} mm^-1)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("X (mm)", fontsize=10)
    ax3.set_ylabel("Z (mm)", fontsize=10)
    cb3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cb3.set_label(r"|$\Delta\mu$| (mm$^{-1}$)", fontsize=10)

    # Panel 4: Orthogonal Coronal Cross-Section (XY)
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(rc_coronal_hu, origin="lower", cmap="gray", vmin=-1000, vmax=1000, extent=[-250, 250, -175, 175], aspect="auto")
    ax4.set_title("4. Orthogonal Coronal Cross-Section (XY, HU)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("X (mm)", fontsize=10)
    ax4.set_ylabel("Y - Translation Axis (mm)", fontsize=10)
    cb4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cb4.set_label("HU", fontsize=10)

    # Panel 5: Orthogonal Sagittal Cross-Section (YZ)
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(rc_sagittal_hu.T, origin="lower", cmap="gray", vmin=-1000, vmax=1000, extent=[-250, 250, -175, 175], aspect="auto")
    ax5.set_title("5. Orthogonal Sagittal Cross-Section (YZ, HU)", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Z (mm)", fontsize=10)
    ax5.set_ylabel("Y - Translation Axis (mm)", fontsize=10)
    cb5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cb5.set_label("HU", fontsize=10)

    # Panel 6: Raw Input Reference Axial Slice
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(gt_axial_ref, origin="lower", cmap="viridis", extent=[-250, 250, -250, 250])
    ax6.set_title("6. Raw Reference Density Slice (g/cm^3)", fontsize=11, fontweight="bold")
    ax6.set_xlabel("X (mm)", fontsize=10)
    ax6.set_ylabel("Z (mm)", fontsize=10)
    cb6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    cb6.set_label("Density (g/cm$^3$)", fontsize=10)

    # Panel 7: 1D Profile Transect (Lung/Soft-Tissue/Bone Boundary)
    ax7 = fig.add_subplot(gs[2, 0:2])
    ax7.plot(profile_dist_mm, profile_gt_mu, "b-", linewidth=2.2, label="Ground Truth mu")
    ax7.plot(profile_dist_mm, profile_rc_mu, "r--", linewidth=1.8, label="OS-SART Recon mu")
    ax7.set_title("7. 1D Attenuation Profile across Lung / Soft Tissue / Bone Boundary", fontsize=11, fontweight="bold")
    ax7.set_xlabel("Distance along Transect Path (mm)", fontsize=10)
    ax7.set_ylabel(r"Linear Attenuation $\mu$ (mm$^{-1}$)", fontsize=10)
    ax7.grid(True, linestyle=":", alpha=0.6)
    ax7.legend(loc="upper left", fontsize=10)
    ax7.set_ylim([-0.005, 0.050])

    # Annotation of materials
    ax7.axhline(y=MU_LUNG, color="cyan", linestyle=":", alpha=0.7, label="Lung baseline")
    ax7.axhline(y=MU_WATER, color="green", linestyle=":", alpha=0.7, label="Water/Tissue baseline")
    ax7.axhline(y=MU_BONE, color="magenta", linestyle=":", alpha=0.7, label="Bone baseline")

    # Panel 8: Cost Function Convergence & Metrics Banner
    ax8 = fig.add_subplot(gs[2, 2])
    if residuals:
        iters = np.arange(1, len(residuals) + 1)
        ax8.semilogy(iters, residuals, "b-o", markersize=6, linewidth=1.8, label=r"$\|Af^{(k)} - p\|_2 / \|p\|_2$")
        ax8.set_xlabel("Iteration Number", fontsize=10)
        ax8.set_ylabel("Relative Data Discrepancy", fontsize=10)
        ax8.grid(True, which="both", linestyle=":", alpha=0.6)
        ax8.set_title("8. Cost Function Convergence", fontsize=11, fontweight="bold")
        ax8.legend(loc="upper right", fontsize=9)

    # Metrics banner
    metrics_text = (
        f"Quantitative Image Metrics:\n"
        f"  PSNR: {metrics['PSNR (dB)']:.2f} dB\n"
        f"  SSIM (Axial): {metrics['SSIM (Axial Slice)']:.4f}\n"
        f"  SSIM (Volume): {metrics['SSIM (Volume Average)']:.4f}\n"
        f"  RMSE: {metrics['RMSE']:.6f} mm^-1"
    )
    ax8.text(
        0.05, 0.05, metrics_text,
        transform=ax8.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef2f7", edgecolor="#8fa4b8", alpha=0.95)
    )

    plt.suptitle(
        "3D Helical Cone-Beam CT Iterative Reconstruction & Quantitative Validation\n"
        f"120 kVp | SID={SID} mm | SDD={SDD} mm | Pitch={PITCH} | Feed/Rot={TABLE_FEED_PER_ROT} mm | Grid: {NZ}x{NY}x{NX}",
        fontsize=14, fontweight="bold", y=0.98
    )

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Output] Multi-panel analysis figure exported to: {output_path}")
    return output_path


# ============================================================================
# Main End-to-End Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="3D Helical CT Iterative Reconstruction Pipeline")
    parser.add_argument("--prototype", action="store_true", default=False,
                        help="Run in prototype mode (1 rotation = 984 views)")
    parser.add_argument("--views", type=int, default=None,
                        help="Custom number of views to reconstruct")
    parser.add_argument("--iterations", type=int, default=8,
                        help="Number of OS-SART iterations")
    parser.add_argument("--subsets", type=int, default=16,
                        help="Number of ordered subsets")
    parser.add_argument("--relaxation", type=float, default=0.6,
                        help="SART relaxation parameter lambda")
    parser.add_argument("--tv_weight", type=float, default=2e-5,
                        help="PyTorch CUDA 3D TV regularization weight")
    args = parser.parse_args()

    print("=" * 70)
    print("GPU-Accelerated 3D Helical Cone-Beam CT Reconstruction Pipeline")
    print("=" * 70)

    # Check CUDA device
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  CUDA Device: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("  CUDA Device: CPU fallback")

    # 1. Load Data Files
    print("\n" + "=" * 70)
    print("STEP 1: Loading Input Data Files")
    print("=" * 70)
    data_dir = Path(".")
    
    print("  Loading air_reference_I0.npy...")
    i0 = np.load(data_dir / "air_reference_I0.npy").astype(np.float32)
    print(f"    Air reference I0: shape={i0.shape}, range=[{i0.min():.1f}, {i0.max():.1f}]")

    print("  Loading helical_sinogram_16728_slice32.npy...")
    sino32_ref = np.load(data_dir / "helical_sinogram_16728_slice32.npy").astype(np.float32)
    print(f"    Reference slice32: shape={sino32_ref.shape}, range=[{sino32_ref.min():.4f}, {sino32_ref.max():.4f}]")

    print("  Loading ground truth chest phantom volumes...")
    gt_density = np.load(data_dir / "voxelized_chest_density.npy").astype(np.float32)
    gt_material = np.load(data_dir / "voxelized_chest_material.npy").astype(np.int32)
    gt_axial_ref = np.load(data_dir / "voxelized_chest_axial_slice.npy").astype(np.float32)
    print(f"    Density volume: shape={gt_density.shape}")
    print(f"    Material volume: shape={gt_material.shape}, materials={np.unique(gt_material)}")
    print(f"    Axial reference: shape={gt_axial_ref.shape}")

    print("  Loading helical_projections_3d_16728.npy (memory-mapped)...")
    raw_projections = np.load(data_dir / "helical_projections_3d_16728.npy", mmap_mode="r")
    total_dataset_views = raw_projections.shape[0]
    print(f"    Projections dataset: shape={raw_projections.shape}")

    # Determine views subset (Prototype mode vs Full mode)
    if args.views is not None:
        num_views_to_run = args.views
    elif args.prototype:
        num_views_to_run = VIEWS_PER_ROT  # 984 views (1 full rotation)
    else:
        # Full scan execution
        num_views_to_run = total_dataset_views

    print(f"\n  [Execution Mode] Reconstructing {num_views_to_run} views (Rotations: {num_views_to_run / VIEWS_PER_ROT:.2f})")
    view_indices = np.arange(num_views_to_run, dtype=np.int32)

    # 2. Preprocessing & Transmission-to-Attenuation Conversion
    print("\n" + "=" * 70)
    print("STEP 2: Beer-Lambert Preprocessing on GPU")
    print("=" * 70)
    t0_prep = time.time()
    # Stream projections to convert
    sinogram = preprocess_projections(raw_projections[:num_views_to_run], i0)
    print(f"  Processed {num_views_to_run} projection views in {time.time() - t0_prep:.2f} s")
    print(f"  Attenuation sinogram shape: {sinogram.shape}, range=[{sinogram.min():.4f}, {sinogram.max():.4f}]")

    # Validate row 32
    validate_sinogram_preprocessing(sinogram, sino32_ref)

    # 3. Iterative Reconstruction: OS-SART
    t0_recon = time.time()
    recon_volume, residuals = reconstruct_os_sart(
        sinogram=sinogram,
        view_indices=view_indices,
        num_iterations=args.iterations,
        num_subsets=args.subsets,
        relaxation_lambda=args.relaxation,
        tv_weight=args.tv_weight,
        non_negative=True
    )
    print(f"\n  Reconstruction completed in {time.time() - t0_recon:.2f} s")
    print(f"  Reconstructed volume shape: {recon_volume.shape}, range=[{recon_volume.min():.6f}, {recon_volume.max():.6f}] mm^-1")

    # 4. Benchmarking & Validation
    print("\n" + "=" * 70)
    print("STEP 4: Ground-Truth Benchmarking & Validation")
    print("=" * 70)
    gt_mu = convert_density_to_attenuation(gt_density, gt_material)
    metrics = compute_reconstruction_metrics(gt_mu, recon_volume)

    print(f"\n  +------------------------------------------------+")
    print(f"  |  Quantitative Benchmarking Metrics             |")
    print(f"  +------------------------------------------------+")
    print(f"  |  PSNR:                 {metrics['PSNR (dB)']:>10.2f} dB         |")
    print(f"  |  SSIM (Axial Slice):   {metrics['SSIM (Axial Slice)']:>10.4f}            |")
    print(f"  |  SSIM (Volume Average):{metrics['SSIM (Volume Average)']:>10.4f}            |")
    print(f"  |  RMSE:                 {metrics['RMSE']:>10.6f} mm^-1     |")
    print(f"  +------------------------------------------------+")

    recon_hu = convert_to_hounsfield_units(recon_volume)
    print(f"\n  Reconstructed HU range: [{recon_hu.min():.1f}, {recon_hu.max():.1f}] HU")

    # 5. Save Deliverables
    print("\n" + "=" * 70)
    print("STEP 5: Exporting Deliverables")
    print("=" * 70)
    out_vol_path = "reconstructed_chest_volume.npy"
    np.save(out_vol_path, recon_volume)
    print(f"  [Output] Reconstructed 3D volume saved to: {out_vol_path} (Size: {os.path.getsize(out_vol_path)/1e6:.2f} MB)")

    out_fig_path = "reconstruction_analysis.png"
    export_reconstruction_analysis_figure(
        recon_mu=recon_volume,
        gt_mu=gt_mu,
        gt_axial_ref=gt_axial_ref,
        residuals=residuals,
        metrics=metrics,
        output_path=out_fig_path
    )

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE - ALL DELIVERABLES GENERATED")
    print("=" * 70)


if __name__ == "__main__":
    main()
