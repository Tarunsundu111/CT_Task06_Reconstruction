# 3D Helical Cone-Beam CT Reconstruction Using Iterative Methods

## 1. Overview

This project implements a **3D iterative reconstruction pipeline for helical cone-beam Computed Tomography (CT)** using Python, PyTorch, and CUDA.

The system reconstructs a 3D chest volume from helical CT projection measurements. It combines Beer–Lambert preprocessing, GPU-based forward and backprojection, Ordered Subsets Simultaneous Algebraic Reconstruction Technique (OS-SART), and 3D Total Variation (TV) regularization.

The reconstructed volume is evaluated against a reference chest phantom using quantitative metrics and visual comparisons.

## 2. Key Features

- 3D helical cone-beam CT geometry modeling.
- Beer–Lambert transformation of measured intensities into line-integral projections.
- GPU-accelerated forward projection using trilinear interpolation.
- Backprojection with geometric weighting.
- Iterative reconstruction using OS-SART.
- 3D Total Variation regularization for noise reduction.
- Quantitative evaluation using RMSE, PSNR, and SSIM.
- Axial, coronal, sagittal, and maximum intensity projection (MIP) visualizations.
- Reconstruction volume export in NumPy format.

## 3. Reconstruction Workflow

The overall workflow is:

1. Load projection measurements and reference phantom data.
2. Validate input files and acquisition settings.
3. Convert measured intensities into line-integral projections.
4. Initialize the 3D reconstruction volume.
5. Perform forward projection to estimate projection measurements.
6. Calculate the difference between measured and estimated projections.
7. Backproject the residuals and update the reconstruction using OS-SART.
8. Apply non-negativity constraints and 3D TV regularization.
9. Repeat the iterative updates until the configured iteration count is reached.
10. Evaluate the reconstructed volume and save numerical and visual results.

## 4. Mathematical Formulation

### 4.1 Beer–Lambert Law

When X-rays pass through an object, their intensity decreases due to attenuation. The relationship is described by:

$$
I = I_0 e^{-\int_L \mu(\mathbf{r})\,d\ell}
$$

Taking the negative logarithm converts the measured intensity into projection data:

$$
p = -\ln\left(\frac{I}{I_0}\right)
$$

Where:

- `I0` = incident X-ray intensity
- `I` = measured X-ray intensity
- `μ` = linear attenuation coefficient
- `p` = projection measurement

**Implementation:** The code divides the measured intensities by the air-reference intensity, applies the negative logarithm, and handles invalid or extreme values.

### 4.2 Forward Projection Model

The projection value along an X-ray path is approximated by summing the attenuation values sampled along that path:

$$
p_i^{\text{pred}} \approx \sum_{j=1}^{N} \mu_j \Delta s
$$

The complete CT system can be represented as:

$$
\mathbf{p} = \mathbf{A}\mathbf{f} + \boldsymbol{\epsilon}
$$

Where:

- `p` = measured projection data
- `A` = forward-projection operator
- `f` = unknown 3D attenuation volume
- `ε` = measurement and modeling error

**Implementation:** The forward projector samples the current 3D volume along the X-ray paths using trilinear interpolation.

### 4.3 Projection Residual

The difference between measured and predicted projections is calculated as:

$$
\mathbf{r}_s = \mathbf{p}_s - \mathbf{A}_s\mathbf{f}
$$

Where:

- `p_s` = measured projections for a subset of views
- `A_s f` = predicted projections
- `r_s` = projection residual

**Implementation:** The residual indicates how much the current reconstruction differs from the measured projection data. It is used to calculate the next volume update.

### 4.4 OS-SART Reconstruction

Ordered Subsets Simultaneous Algebraic Reconstruction Technique (OS-SART) is an iterative method used to reconstruct the 3D attenuation volume from projection measurements.

The reconstruction is updated using the difference between measured and predicted projections.

```text
Residual = Measured Projections - Predicted Projections

f_new = f_old + lambda * C_s * A_s_transpose
                     * R_s * Residual
```

Where:

- `f_old` = current reconstructed volume.
- `f_new` = updated reconstructed volume.
- `A_s_transpose` = backprojection operator for the current subset.
- `R_s` and `C_s` = normalization factors.
- `lambda` = relaxation parameter.
- `s` = current subset of projection views.

The algorithm divides the projection views into subsets and updates the reconstruction using these subsets iteratively.

**Implementation settings:**

- Number of iterations: `8`
- Number of subsets: `16`
- Relaxation parameter: `0.6`

These are the default settings and can be modified through command-line arguments.

### 4.5 Total Variation (TV) Regularization

Iterative reconstruction may produce noise and unwanted variations between neighboring voxels. Total Variation (TV) regularization helps reduce these variations while preserving important boundaries.

The 3D TV function is defined as:

```text
TV(f) = Sum over all voxels of
        sqrt((Dx f)^2 + (Dy f)^2 + (Dz f)^2)
```

Where:

- `Dx f` = intensity difference along the X direction.
- `Dy f` = intensity difference along the Y direction.
- `Dz f` = intensity difference along the Z direction.

The regularization objective is:

```text
f* = argmin_u [ 0.5 * ||u - f||^2 + alpha * TV(u) ]
```

The first term keeps the regularized volume close to the current reconstruction. The second term penalizes unnecessary spatial variations.

**Implementation:** The code uses a Chambolle-style iterative procedure for TV regularization. The default TV weight is `2e-5`.

### 4.6 Reconstruction Evaluation

The reconstructed volume is compared with the reference phantom using quantitative metrics.

**Mean Squared Error (MSE)**

```text
MSE = (1/N) * Sum[(f_i - f_hat_i)^2]
```

MSE measures the average squared difference between reconstructed and reference voxel values.

**Root Mean Square Error (RMSE)**

```text
RMSE = sqrt(MSE)
```

A lower RMSE generally indicates a smaller reconstruction error.

**Peak Signal-to-Noise Ratio (PSNR)**

```text
PSNR = 10 * log10(MAX^2 / MSE)
```

`MAX` is the peak reference attenuation value used by the implementation. A higher PSNR generally indicates lower reconstruction error.

**Structural Similarity Index (SSIM)**

SSIM evaluates similarity between reconstructed and reference images by comparing luminance, contrast, and structure. Values closer to `1` generally indicate greater structural similarity.

**Implementation:** The script calculates RMSE, PSNR, axial SSIM, and sampled-volume SSIM.

### 4.7 Complete Reconstruction Workflow

The reconstruction follows these steps:

1. Convert measured X-ray intensities into projection data.
2. Forward-project the current 3D attenuation volume.
3. Calculate the difference between measured and predicted projections.
4. Backproject the residual and update the volume using OS-SART.
5. Apply non-negativity constraints and TV regularization.
6. Evaluate the reconstructed volume against the reference phantom.

These steps are repeated for the specified number of iterations to improve agreement between the measured projections and the reconstructed volume.

## 5. Acquisition and Reconstruction Settings

The following are the geometry and default settings implemented in the script. Command-line arguments may override selected reconstruction parameters.

| Parameter | Setting |
|---|---:|
| Source-to-isocenter distance (SID) | 541 mm |
| Source-to-detector distance (SDD) | 949 mm |
| Detector dimensions | 64 × 888 |
| Detector pitch | 0.516 mm |
| Views per rotation | 984 |
| Number of rotations | 17 |
| Full acquisition views | 16,728 |
| Reconstruction volume | 142 × 232 × 194 |
| Default ray-sampling step | 1.5 mm |
| Default iterations | 8 |
| Default subsets | 16 |
| Relaxation parameter | 0.6 |
| TV regularization weight | 2 × 10⁻⁵ |
| Processing device | CUDA-compatible GPU |

**Important:** The default prototype uses 984 views. To request the full 16,728-view acquisition, specify `--views 16728`.

## 6. Requirements

- Python 3
- NumPy
- PyTorch with CUDA support
- Matplotlib
- NVIDIA GPU with a compatible CUDA setup

Install the Python dependencies:

```bash
pip install numpy matplotlib
```

Install a PyTorch build compatible with your system's CUDA environment using the official PyTorch installation instructions.

The reconstruction script requires CUDA and does not provide a CPU fallback.

## 7. Input Data

Place the required NumPy files in a data directory.

| File | Purpose |
|---|---|
| `helical_projections_3d_16728.npy` | Measured projection intensities |
| `air_reference_I0.npy` | Incident intensity reference |
| `helical_sinogram_16728_slice32.npy` | Reference sinogram for preprocessing validation |
| `voxelized_chest_material.npy` | Reference phantom material labels |
| `voxelized_chest_density.npy` | Reference density data loaded by the benchmark |
| `helical_sinogram_raw_16728_slice32.npy` | Raw reference sinogram file checked by the script |

The expected projection shape is views × detector rows × detector columns, with detector dimensions of 64 × 888. The reference phantom volume is expected to use the shape `(142, 232, 194)` in `(Z, Y, X)` order.

Check the actual files and dimensions in your dataset before running the reconstruction.

## 8. How to Run

Save the script as `helical_iterative_recon_3d.py`.

### Prototype reconstruction

Run the default prototype using 984 views:

```bash
python helical_iterative_recon_3d.py
```

### Full acquisition

```bash
python helical_iterative_recon_3d.py --views 16728
```

### Custom reconstruction settings

```bash
python helical_iterative_recon_3d.py \
    --views 984 \
    --iterations 8 \
    --subsets 16 \
    --relaxation 0.6 \
    --tv_weight 2e-5 \
    --step 1.5 \
    --data_dir ./data \
    --output_dir ./outputs
```

Available options include:

- `--prototype`: use the prototype configuration.
- `--views`: number of projection views.
- `--iterations`: number of reconstruction iterations.
- `--subsets`: number of ordered subsets.
- `--relaxation`: OS-SART relaxation parameter.
- `--tv_weight`: TV regularization weight.
- `--step`: ray-sampling step in millimeters.
- `--data_dir`: input data directory.
- `--output_dir`: output directory.

Use `--help` to inspect the command-line options supported by the script.

## 9. Output Files

The script saves the reconstructed volume and visualizations in the configured output directory.

| Output | Description |
|---|---|
| `reconstructed_chest_volume.npy` | Reconstructed 3D attenuation volume |
| `01_reconstructed_axial_slice.png` | Reconstructed axial slice |
| `02_ground_truth_axial_slice.png` | Reference axial slice |
| `03_absolute_error_axial_map.png` | Absolute reconstruction error |
| `04_reconstructed_coronal_slice.png` | Coronal reconstruction |
| `05_reconstructed_sagittal_slice.png` | Sagittal reconstruction |
| `06_3d_mip_coronal_view.png` | Coronal maximum intensity projection |
| `07_3d_mip_axial_view.png` | Axial maximum intensity projection |
| `08_convergence_loss_curve.png` | Logged reconstruction discrepancy indicator |

The exact output files depend on successful execution of the script.

## 10. Results and Evaluation

The reconstruction is assessed through numerical metrics and visual comparisons with the reference phantom.

Record the values reported by your actual execution:

| Metric | Result |
|---|---|
| Preprocessing validation error | Fill from console output |
| RMSE | Fill from console output |
| PSNR | Fill from console output |
| Axial SSIM | Fill from console output |
| Sampled-volume SSIM | Fill from console output |
| Total runtime | Measure and record |
| GPU model | Record your hardware |

### Recommended Visual Comparisons

1. Compare the reconstructed axial slice with the reference axial slice using the same display range.
2. Inspect the absolute error map to identify reconstruction inaccuracies.
3. Examine coronal and sagittal slices for consistency across the volume.
4. Use MIP images to inspect large-scale 3D structures.
5. Review the logged discrepancy curve to understand how the reconstruction evolves over iterations.

Do not report numerical values until they have been obtained from an actual run.

## 11. Limitations

- Reconstruction quality depends on the accuracy of the acquisition geometry and input measurements.
- The custom forward and backprojection operators should be validated for geometric consistency and approximate adjointness.
- The preprocessing validation uses a particular detector row and does not independently validate every projection.
- The material-based attenuation mapping uses fixed attenuation values; the loaded density data is not used directly to determine those values.
- The logged discrepancy curve is an indicator accumulated from subset updates, rather than a fresh full-data residual evaluation at every iteration.
- A synthetic phantom evaluation does not establish clinical validity.
- Full-view reconstruction can require substantial GPU memory and computation time.

## 12. Conclusion

This project implements a GPU-based 3D helical cone-beam CT reconstruction pipeline using OS-SART and 3D TV regularization. It integrates projection preprocessing, iterative reconstruction, quantitative evaluation, and 3D visualization into a single workflow.

The implementation provides a foundation for studying iterative CT reconstruction. Further validation of the projection geometry, forward/backprojection consistency, and reconstruction accuracy is necessary before drawing conclusions about its performance on real clinical data.

## 13. References

1. A. C. Kak and M. Slaney, *Principles of Computerized Tomographic Imaging*. SIAM, 2001.
2. A. H. Andersen and A. C. Kak, “Simultaneous algebraic reconstruction technique (SART): A superior implementation of the ART algorithm,” *Ultrasonic Imaging*, vol. 6, no. 1, pp. 81–94, 1984.
3. L. A. Feldkamp, L. C. Davis, and J. W. Kress, “Practical cone-beam algorithm,” *Journal of the Optical Society of America A*, vol. 1, no. 6, pp. 612–619, 1984.
4. A. Chambolle, “An algorithm for total variation minimization and applications,” *Journal of Mathematical Imaging and Vision*, vol. 20, pp. 89–97, 2004.
5. PyTorch documentation, including CUDA support and `torch.nn.functional.grid_sample`.

---

**Project:** 3D Helical Iterative CT Reconstruction  
**Main script:** `helical_iterative_recon_3d.py`  
**Language:** Python  
**Framework:** PyTorch  
**Acceleration:** CUDA
