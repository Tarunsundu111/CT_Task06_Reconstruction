# Task [X]: 3D Cone-Beam CT Image Reconstruction

## 1. Module Overview (What Our Team Is Building)

This module performs **3D CT image reconstruction from cone-beam projection data**. The input to the reconstruction module is not an already existing image. Instead, it receives projection measurements collected by the CT scanner from different gantry angles. These projection measurements contain information about the internal structure of the scanned object.

The reconstruction algorithm used in this module is an **Algebraic Reconstruction Technique (ART)**, which is an iterative reconstruction method. The algorithm starts with an initial estimate of the 3D volume and repeatedly compares the calculated projection from the current volume with the measured projection data. The difference between them is treated as an error, and this error is used to update the voxels crossed by the corresponding X-ray ray.

The reconstruction process uses a **cone-beam forward projection model** to calculate the expected detector measurements and an ART-based correction step to update the volume. The process is repeated for multiple projection views and iterations until the reconstructed volume approaches the measured projection data.

---

## 2. Inputs & Outputs

### Inputs

* **Projection / Sinogram Data**

  * Shape: `(total_views, detector_rows, detector_channels)`
  * Contains the measured CT projection values obtained from different gantry angles.
  * Example:

    ```text
    (180, 16, 1024)
    ```

* **Gantry Angles**

  * Shape: `(total_views,)`
  * Contains the angular position of the X-ray source for each projection view.
  * The angles are required to determine the position of the X-ray source and detector for each view.

* **CT Geometry / Configuration**

  * Contains the physical parameters required for cone-beam reconstruction, such as:

    * Source-to-Object Distance (SOD)
    * Source-to-Detector Distance (SDD)
    * Detector pixel size
    * Voxel size
    * Reconstruction volume dimensions

* **ART Parameters**

  * Number of reconstruction iterations.
  * Relaxation parameter used to control the magnitude of each ART correction.

### Outputs

* **Reconstructed 3D CT Volume**

  * Shape:

    ```text
    (Nz, Ny, Nx)
    ```
  * Represents the estimated attenuation distribution inside the scanned object.

* **Reconstructed CT Slice**

  * A 2D slice from the reconstructed 3D volume can be displayed for visualization.

* **Saved Reconstruction**

  * The reconstructed volume is saved as:

    ```text
    ART_reconstructed_volume.npy
    ```

---

## 3. Current Progress

* [x] Defined the mathematical model for CT reconstruction.
* [x] Implemented a 3D cone-beam geometry model.
* [x] Implemented X-ray source position calculation.
* [x] Implemented detector coordinate calculation.
* [x] Implemented ray-volume intersection calculation.
* [x] Implemented trilinear interpolation.
* [x] Implemented cone-beam forward projection.
* [x] Implemented ART-based iterative reconstruction.
* [x] Implemented an initial 3D test phantom.
* [x] Added synthetic projection generation for testing.
* [x] Added support for loading projection data from `.npy` files.
* [x] Added support for loading gantry angles from `.npy` files.
* [x] Added support for loading CT configuration parameters from a JSON file.
* [x] Added reconstructed volume saving.
* [x] Added visualization of the central reconstructed slice.
* [ ] Test the reconstruction using the actual CT scanner projection data.
* [ ] Verify the real CT geometry and calibration parameters.
* [ ] Optimize the implementation for large projection datasets.
* [ ] Validate reconstructed images against a known reference or phantom.

---

## 4. How We Will Validate Results

The reconstruction will be validated at different levels.

### Data Validation

* Check that the projection data has the expected dimensions:

  ```text
  (total_views, detector_rows, detector_channels)
  ```
* Check that the number of gantry angles matches the number of projection views.
* Check that the CT geometry parameters are available and have physically meaningful values.
* Check for `NaN` and infinite values in the input projection data.

### Algorithm Validation

* Generate projection data from a known synthetic 3D phantom.
* Use the generated projection data as input to the ART reconstruction.
* Compare the reconstructed volume with the original synthetic phantom.
* Monitor the difference between measured and estimated projections during iterations.

### Reconstruction Validation

* Check reconstructed volume dimensions.
* Check for `NaN` or infinite values.
* Check that the reconstructed attenuation values remain physically meaningful.
* Visualize central axial slices and other relevant slices.
* Compare reconstructed structures with the known test phantom.
* For real CT data, compare reconstructed image characteristics with the expected CT system geometry and available reference data.

### Mathematical Model

The ART reconstruction is based on the algebraic system:

$$
Ax = y
$$

where:

* `x` = unknown 3D CT volume
* `A` = cone-beam projection operator
* `y` = measured projection data

For an individual ray, ART calculates the projection error:

$$
e_i = y_i - A_i x
$$

and uses this error to update the voxels contributing to that ray.

A simplified ART update is:

$$
x^{k+1}
=
x^k
+
\lambda
\frac{y_i-A_i x^k}
{\|A_i\|^2}
A_i^T
$$

where `λ` is the relaxation parameter.

---

## 5. Code File

Our main Python reconstruction code can be found in this folder under:

```text
art_cone_beam_reconstruction.py
```

### Supporting Input Files

For real CT data, the reconstruction module expects files such as:

```text
sinogram.npy
angles.npy
ct_config.json
```

Example project structure:

```text
CT-Image-Reconstruction/
│
├── README.md
│
├── art_cone_beam_reconstruction.py
│
├── sinogram.npy
│
├── angles.npy
│
└── ct_config.json
```

> **Note:** The current implementation is an educational, from-scratch implementation of cone-beam ART reconstruction using NumPy. Reconstruction speed and accuracy for real clinical-sized CT datasets will depend strongly on the actual scanner geometry, detector calibration, sampling, and optimization of the ray-tracing implementation.
