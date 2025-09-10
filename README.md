# iMNF: A Spatially Invariant MNF Denoiser

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the Python implementation of standard and patch-wise Spatially Invariant Minimum Noise Fraction (iMNF) denoising algorithm, including the same implementations of the traditional MNF methods:

## Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone [https://github.com/dougalferg/iMNF-denoising.git](https://github.com/dougalferg/iMNF-denoising.git)
    ```

2.  Navigate into the project directory:
    ```bash
    cd iMNF-denoising
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Install the `imnf` package in editable mode:
    ```bash
    pip install -e .
    ```

## Quick Start

Here is a simple example of how to use the `imnf_denoise` function on a sample dataset.

```python
import numpy as np
from imnf.mnf_invariant import imnf_denoise

# 1. Create a mock hyperspectral image (100x100 pixels, 425 bands)
print("Creating mock hyperspectral data...")
mock_image = np.random.rand(100, 100, 425)
mock_wavenumbers = np.linspace(950, 1800, 425)

# 2. Apply the iMNF denoiser
print("Applying iMNF denoising...")
denoised_image, _, _ = imnf_denoise(
    mock_image,
    wavenumbers=mock_wavenumbers,
    bands=30,
    noise_method='silent_region',
    silent_region_range=(1750, 1800),
    silent_region_scale='non_uniform'
)

print(f"\nDenoising complete.")
print(f"Original image shape: {mock_image.shape}")
print(f"Denoised image shape: {denoised_image.shape}")

For patch-wise denoising use the 'patchwise_imnf_denoise' function.

denoised_patch_imnf = patchwise_imnf_denoise(
        image,
        wavenumbers=mock_wavenumbers,
        patch_size=(32, 32),
        overlap=16,
        bands=30,
        noise_method='silent_region',
        silent_region_range=(1750, 1800),
    	silent_region_scale='non_uniform'
    )


License
This project is licensed under the MIT License - see the LICENSE file for details.