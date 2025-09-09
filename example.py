import numpy as np
import matplotlib.pyplot as plt

##### NOTE #####
#The imnf_denoise function has the functionality to perform the 
# original MNF implementation, we have the 'traditional_MNF' method argument.
#However, the original fast_mnf code can still be accessed if required using:
#from imnf.mnf_standard import fast_mnf_denoise, patchwise_fast_mnf_denoise
#################

# Import the denoising functions
from iMNF.mnf_invariant import imnf_denoise, patchwise_imnf_denoise
# Import the noise adder function
from iMNF.noise_adder import add_hyperspectral_noise


# Create a phantom image
def create_phantom_image(height=256, width=256, bands=425, noise_level=0.02, seed=42):
    """Creates a mock hyperspectral image with a circular feature for demonstration."""
    print("--- Creating mock hyperspectral data ---")
    rng = np.random.default_rng(seed)
    
    # Create a base image with a circular feature
    x, y = np.ogrid[:height, :width]
    circle = (x - height // 2)**2 + (y - width // 2)**2 < (height // 4)**2
    
    # Create a simple spectrum for the feature and the background
    wavenumbers = np.linspace(950, 1800, bands)
    background_spectrum = 0.1 + 0.2 * np.exp(-((wavenumbers - 1200) / 100)**2)
    feature_spectrum = 0.5 * np.exp(-((wavenumbers - 1650) / 50)**2)
    
    # Combine into a clean hyperspectral image (absorbance)
    clean_image_abs = np.zeros((height, width, bands))
    clean_image_abs[~circle, :] = background_spectrum
    clean_image_abs[circle, :] = background_spectrum + feature_spectrum
    
    #Add the noise to the data
    noisy_image_abs = add_hyperspectral_noise(clean_image_abs.reshape(height*width, bands), 0.5, seed=1)
    noisy_image_abs = noisy_image_abs.reshape(height, width, bands)
    
    print(f"Mock data created with shape: {noisy_image_abs.shape}")
    return noisy_image_abs, wavenumbers

# Create comparison plot
def plot_comparison(original, denoised, title, band_index=150):
    """Plots the original noisy and denoised images side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Use a consistent color scale
    vmin = np.min(original[:, :, band_index])
    vmax = np.max(original[:, :, band_index])
    
    axes[0].imshow(original[:, :, band_index], cmap='inferno', vmin=vmin, vmax=vmax)
    axes[0].set_title("Original Noisy Image")
    axes[0].axis('off')
    
    axes[1].imshow(denoised[:, :, band_index], cmap='inferno', vmin=vmin, vmax=vmax)
    axes[1].set_title("Denoised Image")
    axes[1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
#Now perform the denoising steps
if __name__ == "__main__":
    # --- 1. Generate Data ---
    noisy_image, wavenumbers = create_phantom_image()
    AMIDE_I_BAND_INDEX = np.argmin(np.abs(wavenumbers - 1656))


    # --- 2. Performing iMNF Denoising (Spatially Invariant) ---
    print("\n--- Running Example 1: iMNF Denoising (Full Image) ---")
    denoised_imnf, _, _ = imnf_denoise(
        noisy_image,
        wavenumbers=wavenumbers,
        bands=30,
        noise_method='silent_region',
        silent_region_range=(1750, 1800)
    )
    plot_comparison(noisy_image, denoised_imnf, "iMNF (Full Image)", AMIDE_I_BAND_INDEX)
    

    # --- 3. Performing Patch-wise iMNF Denoising ---
    print("\n--- Running Example 2: Patch-wise iMNF Denoising ---")
    denoised_patch_imnf = patchwise_imnf_denoise(
        noisy_image,
        wavenumbers=wavenumbers,
        patch_size=(32, 32),
        overlap=16,
        bands=30,
        noise_method='silent_region',
        silent_region_range=(1750, 1800)
    )
    plot_comparison(noisy_image, denoised_patch_imnf, "Patch-wise iMNF", AMIDE_I_BAND_INDEX)


    # --- 4. Performing Standard MNF Denoising (Order-Dependent) ---
    print("\n--- Running Example 3: Standard MNF Denoising (Full Image) ---")
    denoised_mnf, _, _ = imnf_denoise(
        noisy_image,
        wavenumbers=wavenumbers,
        bands=30,
        noise_method='traditional_MNF'
    )
    plot_comparison(noisy_image, denoised_mnf, "Standard MNF (Full Image)", AMIDE_I_BAND_INDEX)


    # --- 5. Performing Patch-wise Standard MNF Denoising ---
    print("\n--- Running Example 4: Patch-wise Standard MNF Denoising ---")
    # Note: We call the same patch-wise function but pass 'image_array' as the noise_method
    denoised_patch_mnf = patchwise_imnf_denoise(
        noisy_image,
        wavenumbers=wavenumbers,
        patch_size=(32, 32),
        overlap=16,
        bands=30,
        noise_method='traditional_MNF'
    )
    plot_comparison(noisy_image, denoised_patch_mnf, "Patch-wise Standard MNF", AMIDE_I_BAND_INDEX)
