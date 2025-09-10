##### NOTE #####
# The imnf_denoise function has the functionality to perform the 
# original MNF implementation, we have the 'traditional_MNF' method argument.
# However, the original fast_mnf code can still be accessed if required using:
# from imnf.mnf_standard import fast_mnf_denoise, patchwise_fast_mnf_denoise

# Please also note that the data had to be compressed to fit in GitHub, the
# Original QCL data is ~200mb, so we compressed the data to 100 iMNF components
# and stored the scores and loadings to reconstruct the data. It is a closer
# approximation to real data and better than a siulated set.

#################

#%% Importing functions
# Import the denoising functions
from iMNF.mnf_invariant import imnf_denoise, patchwise_imnf_denoise
# Import the noise adder function
from iMNF.noise_adder import add_hyperspectral_noise
from iMNF.helpers import load_example_data, plot_comparison, interactive_comparison
# Import numpy
import numpy as np
    
#%% Now perform the denoising steps and comparisons
if __name__ == "__main__":
    # --- 1 - Load the data and add noise ---
    real_image, wavenumbers = load_example_data()
    ydims, xdims, wav_dims = real_image.shape
    
    # Noise function requires data to be unrolled 2D format (y*x by v)
    noisy_image = add_hyperspectral_noise(real_image.reshape(ydims*xdims, wav_dims), noise_level=0.01, seed=1)
    # Only the plotter needs the rest in 3D, so we reshape before running
    noisy_image = noisy_image.reshape(ydims, xdims, wav_dims)
    
    # If the function is unable to load the data you can just use np.load()
    # and provide the filepath directly to ...\exampledata\sample_core.npy
    
    # AMIDE_I_BAND_INDEX is just where the wavenumbers = ~1656 cm-1
    AMIDE_I_BAND_INDEX = np.where(wavenumbers==1654)[0][0]

    # --- 2. Performing iMNF Denoising (Spatially Invariant) ---
    print("\n--- Running Example 1: iMNF Denoising (Full Image) ---")
    denoised_imnf, _, _ = imnf_denoise(
        noisy_image,
        wavenumbers=wavenumbers,
        bands=30,
        noise_method='silent_region',
        silent_region_range=(1750, 1800)
    )

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

    # --- 4. Performing Standard MNF Denoising (Order-Dependent) ---
    print("\n--- Running Example 3: Standard MNF Denoising (Full Image) ---")
    denoised_mnf, _, _ = imnf_denoise(
        noisy_image,
        wavenumbers=wavenumbers,
        bands=30,
        noise_method='traditional_MNF'
    )

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

    # --- 6. Plot results to compare ---
    denoising_results = {
        "iMNF": denoised_imnf,
        "Patch-wise iMNF": denoised_patch_imnf,
        "Standard MNF": denoised_mnf,
        "Patch-wise MNF": denoised_patch_mnf
    }
    
    plot_comparison(
        original_image=noisy_image,
        denoised_results=denoising_results,
        band_index=AMIDE_I_BAND_INDEX
    )
    
    # --- 7. Clickable interative plotter ---
    all_datasets = {
        "Noisy": noisy_image,
        "iMNF": denoised_imnf,
        "Patch-wise iMNF": denoised_patch_imnf,
        "Standard MNF": denoised_mnf,
        "Patch-wise MNF": denoised_patch_mnf
    }
    
    interactive_comparison(
        display_image=denoised_imnf[:,:,AMIDE_I_BAND_INDEX],
        datasets=all_datasets,
        wavenumbers=wavenumbers
    )
    
    
    