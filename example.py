##### NOTE #####
# The imnf_denoise function has the functionality to perform the 
# original MNF implementation, we have the 'traditional_MNF' method argument.
# However, the original fast_mnf code can still be accessed if required using:
# from imnf.mnf_standard import fast_mnf_denoise, patchwise_fast_mnf_denoise

# Please also note that the data had to be compressed to fit in GitHub, the
# Original QCL data is ~200mb, so we compressed the data to 100 iMNF components
# and stored the scores and loadings to reconstruct the data. It is a closer
# approximation to real data and better than a simulated set.

#################

#If the user hits issues importing the package/functions following the readme,
# I have found the following workaround can help (uncomment from below):
#import sys
#import os
## Paste your filepath to the GitHub\iMNF-denoising\iMNF folder
#github_path = r'C:\Users\Dougal\Documents\GitHub\iMNF-denoising\iMNF'
#if github_path not in sys.path:
#    sys.path.append(github_path)

#%% Importing functions
from iMNF.mnf_invariant import imnf_denoise, patchwise_imnf_denoise, check_for_atypical_chemistry, find_optimal_silent_region
from iMNF.noise_adder import add_hyperspectral_noise
from iMNF.helpers import load_example_data, plot_comparison, interactive_comparison, plot_silent_region_variance
import numpy as np
    
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

    # --- 2. Quality Control: Check for Atypical Chemistry ---
    print("\n--- Running QC Check on Default Region (1750-1800 cm⁻¹) ---")
    is_clean = check_for_atypical_chemistry(
        noisy_image, 
        wavenumbers, 
        silent_region_range=(1750, 1800)
    )
    if is_clean:
        print("QC Passed: No atypical structured interference detected.")
    else:
        print("QC Warning: Contaminants detected. Proceed with caution.")

    # --- 3. Automated Heuristic: Find the Optimal Silent Region ---
    print("\n--- Finding Optimal Silent Region ---")
    # We sweep a slightly broader range to let the algorithm find the absolute lowest variance
    opt_start, opt_end = plot_silent_region_variance(
        noisy_image, 
        wavenumbers, 
        search_range=(1700, 1850), 
        window_size= 30
    )
    
    # Save the output to dynamically feed our denoisers
    optimal_region = (opt_start, opt_end)

    # --- 4. Performing iMNF Denoising (Spatially Invariant) ---
    print("\n--- Running Example 1: iMNF Denoising (Full Image) ---")
    denoised_imnf, _, _ = imnf_denoise(
        noisy_image,
        wavenumbers=wavenumbers,
        bands=30,
        noise_method='silent_region',
        silent_region_range= optimal_region
    )

    # --- 5. Performing Patch-wise iMNF Denoising ---
    print("\n--- Running Example 2: Patch-wise iMNF Denoising ---")
    denoised_patch_imnf = patchwise_imnf_denoise(
        noisy_image,
        wavenumbers=wavenumbers,
        patch_size=(32, 32),
        overlap=16,
        bands=30,
        noise_method='silent_region',
        silent_region_range=optimal_region
    )

    # --- 6. Performing Standard MNF Denoising (Order-Dependent) ---
    print("\n--- Running Example 3: Standard MNF Denoising (Full Image) ---")
    denoised_mnf, _, _ = imnf_denoise(
        noisy_image,
        wavenumbers=wavenumbers,
        bands=30,
        noise_method='traditional_MNF'
    )

    # --- 7. Performing Patch-wise Standard MNF Denoising ---
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

    # --- 8. Plot results to compare ---
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
    
    # --- 9. Clickable interative plotter ---
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
    
    
    