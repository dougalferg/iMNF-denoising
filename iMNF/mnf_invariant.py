import numpy as np
import warnings
from scipy.signal import savgol_filter

def check_for_atypical_chemistry(hyperspectraldata, wavenumbers, 
                                 silent_region_range=(1650, 1800), 
                                 chunk_size_cm1=50, 
                                 variance_threshold_multiplier=3.0):
    """
    Acts as an automated QC warning system to check for atypical chemistry or 
    narrow-band contaminants (e.g., water vapor) in the user-defined silent region.
    """
    # 1. Reshape data
    if hyperspectraldata.ndim == 3:
        m, n, s = hyperspectraldata.shape
        X = np.reshape(hyperspectraldata, (-1, s))
    elif hyperspectraldata.ndim == 2:
        X = hyperspectraldata

    # Ensure wavenumbers are sorted
    sort_idx = np.argsort(wavenumbers)
    sorted_wn = wavenumbers[sort_idx]
    sorted_X = X[:, sort_idx]

    low_wn, high_wn = min(silent_region_range), max(silent_region_range)
    region_indices = np.where((sorted_wn >= low_wn) & (sorted_wn <= high_wn))[0]
    
    # Sanity Check 1: Does the region exist?
    if len(region_indices) == 0:
        raise ValueError(f"Silent region {silent_region_range} not found in wavenumbers.")

    # Sanity Check 2: Is the region large enough to chunk? 
    # (Need at least 5 points for the SG filter, let's require at least 15 points total)
    if len(region_indices) < 15:
        warnings.warn("Silent region is too narrow for reliable automated contaminant checking. Proceeding without QC check.")
        return True # Pass by default if we can't test it

    chunk_variances = []
    chunk_centers = []
    
    # 2. Slide the chunking window across the selected region
    current_start = low_wn
    step_size = chunk_size_cm1 / 2 # 50% overlap for sliding window
    
    while current_start + chunk_size_cm1 <= high_wn:
        current_end = current_start + chunk_size_cm1
        chunk_idx = np.where((sorted_wn >= current_start) & (sorted_wn <= current_end))[0]
        
        # Sanity Check 3: Does this specific chunk have enough points for SG filter?
        if len(chunk_idx) >= 7: 
            chunk_slice = slice(chunk_idx.min(), chunk_idx.max() + 1)
            noise_region = sorted_X[:, chunk_slice]
            
            # Apply SG derivative
            noise_deriv = savgol_filter(noise_region, window_length=5, polyorder=2, deriv=1, axis=1)
            base_noise_var = np.mean(np.var(noise_deriv, axis=0))
            
            chunk_variances.append(base_noise_var)
            chunk_centers.append(current_start + (chunk_size_cm1 / 2))
            
        current_start += step_size

    if not chunk_variances:
        return True # Failsafe

    # 3. Evaluate Stability
    median_variance = np.median(chunk_variances)
    max_variance = np.max(chunk_variances)
    max_idx = np.argmax(chunk_variances)
    
    # If the max variance exceeds our multiplier threshold compared to the median
    if max_variance > (median_variance * variance_threshold_multiplier):
        problem_center = chunk_centers[max_idx]
        warning_msg = (
            f"\n--- iMNF QC WARNING ---\n"
            f"Atypical chemistry or structured interference detected in the silent region!\n"
            f"Variance spiked at ~{problem_center:.1f} cm⁻¹. Max variance is {max_variance/median_variance:.1f}x higher than the regional median.\n"
            f"Recommendation: Visual inspection of this region is strongly advised. Consider shifting your silent_region_range to avoid this contaminant."
        )
        warnings.warn(warning_msg)
        return False
        
    return True

def find_optimal_silent_region(hyperspectraldata, wavenumbers, 
                               search_range=(1750, 2200), 
                               window_size=10, 
                               step_size=5):
    """
    Dynamically finds the optimal biochemically silent region by sliding a window.
    Includes robust boundary checks to prevent out-of-bounds errors.
    """
    if hyperspectraldata.ndim == 3:
        m, n, s = hyperspectraldata.shape
        X = np.reshape(hyperspectraldata, (-1, s))
    elif hyperspectraldata.ndim == 2:
        X = hyperspectraldata
    else:
        raise ValueError("Input data must be 2D or 3D.")

    # 1. Boundary Intersection Check
    actual_min_wn, actual_max_wn = np.min(wavenumbers), np.max(wavenumbers)
    low_wn = max(min(search_range), actual_min_wn)
    high_wn = min(max(search_range), actual_max_wn)
    
    if (high_wn - low_wn) < window_size:
        raise ValueError(
            f"Cannot run heuristic. The available overlapping wavenumber range "
            f"({low_wn:.1f} - {high_wn:.1f} cm⁻¹) is smaller than the requested "
            f"window size ({window_size} cm⁻¹)."
        )

    # Ensure wavenumbers are sorted for slicing
    sort_idx = np.argsort(wavenumbers)
    sorted_wn = wavenumbers[sort_idx]
    sorted_X = X[:, sort_idx]

    results = []
    
    # 2. Slide the window across the valid search range
    current_start = low_wn
    while current_start + window_size <= high_wn:
        current_end = current_start + window_size
        
        indices = np.where((sorted_wn >= current_start) & (sorted_wn <= current_end))[0]
        
        if len(indices) >= 7: # Ensure window is large enough for SG filter safely
            window_slice = slice(indices.min(), indices.max() + 1)
            noise_region = sorted_X[:, window_slice]
            
            noise_deriv = savgol_filter(noise_region, window_length=5, polyorder=2, deriv=1, axis=1)
            base_noise_var = np.mean(np.var(noise_deriv, axis=0))
            
            center_wn = current_start + (window_size / 2)
            results.append((center_wn, current_start, current_end, base_noise_var))
            
        current_start += step_size

    if not results:
        raise ValueError("No valid spectral windows could be formed. Check data spacing and window_size.")

    return results


def imnf_denoise(hyperspectraldata, wavenumbers=None, bands=30,
                 noise_method='silent_region', silent_region_range=(1750, 1800),
                 silent_region_slice=None, silent_region_scale='non_uniform'):
    """
    Perform Spatially Invariant Minimum Noise Fraction (iMNF) denoising.

    This implementation includes the proposed iMNF method ('silent_region') as well as
    the traditional order-dependent MNF ('image_array').

    Parameters
    ----------
    hyperspectraldata : numpy.ndarray
        The input hyperspectral data. Can be either a 2D array (pixels × spectral bands)
        or a 3D array (rows × columns × spectral bands).
    wavenumbers : numpy.ndarray, optional
        Wavenumber array. Required for 'silent_region' method if slice is not provided.
    bands : int, optional
        The number of MNF bands (K) to retain for reconstruction. Default is 30.
    noise_method : str, optional
        - 'silent_region': (Recommended iMNF) Spatially invariant noise estimation.
        - 'traditional_MNF': Traditional order-dependent noise estimation.
    silent_region_range : tuple, optional
        The (start, end) wavenumbers for the 'silent_region' method.
    silent_region_slice : slice, optional
        A manual slice object to specify the silent region, overriding the range.
    silent_region_scale : str, optional
        For 'silent_region' method, specifies the noise profile. Default 'non_uniform'.
        - 'non_uniform': Physically motivated model based on transmittance.
        - 'uniform': Assumes uniform noise variance across all bands.

    Returns
    ----------
    D_reshaped : numpy.ndarray
        The denoised hyperspectral data, reshaped to its original dimensions.
    P_hat_K : numpy.ndarray
        The forward transformation matrix (P_hat_K).
    P_tilde_K : numpy.ndarray
        The reconstruction matrix (P_tilde_K).
    """

    #   1. Reshape data to a 2D matrix X  
    if hyperspectraldata.ndim == 3:
        m, n, s = hyperspectraldata.shape
        X = np.reshape(hyperspectraldata, (-1, s))
    elif hyperspectraldata.ndim == 2:
        X = hyperspectraldata
        m, s = X.shape
    else:
        raise ValueError("Input data must be 2D or 3D.")

    #   2. Estimate the Noise Covariance Matrix (Sigma_delta)  
    if noise_method == 'silent_region':  # Proposed iMNF method
        if silent_region_slice is None:
            if wavenumbers is None:
                raise ValueError("A 'wavenumbers' array is required for 'silent_region' method.")
            low_wn, high_wn = min(silent_region_range), max(silent_region_range)
            indices = np.where((wavenumbers >= low_wn) & (wavenumbers <= high_wn))[0]
            if len(indices) == 0:
                raise ValueError(f"Silent region range {silent_region_range} not found.")
            silent_region_slice = slice(indices.min(), indices.max() + 1)

        # Eq (9): Calculate base noise variance from the derivative of a silent region
        noise_region = X[:, silent_region_slice]
        noise_deriv = savgol_filter(noise_region, window_length=5, polyorder=2, deriv=1, axis=1)
        base_noise_var = np.mean(np.var(noise_deriv, axis=0))

        if silent_region_scale == 'non_uniform':
            # Eq (10): Convert mean absorbance (A) to mean transmittance (T)
            mean_spectrum_abs = np.mean(X, axis=0)
            mean_spectrum_trans = 10**(-mean_spectrum_abs)
            
            # Eq (11): Calculate reference transmittance from the silent region
            transmittance_ref = np.mean(mean_spectrum_trans[silent_region_slice])
            
            # Eq (12): Create a non-uniform noise profile based on transmittance
            epsilon = 1e-9 # Add for numerical stability
            scaling_factor = (transmittance_ref / (mean_spectrum_trans + epsilon))**2
            noise_profile_vector = base_noise_var * scaling_factor
            
            # Eq (13): Construct the diagonal noise covariance matrix
            Sigma_delta = np.diag(noise_profile_vector)

        elif silent_region_scale == 'uniform':
            Sigma_delta = np.eye(s) * base_noise_var
        else:
            raise ValueError("silent_region_scale must be 'uniform' or 'non_uniform'.")

    elif noise_method == 'traditional_MNF':  # Traditional MNF method
        # Eq (1): Estimate noise matrix N by differencing adjacent pixels
        N = X[:-1, :] - X[1:, :]
        # Eq (2): Calculate the noise covariance scatter matrix
        Sigma_delta = N.T @ N
    else:
        raise ValueError(f"Unknown noise_method: {noise_method}")

    #   3. First Eigendecomposition on the Noise  
    # Eq (2): Diagonalize the noise covariance matrix: Sigma_delta = V * Lambda_delta * V^T
    Lambda_delta_vals, V = np.linalg.eigh(Sigma_delta)

    # Robustly handle numerical instability: clamp small negative eigenvalues to zero.
    Lambda_delta_vals[Lambda_delta_vals < 0] = 0

    # Sort eigenvalues/eigenvectors in descending order for consistency
    sort_indices = np.argsort(Lambda_delta_vals)[::-1]
    Lambda_delta_vals = Lambda_delta_vals[sort_indices]
    V = V[:, sort_indices]

    #   4. Whiten the Data  
    # Create the diagonal matrix for whitening: Lambda_delta^(-1/2)
    Lambda_delta_inv_sqrt_diag = np.divide(1.0, np.sqrt(Lambda_delta_vals),
                                           out=np.zeros_like(Lambda_delta_vals),
                                           where=Lambda_delta_vals != 0)
    
    # Eq (3): Apply noise whitening transformation: W = X * V * Lambda_delta^(-1/2)
    W = X @ V @ np.diag(Lambda_delta_inv_sqrt_diag)

    #   5. Second Eigendecomposition on Whitened Data  
    # Eq (4): Diagonalize the whitened data covariance: W^T*W = G * Lambda_omega * G^T
    Lambda_omega_vals, G = np.linalg.eigh(W.T @ W)
    
    # Sort eigenvalues/eigenvectors in descending SNR order
    sort_indices_2 = np.argsort(Lambda_omega_vals)[::-1]
    G = G[:, sort_indices_2]

    #   6. Compute Transformation and Reconstruction Matrices  
    # Truncate to the desired number of bands (K)
    G_K = G[:, :bands]
    
    # Eq (5): Define the forward transformation matrix (P_hat_K)
    P_hat_K = V @ np.diag(Lambda_delta_inv_sqrt_diag) @ G_K

    # Eq (6): Define the reconstruction matrix (P_tilde_K)
    # The previous clamping of eigenvalues makes adding epsilon here unnecessary.
    Lambda_delta_sqrt_diag = np.sqrt(Lambda_delta_vals)
    P_tilde_K = V @ np.diag(Lambda_delta_sqrt_diag) @ G_K
    
    #   7. Denoise by Projecting and Reconstructing  
    # Eq (7): Project data into truncated MNF space: M_K = X * P_hat_K
    M_K = X @ P_hat_K
    
    # Eq (8): Reconstruct denoised data: D = M_K * P_tilde_K^T
    D = M_K @ P_tilde_K.T

    #   8. Reshape Data to Original Dimensions  
    if hyperspectraldata.ndim == 3:
        D_reshaped = np.reshape(D, (m, n, s))
    else:
        D_reshaped = D
        
    return D_reshaped, P_hat_K, P_tilde_K


def patchwise_imnf_denoise(
    hyperspectraldata,
    wavenumbers,
    patch_size=(64, 64),
    overlap=32,
    **imnf_kwargs
):
    """
    Performs patch-wise iMNF denoising on a large 3D hyperspectral image.

    This function breaks the input image into smaller, overlapping patches, applies
    the iMNF algorithm to each patch, and reconstructs the full image by
    blending the results using a weighted average (Hann window) to prevent
    edge artifacts.

    ----------
    Parameters
    ----------
    hyperspectraldata : numpy.ndarray
        The 3D input hyperspectral image with shape (height, width, bands).
    wavenumbers : numpy.ndarray
        The array of wavenumber values, required for iMNF noise estimation.
    patch_size : tuple, optional
        The (height, width) of the patches to process. Default is (64, 64).
    overlap : int, optional
        The number of overlapping pixels between adjacent patches.
    **imnf_kwargs : dict
        Keyword arguments to be passed to the imnf_denoise function.
        Common arguments include `bands=30`, `noise_method='silent_region'`, etc.

    ----------
    Returns
    ----------
    numpy.ndarray
        The final denoised 3D hyperspectral image.
    """
    #   1. Input Validation  
    if hyperspectraldata.ndim != 3:
        raise ValueError("Input must be a 3D array (height, width, bands).")
    patch_h, patch_w = patch_size
    if overlap >= patch_h or overlap >= patch_w:
        raise ValueError("Overlap must be smaller than the patch dimensions.")

    #   2. Initialization  
    m, n, s = hyperspectraldata.shape
    denoised_image_accumulator = np.zeros_like(hyperspectraldata, dtype=np.float64)
    weight_accumulator = np.zeros((m, n), dtype=np.float64)

    #   3. Create 2D Hann Window for Smooth Blending  
    window_2d = np.outer(np.hanning(patch_h), np.hanning(patch_w))

    #   4. Iterate Overlapping Patches  
    step_h, step_w = patch_h - overlap, patch_w - overlap
    for y in range(0, m, step_h):
        for x in range(0, n, step_w):
            y_start, y_end = y, min(y + patch_h, m)
            x_start, x_end = x, min(x + patch_w, n)

            current_patch = hyperspectraldata[y_start:y_end, x_start:x_end, :]

            # Denoise the patch using the main iMNF function
            denoised_patch, _, _ = imnf_denoise(
                current_patch, wavenumbers=wavenumbers, **imnf_kwargs
            )

            # Ensure window matches the actual patch size (for edge cases)
            current_window = window_2d[:(y_end - y_start), :(x_end - x_start)]

            #   5. Accumulate Weighted Results  
            denoised_image_accumulator[y_start:y_end, x_start:x_end, :] += \
                denoised_patch * current_window[:, :, np.newaxis]
            weight_accumulator[y_start:y_end, x_start:x_end] += current_window

    #   6. Normalize by Weights to Get Final Image  
    # Add an epsilon to the denominator to prevent division by zero in empty areas
    epsilon = 1e-9
    final_denoised_image = np.divide(
        denoised_image_accumulator,
        weight_accumulator[:, :, np.newaxis] + epsilon
    )

    return final_denoised_image.astype(hyperspectraldata.dtype)