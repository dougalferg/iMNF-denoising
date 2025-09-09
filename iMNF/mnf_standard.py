import numpy as np

def fast_mnf_denoise(hyperspectraldata, SNR = 5, bands = 0):
        """
        Perform Fast Minimum Noise Fraction (MNF) denoising on hyperspectral data.
        Code derived from supplementary information from Gupta et al:
        https://doi.org/10.1371/journal.pone.0205219    
    
        This function reduces noise in hyperspectral images using the MNF 
        transformation. The input data `hyperspectraldata` can be 2D or 3D. 
        If the input is 3D (i.e., a hyperspectral image with spatial and 
        spectral dimensions), it will be reshaped to 2D for processing and 
        reshaped back to its original dimensions after denoising. If the input 
        is already 2D, it will be processed directly.
    
        Steps:
            1. Compute the difference matrix `dX` for noise estimation.
            2. Perform eigenvalue decomposition on `dX^T * dX`.
            3. Weight the input data by the inverse square root of the eigenvalues.
            4. Perform eigenvalue decomposition on the weighted data.
            5. Retain the top K components based on Rose's noise criterion.
            If bands =/= 0, the number of components is set to bands.
            6. Compute the transformation matrices `Phi_hat` and `Phi_tilde`.
            7. Project the data onto MNF components and reconstruct the denoised data.
    
        ----------
        Parameters
        ----------
        hyperspectraldata : numpy.ndarray
            The input hyperspectral data. Can be either a 2D array (pixels × spectral bands)
            or a 3D array (rows × columns × spectral bands).
            
        SNR : int
            The signal to noise ratio value for detectability threshold outlined
            in the Rose criterion for medical imaging signal detection theory.
            A SNR value of 5 (default)  is a 95% probability of object
            detection by humans visually. 
            
        Bands : int 
            The number of MNF bands (K) to retain for reconstruction. Default is 30.
    
        ----------
        Returns
        ----------
        clean_data : numpy.ndarray
            The denoised hyperspectral data. The output will have the same dimensions as the input:
            - If the input was 3D, the output will be reshaped back to 3D.
            - If the input was 2D, the output will remain 2D.


        """
        
        # Check if the input is 3D and reshape to 2D if needed
        if hyperspectraldata.ndim == 3:
            m, n, s = hyperspectraldata.shape
            X = np.reshape(hyperspectraldata, (-1, s))  # Reshape to 2D
        elif hyperspectraldata.ndim == 2:
            X = hyperspectraldata
            m, n = X.shape
            s = n  # If already 2D, assume second dimension is the spectral dimension
        else:
            raise ValueError("Input C must be either 2D or 3D.")
    
        # Step 2: Create the dX matrix
        dX = np.zeros((m, s))
        for i in range(m - 1):
            dX[i, :] = X[i, :] - X[i + 1, :]
    
        # Step 3: Perform eigenvalue decomposition of dX' * dX
        S1, U1 = np.linalg.eigh(dX.T @ dX)
        # Small negative eigenvalues are pushed to zero
        # This prevents the sqrt of a negative number due to numerical instability.
        S1[S1 < 0] = 0
        
        ix = np.argsort(S1)[::-1]  # Sort in descending order
        U1 = U1[:, ix]
        D1 = S1[ix]
        # Use a safe division to avoid warnings when D1 contains zeros
        diagS1 = np.divide(1.0, np.sqrt(D1), out=np.zeros_like(D1), where=D1!=0)

    
        # Step 4: Compute weighted X
        wX = X @ U1 @ np.diag(diagS1)
    
        # Step 5: Perform eigenvalue decomposition of wX' * wX
        S2, U2 = np.linalg.eigh(wX.T @ wX)
        iy = np.argsort(S2)[::-1]  # Sort in descending order
        U2 = U2[:, iy]
        D2 = S2[iy]
    
        # Step 6: Retain top K components according to input SNR threshold
        S2_diag = D2 - 1
        if bands !=0:
            K = bands
        else:
            K = np.sum(S2_diag > SNR) 
        U2 = U2[:, :K]

        # Step 7: Compute Phi_hat and Phi_tilde
        Phi_hat = U1 @ np.diag(diagS1) @ U2
        Phi_tilde = U1 @ np.diag(np.sqrt(D1)) @ U2
    

        # Step 8: Project data onto MNF components and reshape to original dimensions
        mnfX = X @ Phi_hat
        Xhat = mnfX @ Phi_tilde.T
        
        if hyperspectraldata.ndim == 3:

            clean_data = np.reshape(Xhat, (m, n, s))  # Reshape back to 3D if input was 3D
        else:

            clean_data = Xhat  # Keep 2D if input was 2D
        return clean_data,  Phi_hat, Phi_tilde
    

def patchwise_fast_mnf_denoise(hyperspectral_image, patch_size=(64, 64), overlap=32, **mnf_kwargs):
    """
    Performs patch-wise Fast MNF denoising on a large 3D hyperspectral image.

    This function is designed to handle large datasets that may not fit into memory
    by breaking the input image into smaller, overlapping patches. It applies the
    Fast MNF algorithm to each patch individually and then reconstructs the full
    image by blending the denoised patches using a weighted average (based on a
    2D Hann window) to prevent edge artifacts and ensure a smooth result.

    ----------
    Parameters
    ----------
    hyperspectral_image : numpy.ndarray
        The 3D input hyperspectral image with shape (height, width, bands).
    patch_size : tuple, optional
        The (height, width) of the patches to process, by default (64, 64).
    overlap : int, optional
        The number of overlapping pixels between adjacent patches. A larger
        overlap provides smoother blending but increases computation time.
        By default, this is set to half the patch size.
    **mnf_kwargs : dict
        Keyword arguments to be passed to the underlying single-patch MNF function.
        Common arguments include `SNR=5` or `bands=30`.

    ----------
    Returns
    ----------
    numpy.ndarray
        The final denoised 3D hyperspectral image, with the same shape and
        data type as the input.

    """
    # 1. Input validation
    if hyperspectral_image.ndim != 3:
        raise ValueError("Input image must be a 3D array (height, width, bands).")
    patch_h, patch_w = patch_size
    if overlap >= patch_h or overlap >= patch_w:
        raise ValueError("Overlap must be smaller than the patch size in both dimensions.")

    # 2. Initialization
    height, width, num_bands = hyperspectral_image.shape
    
    # Create accumulator arrays for the final image and the sum of weights
    denoised_image = np.zeros_like(hyperspectral_image, dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)

    # 3. Create a 2D Hann window for smooth blending
    window_col = np.hanning(patch_w)
    window_row = np.hanning(patch_h)
    window_2d = np.outer(window_row, window_col)
    
    # 4. Iterate through the image, processing overlapping patches
    step_h = patch_h - overlap
    step_w = patch_w - overlap

    for y in range(0, height, step_h):
        for x in range(0, width, step_w):
            # Define patch boundaries, clipping at the image edges
            y_start, y_end = y, min(y + patch_h, height)
            x_start, x_end = x, min(x + patch_w, width)
            
            # Extract the current 3D patch
            current_patch_3d = hyperspectral_image[y_start:y_end, x_start:x_end, :]
            patch_height, patch_width, _ = current_patch_3d.shape
            
            # Reshape the 3D patch to 2D (pixels x bands) for the MNF function
            current_patch_2d = np.reshape(current_patch_3d, (-1, num_bands))

            # Denoise the 2D patch
            # Assuming fast_mnf_denoise_OLD is defined elsewhere and handles 2D input
            denoised_patch_2d, _, _ = fast_mnf_denoise(current_patch_2d, **mnf_kwargs)
            
            # Reshape the denoised 2D patch back to its original 3D shape
            denoised_patch_3d = np.reshape(denoised_patch_2d, (patch_height, patch_width, num_bands))
            
            # Crop the window to match the patch size (important for edge patches)
            current_window = window_2d[:patch_height, :patch_width]
            
            # 5. Add the weighted, denoised 3D patch to our accumulator
            denoised_image[y_start:y_end, x_start:x_end, :] += denoised_patch_3d * current_window[:, :, np.newaxis]
            weight_sum[y_start:y_end, x_start:x_end] += current_window

    # 6. Normalize the denoised image by the sum of weights
    final_image = np.divide(denoised_image, weight_sum[:, :, np.newaxis], 
                            out=np.zeros_like(denoised_image), 
                            where=weight_sum[:, :, np.newaxis] != 0)

    return final_image.astype(hyperspectral_image.dtype)