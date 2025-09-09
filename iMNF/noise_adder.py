import numpy as np

def add_hyperspectral_noise(absorbance_datacube, noise_level, seed=None):
    """
    Adds realistic, spectrometer-like noise to a hyperspectral dataset.

    This function operates on a 2D array where each row is an individual
    absorbance spectrum. It converts the data to transmittance, adds
    independent Gaussian noise to each spectrum, and then converts back
    to absorbance.

    Args:
        absorbance_datacube (np.ndarray): A 2D NumPy array with a shape of
                                         (num_spectra, num_wavenumbers).
        noise_level (float): The standard deviation of the Gaussian noise
                             to add to the transmittance. A good starting
                             value is between 0.001 and 0.01.
        seed (int, optional): A seed for the random number generator to ensure
                              reproducibility. Defaults to None.

    Returns:
        np.ndarray: The noisy hyperspectral dataset with the same shape as the input.
    """
    if absorbance_datacube.ndim != 2:
        raise ValueError("Input must be a 2D array of shape (num_spectra, num_wavenumbers).")
    
    # 1. Convert the entire datacube from absorbance to transmittance
    transmittance_datacube = 10**(-absorbance_datacube)

    # 2. Generate a noise array of the same shape and add it
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0, scale=noise_level, size=transmittance_datacube.shape)
    transmittance_noisy = transmittance_datacube + noise

    # Clip all values to be physically realistic (0 to 1)
    transmittance_noisy = np.clip(transmittance_noisy, 1e-5, 1.0)
    
    # 3. Convert the entire noisy datacube back to absorbance
    absorbance_noisy = -np.log10(transmittance_noisy)
    
    return absorbance_noisy
