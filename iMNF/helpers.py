import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Example data loader
def load_example_data():
    """Loads the sample dataset from the 'exampledata' folder."""
    print("Loading example dataset...")
    
    current_dir = Path.cwd()
    # Construct the file paths relative to the script location
    scores_path = os.path.join(current_dir, "Documents\GitHub\iMNF-denoising", "exampledata", "scores.npy")
    loadings_path = os.path.join(current_dir, "Documents\GitHub\iMNF-denoising", "exampledata", "loadings.npy") 
    wav_path = os.path.join(current_dir, "Documents\GitHub\iMNF-denoising", "exampledata", "wavenumbers.npy")
    
    
    # Check if the files exist before trying to load them
    if not (os.path.exists(scores_path) and os.path.exists(wav_path) and
            os.path.exists(loadings_path)):
        raise FileNotFoundError(
            f"Example data not found. Make sure the 'exampledata' folder is "
            f"in the same directory as the script."
        )
        
    # Load the data using the constructed paths
    scores_data = np.load(scores_path)
    loadings_data = np.load(loadings_path)
    wavenumbers = np.load(wav_path)
    
    image_data = scores_data @ loadings_data.T
    
    print(f"Dataset loaded with shape: {image_data.shape}")
    return image_data, wavenumbers

# Plot comparison plotter
def plot_comparison(original_image, denoised_results, band_index, main_title="Denoising Method Comparison"):
    """
    Creates an interactive dashboard to compare denoising results.

    The figure has two columns:
    1. Left: A large plot of the original noisy image.
    2. Right: A 2x2 grid of the four denoised results with shared zoom.
    """
    # --- 1. Setup Figure and Custom Grid Layout ---
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(main_title, fontsize=16, fontweight='bold')

    gs_main = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])

    # --- 2. Create Axes and Link Them ---
    ax_main = fig.add_subplot(gs_main[0, 0])
    gs_nested = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_main[0, 1], hspace=0.05, wspace=0.05)
    
    ax_imnf = fig.add_subplot(gs_nested[0, 0], sharex=ax_main, sharey=ax_main)
    ax_patch_imnf = fig.add_subplot(gs_nested[0, 1], sharex=ax_main, sharey=ax_main)
    ax_mnf = fig.add_subplot(gs_nested[1, 0], sharex=ax_main, sharey=ax_main)
    ax_patch_mnf = fig.add_subplot(gs_nested[1, 1], sharex=ax_main, sharey=ax_main)
    
    # --- 3. Plot the Data ---
    vmin = np.min(original_image[:, :, band_index])
    vmax = np.max(original_image[:, :, band_index])

    ax_main.imshow(original_image[:, :, band_index], cmap='inferno', vmin=vmin, vmax=vmax)
    ax_main.set_title("Original Noisy Image (Zoom Here)")

    # --- 4. Plot Denoised Results with New Labeling ---
    # Unpack the results dictionary
    imnf_img = denoised_results["iMNF"]
    patch_imnf_img = denoised_results["Patch-wise iMNF"]
    mnf_img = denoised_results["Standard MNF"]
    patch_mnf_img = denoised_results["Patch-wise MNF"]

    # Plot the 2x2 grid
    ax_imnf.imshow(imnf_img[:, :, band_index], cmap='inferno', vmin=vmin, vmax=vmax)
    ax_patch_imnf.imshow(patch_imnf_img[:, :, band_index], cmap='inferno', vmin=vmin, vmax=vmax)
    ax_mnf.imshow(mnf_img[:, :, band_index], cmap='inferno', vmin=vmin, vmax=vmax)
    ax_patch_mnf.imshow(patch_mnf_img[:, :, band_index], cmap='inferno', vmin=vmin, vmax=vmax)

    # Set titles for the columns
    ax_imnf.set_title("Full Image")
    ax_patch_imnf.set_title("Patch-wise")

    # Set rotated y-labels for the rows on the far-left axes of the grid
    ax_imnf.set_ylabel("iMNF", rotation=90, fontsize=14, fontweight='bold', labelpad=10)
    ax_mnf.set_ylabel("Standard MNF", rotation=90, fontsize=14, fontweight='bold', labelpad=10)

    # Hide all tick labels on the shared axes for a cleaner look
    for ax in [ax_imnf, ax_patch_imnf, ax_mnf, ax_patch_mnf]:
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
def interactive_comparison(display_image, datasets, wavenumbers):
    """
    Displays an image and interactively plots spectra from multiple datasets.

    A single 2D image is displayed. A left click on the image places a marker
    and plots the corresponding spectra from all five provided 3D datasets
    on an adjacent graph. Each click clears the previous selection.

    Parameters
    ----------
    display_image : numpy.ndarray
        The 2D numpy array (e.g., the Amide I slice) to be displayed and clicked on.
    datasets : dict
        A dictionary where keys are string labels (e.g., "Noisy", "iMNF") and
        values are the corresponding 3D hyperspectral cubes.
    wavenumbers : numpy.ndarray
        A 1D numpy array for the x-axis (wavenumbers) of the spectra plot.
    """
    # --- 1. Plot Setup ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('Click Image (Left) to Compare Spectra (Right)')

    ax1.imshow(display_image, cmap='inferno', interpolation='nearest')
    ax1.set_title('Click on a pixel')

    ax2.set_xlabel('Wavenumber (cm⁻¹)')
    ax2.set_ylabel('Absorbance (a.u.)')
    ax2.invert_xaxis()

    # List to keep track of the currently plotted elements (pin)
    plotted_elements = []

    # --- 2. Click Event Function ---
    def onclick(event):
        # Ignore clicks outside the image axis
        if event.inaxes != ax1:
            return

        # On left click, clear and plot
        if event.button == 1:
            x, y = int(round(event.xdata)), int(round(event.ydata))

            # Ignore clicks outside the image boundaries
            if not (0 <= y < display_image.shape[0] and 0 <= x < display_image.shape[1]):
                return

            # --- Clear previous plots ---
            ax2.cla() # Clear the spectra axis completely
            while plotted_elements:
                plotted_elements.pop().remove() # Remove the old pin

            # --- Plot new data ---
            # Place a new pin on the image
            pin = ax1.plot(x, y, 'o', markerfacecolor='cyan', markeredgecolor='black', markersize=8)
            plotted_elements.extend(pin)

            # Plot the spectrum from each dataset at the clicked location
            for name, data_cube in datasets.items():
                spectrum = data_cube[y, x, :]
                ax2.plot(wavenumbers, spectrum, label=name, alpha = 0.6)

            # --- Update plot aesthetics ---
            ax2.set_title(f'Spectra at ({x}, {y})')
            ax2.set_xlabel('Wavenumber (cm⁻¹)')
            ax2.set_ylabel('Absorbance (a.u.)')
            ax2.invert_xaxis()
            ax2.legend()
            ax2.relim()
            ax2.autoscale_view()
            fig.canvas.draw()

    # --- 3. Connect event handler and show plot ---
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()