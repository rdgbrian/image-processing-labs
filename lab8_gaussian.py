import matplotlib.pyplot as plt
import numpy as np
import os

from src.filters import lapacian_gausian_fourier, apply_filter_in_frequency_domain
from src.utils import rgb2gray
from src.resize import crop_power2

# Define directories
image_dir = "images\\"
image_names = ["crying-cat-meme.jpg", "juliana.jpg"]
output_dir = "outputs\\lab8_laplacian_gaussian"  # Save all outputs here

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to apply Laplacian of Gaussian for different thresholds and sigmas
def apply_laplacian_of_gaussian(image_name, thresholds, sigmas):
    # Load and preprocess the image
    image_path = os.path.join(image_dir, image_name)
    cropped_image = crop_power2(image_path, save_path=None)
    cropped_image = np.array(cropped_image)
    cropped_image = rgb2gray(cropped_image)

    for sigma in sigmas:
        # Generate the Laplacian of Gaussian filter
        filter_matrix = lapacian_gausian_fourier(cropped_image.shape, sigma)

        # Apply filter in the frequency domain
        filtered_image = apply_filter_in_frequency_domain(cropped_image, filter_matrix)

        # Normalize the filtered image
        filtered_image = (filtered_image - filtered_image.min()) / (filtered_image.max() - filtered_image.min())

        for threshold in thresholds:
            # Apply threshold to detect edges
            edges = filtered_image.real > threshold

            # Save the resulting edge-detected image
            plt.imshow(edges, cmap="gray")
            plt.title(f"LoG: Sigma={sigma}, Threshold={threshold}")
            plt.axis('off')
            plt.savefig(f"{output_dir}\\{image_name.split('.')[0]}_LoG_sigma{sigma}_threshold{threshold}.jpg", bbox_inches='tight')
            plt.close()

# Main function to process all images
def main():
    thresholds = [0.2, 0.4, 0.6, 0.8]  # Four threshold options
    sigmas = [0.01, 0.05, 0.1, 0.2]  # Four sigma options

    for image_name in image_names:
        apply_laplacian_of_gaussian(image_name, thresholds, sigmas)

if __name__ == "__main__":
    main()
