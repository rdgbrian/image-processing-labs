import matplotlib.pyplot as plt
import numpy as np
import os
from time import time

from src.discrete_fourier_transform import dft2d
from src.fast_fourier_transform import fft2d
from src.image_transforms import dct2d, walsh_transform, haar_transform
from src.utils import rgb2gray
from src.resize import crop_power2

# Define directories
image_dir = "images\\"
image_names = ["crying-cat-meme.jpg", "nebula.jpg", "squirrel.jpg", "beach_sunset.jpg"]

image_names = ["crying-cat-meme.jpg"]

output_dir = "outputs\\lab7_reduced_transforms"  # Save all outputs here

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to remove the K lowest value coefficients in the transform
def remove_lowest_k_coefficients(transform, K):
    # Flatten the transform and sort by absolute values
    flat_transform = transform.flatten()
    indices = np.argsort(np.abs(flat_transform))[:K]
    flat_transform[indices] = 0

    # Reshape back to the original shape
    reduced_transform = flat_transform.reshape(transform.shape)
    return reduced_transform

# Function to calculate Mean Squared Error (MSE)
def calculate_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

# Function to process and save results for each image
def process_image(image_name, num_reconstructions=5):
    # Load and preprocess the image
    image_path = os.path.join(image_dir, image_name)
    original_image = crop_power2(image_path, save_path=None)
    original_image = np.array(original_image)
    original_image = rgb2gray(original_image)

    # Calculate FFT, DCT, Walsh, and Haar Transforms
    fft_transform, _ = fft2d(original_image, inverse=False)
    dct_transform = dct2d(original_image)
    walsh_transform_data = walsh_transform(original_image)
    haar_transform_data = haar_transform(original_image)

    transforms = {
        "FFT": fft_transform,
        "DCT": dct_transform,
        "Walsh": walsh_transform_data,
        "Haar": haar_transform_data
    }

    mse_results = {key: [] for key in transforms.keys()}
    max_coefficients = original_image.size
    k_values = np.linspace(0, max_coefficients, num=5, endpoint=False, dtype=int)  # 5 iterations over K

    for transform_name, transform_data in transforms.items():
        for K in k_values:  # Iterate over 5 values of K
            reduced_transform = remove_lowest_k_coefficients(transform_data, K)

            # Reconstruct the image
            if transform_name == "FFT":
                reconstructed_image, _ = fft2d(reduced_transform, inverse=True)
            elif transform_name == "DCT":
                reconstructed_image = dct2d(reduced_transform, inverse=True)
            elif transform_name == "Walsh":
                reconstructed_image = walsh_transform(reduced_transform, inverse=True)
            elif transform_name == "Haar":
                reconstructed_image = haar_transform(reduced_transform, inverse=True)

            # Calculate MSE
            mse = calculate_mse(original_image, reconstructed_image.real)
            mse_results[transform_name].append(mse)

            # Save a subset of reconstructed images
            plt.imshow(reconstructed_image.real, cmap="gray")
            plt.title(f"{transform_name} Reconstruction (K={K})")
            plt.axis('off')
            plt.savefig(f"{output_dir}\{image_name.split('.')[0]}_{transform_name}_K{K}.jpg", bbox_inches='tight')
            plt.close()

    # Plot MSE results
    plt.figure(figsize=(10, 6))
    for transform_name, mse_values in mse_results.items():
        plt.plot(k_values, mse_values, label=f"{transform_name} MSE")

    plt.xlabel("K (Number of Removed Coefficients)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title(f"{image_name} - MSE vs K for Transforms")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(f"{output_dir}\{image_name.split('.')[0]}_mse_plot.jpg")
    plt.close()

# Main function to process all images
def main():
    for image_name in image_names:
        process_image(image_name)

if __name__ == "__main__":
    main()
