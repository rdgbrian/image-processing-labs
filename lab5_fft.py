import matplotlib.pyplot as plt
import numpy as np
import os
from time import time
from PIL import Image

from src.discrete_fourier_transform import dft2d
from src.fast_fourier_transform import fft2d
from src.utils import fourier_spectrum, create_white_square, rgb2gray, center_fourier, save_transforms
from src.resize import crop_power2, reduce_image

# Define directories
image_dir = "images\\"
image_names = ["crying-cat-meme.jpg", "nebula.jpg", "squirrel.jpg", "beach_sunset.jpg"]
output_dir = "outputs\\lab5_transforms"  # Save all transform outputs here

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to compute and save FFT and DFT transforms for an image
def save_transforms_for_image(image_name, min_size=16):
    # Load and preprocess the image
    image_path = os.path.join(image_dir, image_name)
    cropped_image = crop_power2(image_path, save_path=None)  # Crop to power of 2
    cropped_image = np.array(cropped_image)
    cropped_image = rgb2gray(cropped_image)

    cropped_image = reduce_image(cropped_image)
    # Reduce the image size and compute transforms
    size = cropped_image.shape[0]
    print(size)
    n = 1  # Reduction factor (2^n)

    while size >= min_size:
        print(f"Processing image {image_name} at size {size}x{size}...")

        # Reduce image size
        reduced_image = reduce_image(cropped_image, n)

        # Dictionary to store transforms and their computation times
        transform_dict = {}
        time_dict = {}

        # Compute FFT and time it
        start_time = time()
        fft_og, _ = fft2d(reduced_image, inverse=False)
        time_dict["fft"] = time() - start_time
        transform_dict["fft"] = center_fourier(fft_og)

        # Compute DFT and time it
        start_time = time()
        dft_og = dft2d(reduced_image, inverse=False)
        time_dict["dft"] = time() - start_time
        dft_og_center = center_fourier(dft_og)
        
        transform_dict["dft"] = center_fourier(dft_og)

        # # Save transforms
        save_transforms(image_name + f"_{size}", transform_dict, output_dir)

        # Plot and save Fourier spectrum
        for key, transform in transform_dict.items():
            spectrum = fourier_spectrum(transform.real, transform.imag)
            plt.imshow(spectrum, cmap='gray')
            plt.title(f"{image_name} - {key.upper()} Spectrum ({size}x{size})")
            plt.axis('off')
            plt.savefig(f"{output_dir}\\{image_name.split('.')[0]}_{key}_{size}x{size}.jpg", bbox_inches='tight')
            plt.close()

        # Update size and reduction factor
        n += 1
        size = reduced_image.shape[0]

# Function to generate timing plots
def plot_transform_times(all_times):
    for i, times in enumerate(all_times):
        plt.figure(figsize=(8, 6))

        # Plot DFT times
        plt.plot(times["size"], times["dft"], marker='o', label='DFT Time', color='blue')

        # Plot FFT times
        plt.plot(times["size"], times["fft"], marker='s', label='FFT Time', color='green')

        # Add labels, title, and legend
        plt.xlabel('Image Size')
        plt.ylabel('Time (seconds)')
        plt.title(f'Image {i + 1}: DFT vs FFT Times')
        plt.legend()

        # Show grid
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Show the plot
        plt.savefig(f"{output_dir}\\transform_times_image_{i + 1}.jpg")
        plt.close()

# Main function to process all images
def main():
    all_times = []
    for image_name in image_names:
        save_transforms_for_image(image_name)

    # Example sanity check
    test_image = create_white_square(32, 8)
    f = np.fft.fft2(test_image)
    f_centered = np.fft.fftshift(f)
    np_real = f.real
    np_imag = f.imag

    fft_og, _ = fft2d(test_image, centered=True)
    og_real = fft_og.real
    og_imag = fft_og.imag

    are_real_equal = np.allclose(np_real, og_real, rtol=1e-6, atol=1e-9)
    are_imag_equal = np.allclose(np_imag, og_imag, rtol=1e-6, atol=1e-9)
    print("Sanity Check:")
    print("Are real parts equal?", are_real_equal)
    print("Are imaginary parts equal?", are_imag_equal)

if __name__ == "__main__":
    main()
