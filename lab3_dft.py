import matplotlib.pyplot as plt
import numpy as np
from src.discrete_fourier_transform import dft2d
import time

from src.utils import fourier_spectrum, create_white_square, rgb2gray, center_fourier
from src.resize import crop_power2, reduce_image

# Define directories
image_dir = "images\\"
image_names = ["crying-cat-meme.jpg", "nebula.jpg", "squirrel.jpg", "beach_sunset.jpg"]
image_names = ["crying-cat-meme.jpg"]

output_dir = "outputs\\lab3_dft"  # Save all DFT outputs here

# Function to calculate and save the DFTs of an image
def save_image_dfts(image_name,add_log=True):
    # Load and preprocess the image
    image = crop_power2(image_dir + image_name, save_path=None)  # Crop to power of 2
    image = np.array(image)
    image = rgb2gray(image)  # Convert to grayscale

    num_reduce = 0
    image = reduce_image(image, num_red=2)
    size = image.shape[0]
    print(size)

    sizes = []
    times = []

    while size >= 16:
        # Perform the DFT
        start_time = time.time()
        dft_og = dft2d(image)
        # dft_og = np.fft.fft2(image)
        end_time = time.time()
        dft_duration = end_time - start_time
        
        # Store the size and timing result
        sizes.append(size)
        times.append(dft_duration)

        dft_real = dft_og.real
        dft_imag = dft_og.imag
        spectrum = fourier_spectrum(dft_real, dft_imag)

        # Center the DFT
        dft_centered = center_fourier(dft_og)
        spectrum_centered = fourier_spectrum(dft_centered.real, dft_centered.imag)

        if(add_log):
            spectrum = (255 / np.log10(255)) * np.log10(1 + 255/(np.max(spectrum)) * np.abs(spectrum))  # log1p(x) = log(1 + x), safe for 0 values
            spectrum_centered = (255 / np.log10(255)) * np.log10(1 + 255/(np.max(spectrum_centered)) * np.abs(spectrum_centered))  # log1p(x) = log(1 + x), safe for 0 values


        # Save the DFT spectra
        suffix = f"_{size}x{size}" if num_reduce > 0 else ""
        plt.imshow(spectrum, cmap='gray')
        plt.title(f"{image_name} - DFT Spectrum{suffix}")
        plt.axis('off')
        plt.savefig(f"{output_dir}\\{image_name.split('.')[0]}{suffix}_dft_spectrum.jpg", bbox_inches='tight')
        plt.close()

        plt.imshow(spectrum_centered, cmap='gray')
        plt.title(f"{image_name} - Centered DFT Spectrum{suffix}")
        plt.axis('off')
        plt.savefig(f"{output_dir}\\{image_name.split('.')[0]}{suffix}_centered_dft_spectrum.jpg", bbox_inches='tight')
        plt.close()

        # Reduce the image size for the next iteration
        if size > 32:
            num_reduce += 1
            image = reduce_image(image, num_red=1)
            size = image.shape[0]
        else:
            break

        
    sorted_data = sorted(zip(sizes, times), key=lambda x: x[0], reverse=False)
    sorted_sizes, sorted_times = zip(*sorted_data)

    # Plot timing results for this image
    plt.figure()
    plt.plot(sorted_sizes, sorted_times, marker='o')
    plt.gca() # Optional: show largest size on the left
    plt.xlabel("Image Size (N)")
    plt.ylabel("DFT Computation Time (s)")
    plt.title(f"DFT Timing for {image_name}")
    plt.grid(True)
    plt.savefig(f"{output_dir}\\{image_name.split('.')[0]}_dft_timing.jpg", bbox_inches='tight')
    plt.close()

# Function to compare DFT implementation with numpy
def compare_dft_with_numpy(image_name):
    # Load and preprocess the image
    image = crop_power2(image_dir + image_name, save_path=None)  # Crop to power of 2
    image = np.array(image)
    image = rgb2gray(image)  # Convert to grayscale

    # Perform DFT using numpy
    f = np.fft.fft2(image)
    f_centered = np.fft.fftshift(f)
    np_real = f.real
    np_imag = f.imag

    # Perform DFT using custom implementation
    dft_og = dft2d(image, centered=True)
    og_real = dft_og.real
    og_imag = dft_og.imag

    # Compare the results
    are_real_equal = np.allclose(np_real, og_real, rtol=1e-6, atol=1e-9)
    are_imag_equal = np.allclose(np_imag, og_imag, rtol=1e-6, atol=1e-9)

    print(f"{image_name} - Are real parts equal? {are_real_equal}")
    print(f"{image_name} - Are imaginary parts equal? {are_imag_equal}")

# Main function to process all images
def main():
    for image_name in image_names:
        save_image_dfts(image_name)
        # compare_dft_with_numpy(image_name)

if __name__ == "__main__":
    main()
