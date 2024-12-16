# %%
import matplotlib.pyplot as plt
import numpy as np

from src.discrete_fourier_transform import dft2d
from src.fast_fourier_transform import fft2d

from src.utils import fourier_spectrum, create_white_square, rgb2gray, center_fourier, save_transforms
from src.resize import crop_power2, reduce_image
from time import time
from PIL import Image

import os

#%%
cropped_image = crop_power2("images\crying-cat-meme.jpg", save_path="output_image.jpg")
cropped_image = rgb2gray(cropped_image)
cropped_image = reduce_image(cropped_image,4)

plt.imshow(cropped_image,cmap="grey")
plt.show()

image = cropped_image

fft_og, _ = fft2d(image,inverse=False)
og_real = fft_og.real
og_imag = fft_og.imag

spectrum = fourier_spectrum(fft_og.real,fft_og.imag)

recons, _ = fft2d(fft_og,inverse=True)
plt.imshow(recons.real,cmap="grey")
plt.show()

are_equal = np.allclose(recons,image, rtol=1e-6, atol=1e-9)
print("Are the arrays equal to the given precision?", are_equal)


#%%

# Main script
image_dir = "images\\"
image_names = ["crying-cat-meme.jpg","crying-cat-meme.jpg"]
output_dir = "outputs\\lab5"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Store all transforms in a list
# Store all transforms in a list
transforms_list = []
time_list = []
min_size = 16  # Minimum size of the reduced image
for i, image_path in enumerate(image_names):
    image_name = os.path.splitext(image_path)[0]

    # Load and preprocess the image
    cropped_image = crop_power2(os.path.join(image_dir, image_path))
    cropped_image = rgb2gray(cropped_image)
    
    cropped_image = reduce_image(cropped_image, 1)
    
    size = cropped_image.shape[0]
    n = 1  # Reduction factor (2^n)

    transforms_list.append([])
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
        fft_time = time() - start_time
        transform_dict["fft"] = fft_og
        time_dict["fft"] = fft_time

        # Compute DFT and time it
        start_time = time()
        dft_og = dft2d(reduced_image, inverse=False)
        dft_time = time() - start_time
        transform_dict["dft"] = dft_og
        time_dict["dft"] = dft_time

        # Save transforms
        save_transforms(image_name + f"_{size}", transform_dict, output_dir)

        # Append results to the transforms list
        transforms_list[i].append({
            "image": reduced_image,
            "image_name": image_name,
            "size": size,
            "transforms": transform_dict,
            "n": n,
            "time": time_dict,
        })

        # Update size and reduction factor
        n += 1
        size = reduced_image.shape[0]

print("All transforms computed and saved.")
#%%

all_times = []
for image_info in transforms_list:
    dft_times = []
    fft_times = []
    sizes = []
    for partition_info in image_info:
        sizes.append(partition_info["size"])
        dft_times.append(partition_info["time"]["dft"])
        fft_times.append(partition_info["time"]["fft"])
                         
    all_times.append({
        "size": sizes,
        "dft": dft_times,
        "fft":fft_times
    })

all_times[1]

#%%
# Generate a plot for each image
for i, times in enumerate(all_times):
    plt.figure(figsize=(8, 6))
    
    # Plot DFT times
    plt.plot(times["size"], times["dft"], marker='o', label='DFT Time', color='blue')
    
    # Plot FFT times
    plt.plot(times["size"], times["fft"], marker='s', label='FFT Time', color='green')
    
    # Set log scale for better visualization if values vary significantly
    # plt.yscale('log')  # Optional: Use a log scale for time
    
    # Add labels, title, and legend
    plt.xlabel('Partition Size')
    plt.ylabel('Time (seconds)')
    plt.title(f'Image {i+1}: DFT vs FFT Times')
    plt.legend()
    
    # Show grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Show the plot
    plt.show()

#%%

# ## Sanity Check
f = np.fft.fft2(image)
f_centered = np.fft.fftshift(f)
np_real = f.real
np_imag = f.imag

fft_og, _ = fft2d(image,centered=True)
og_real = fft_og.real
og_imag = fft_og.imag

are_equal = np.allclose(np_real,og_real, rtol=1e-6, atol=1e-9)
print("Are the arrays equal to the given precision?", are_equal)
are_equal = np.allclose(np_imag,og_imag, rtol=1e-6, atol=1e-9)
print("Are the arrays equal to the given precision?", are_equal)


