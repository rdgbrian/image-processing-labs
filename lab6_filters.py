# %%
import matplotlib.pyplot as plt
import numpy as np

from src.discrete_fourier_transform import dft2d
from src.fast_fourier_transform import fft2d
from src.filters import ideal_filter, butterworth_filter, apply_filter_in_frequency_domain

from src.utils import fourier_spectrum, create_white_square, rgb2gray, center_fourier, save_transforms
from src.resize import crop_power2, reduce_image
from time import time
from PIL import Image

import os

def apply_filter_in_frequency_domain(image, filter_matrix):
    # Compute the FFT of the image
    fft_image, _ = fft2d(image, inverse=False)
    fft_image_centered = center_fourier(fft_image)
    fft_image_filtered_centered = fft_image_centered * filter_matrix
    fft_image_filtered = center_fourier(fft_image_filtered_centered)
    # Reconstruct the complex FFT and perform the inverse FFT
    filtered_image, _ = fft2d(fft_image_filtered, inverse=True)
    return filtered_image

# %%
image_dir = "images\\"
image_names = ["crying-cat-meme.jpg", "crying-cat-meme.jpg"]
output_dir = "outputs\\lab6"

image_path = image_dir + image_names[0]
cropped_image = crop_power2(image_path)
cropped_image = rgb2gray(cropped_image)
# cropped_image = reduce_image(cropped_image, 4)

# Display the preprocessed image
plt.imshow(cropped_image, cmap="gray")
plt.title(f"Original Image: {image_names[0]}")
plt.show()

filter_matrix = ideal_filter(cropped_image.shape,5, pass_type="low")
# Apply filter in the frequency domain
filtered_image = apply_filter_in_frequency_domain(cropped_image,filter_matrix)

# Display the filtered image
plt.imshow(filtered_image.real, cmap="gray")
plt.title(f"Filtered Image: {image_names[0]}")
plt.show()


filter_matrix = ideal_filter(cropped_image.shape,5, pass_type="high")
# Apply filter in the frequency domain
filtered_image = apply_filter_in_frequency_domain(cropped_image,filter_matrix)

# Display the filtered image
plt.imshow(filtered_image.real, cmap="gray")
plt.title(f"Filtered Image: {image_names[0]}")
plt.show()

filter_matrix = butterworth_filter(cropped_image.shape,5,order=3, pass_type="low")
# Apply filter in the frequency domain
filtered_image = apply_filter_in_frequency_domain(cropped_image,filter_matrix)

# Display the filtered image
plt.imshow(filtered_image.real, cmap="gray")
plt.title(f"Filtered Image: {image_names[0]}")
plt.show()


filter_matrix = butterworth_filter(cropped_image.shape,5,order=3, pass_type="high")
# Apply filter in the frequency domain
filtered_image = apply_filter_in_frequency_domain(cropped_image,filter_matrix)

# Display the filtered image
plt.imshow(filtered_image.real, cmap="gray")
plt.title(f"Filtered Image: {image_names[0]}")
plt.show()

# # Save the filtered image
# filtered_image_path = os.path.join(output_dir, "filtered_" + image_name)
# io.imsave(filtered_image_path, filtered_image)


# # %%
# # Main processing
# image_dir = "images\\"
# image_names = ["crying-cat-meme.jpg", "crying-cat-meme.jpg"]
# output_dir = "outputs\\lab6"

# # Create output directory if not exists
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Process each image
# for image_name in image_names:
#     # Load and preprocess the image
#     image_path = os.path.join(image_dir, image_name)
#     cropped_image = crop_power2(image_path, save_path=os.path.join(output_dir, "cropped_" + image_name))
#     cropped_image = rgb2gray(cropped_image)
#     cropped_image = reduce_image(cropped_image, 4)

#     # Display the preprocessed image
#     plt.imshow(cropped_image, cmap="gray")
#     plt.title(f"Original Image: {image_name}")
#     plt.show()

#     # Apply filter in the frequency domain
#     filtered_image = apply_filter_in_frequency_domain(cropped_image,)

#     # Display the filtered image
#     plt.imshow(filtered_image, cmap="gray")
#     plt.title(f"Filtered Image: {image_name}")
#     plt.show()

#     # Save the filtered image
#     filtered_image_path = os.path.join(output_dir, "filtered_" + image_name)
#     io.imsave(filtered_image_path, filtered_image)

# %%
