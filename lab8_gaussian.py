# %%
import matplotlib.pyplot as plt
import numpy as np

from src.discrete_fourier_transform import dft2d
from src.fast_fourier_transform import fft2d
from src.filters import lapacian_gausian_fourier, apply_filter_in_frequency_domain

from src.utils import fourier_spectrum, create_white_square, rgb2gray, center_fourier, save_transforms
from src.resize import crop_power2, reduce_image
from time import time
from PIL import Image

# %%
image_dir = "images\\"
image_names = ["crying-cat-meme.jpg", "juliana.jpg"]
output_dir = "outputs\\lab6"

image_path = image_dir + image_names[1]
cropped_image = crop_power2(image_path)
cropped_image = rgb2gray(cropped_image)
# cropped_image = reduce_image(cropped_image, 4)

# Display the preprocessed image
plt.imshow(cropped_image, cmap="gray")
plt.title(f"Original Image: {image_names[0]}")
plt.show()

# %%

filter_matrix = lapacian_gausian_fourier(cropped_image.shape,0.01)
# Apply filter in the frequency domain
filtered_image = apply_filter_in_frequency_domain(cropped_image,filter_matrix)

filtered_image = (filtered_image - filtered_image.min())/(filtered_image.max() - filtered_image.min())

# %%
edges = filtered_image.real > 0.55
# Display the filtered image
plt.imshow(edges, cmap="gray")
plt.title(f"laplacian of gaussian: {image_names[1]}")
plt.show()
# %%
