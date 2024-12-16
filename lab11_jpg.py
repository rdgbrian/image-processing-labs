# %%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.jpeg import jpeg_encode, dct_quantize_matrix, zigzag_scan, block_to_code
from src.utils import rgb2gray
from src.resize import crop_power2
# %%

# Example usage
image_path = "images\\juliana.jpg"
cropped_image = crop_power2(image_path)
cropped_image = rgb2gray(cropped_image)
image = cropped_image
# cropped_image = reduce_image(cropped_image, 4)

# Display the preprocessed image
plt.imshow(cropped_image, cmap="gray")
plt.title(f"Original Image: {image_path}")
plt.show()

# %%
# # Define a basic quantization matrix
# quantization_matrix = np.ones((8, 8)) * 50  # Simple example, real JPEG uses much more complex matrices

# Encode the image
code = jpeg_encode(image)

# %%
# blocks = dct_quantize_matrix(image)
# zig_blocks = [zigzag_scan(block) for block in blocks]

temp = [-35,0,0,0,-2,-1,0,0,4,0,0,0,0,0,1]
temp_code = block_to_code(temp,prev_dc=-8)
temp_code
# %%
