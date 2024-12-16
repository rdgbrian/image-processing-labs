# %%
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from src.image_transforms import dct2d, haar_transform, hadamard_transform, walsh_transform
from src.utils import create_white_square, rgb2gray, save_transforms
from src.resize import crop_power2, reduce_image

# %%
# Example: 10x10 image with a 4x4 white square in the center
n = 32
square_size = 8
image = create_white_square(n, square_size)

# image = plt.imread("Lab1-Inverting_Image/crying-cat-meme.jpg")
print(image.shape) # show the dimensions of the image
plt.imshow(image)
plt.show()

# %%
cropped_image = crop_power2("images\crying-cat-meme.jpg", save_path="output_image.jpg")
# cropped_image = np.array(cropped_image)
cropped_image = rgb2gray(cropped_image)


cropped_image = reduce_image(cropped_image,4)
plt.imshow(cropped_image,cmap="grey")
plt.show()
image = cropped_image

# %%

# Main script
image_dir = "images\\"
image_names = ["crying-cat-meme.jpg"]
output_dir = "outputs\\lab4"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Store all transforms in a list
transforms_list = []

for image_path in image_names:
    path = os.path.join(image_dir, image_path)
    cropped_image = crop_power2(path)  # Pass the actual file path
    cropped_image = rgb2gray(cropped_image)
    cropped_image = reduce_image(cropped_image,4)
    size = cropped_image.shape[0]

    # Perform the transformations and store them in a dictionary
    transform_dict = {
        "dct": dct2d(cropped_image),
        "walsh": walsh_transform(cropped_image),
        "hadamard": hadamard_transform(cropped_image),
        "haar": haar_transform(cropped_image)
    }

    # Add the transform dictionary to the list
    transforms_list.append((image_path, transform_dict))

# Save all transforms in the list
for image_name, transform_dict in transforms_list:
    save_transforms(image_name + "_size", transform_dict, output_dir,add_log=True)



     
# %%

def sanity_check(image):
    dct_og = dct2d(image)
    plt.imshow(dct_og)
    plt.show()
    image_rec = dct2d(dct_og,inverse=True)
    plt.imshow(image_rec,cmap="grey")
    plt.show()
    are_equal = np.allclose(image_rec,image, rtol=1e-6, atol=1e-9)
    print("Are the arrays equal to the given precision?", are_equal)
    
    walsh_og = walsh_transform(image)
    plt.imshow(walsh_og)
    plt.show()
    image_rec = walsh_transform(walsh_og)
    plt.imshow(image_rec,cmap="grey")
    plt.show()
    are_equal = np.allclose(image_rec,image, rtol=1e-6, atol=1e-9)
    print("Are the arrays equal to the given precision?", are_equal)

    hadamard_og = hadamard_transform(image)
    plt.imshow(walsh_og)
    plt.show()
    image_rec = hadamard_transform(hadamard_og)
    plt.imshow(image_rec,cmap="grey")
    plt.show()
    are_equal = np.allclose(image_rec,image, rtol=1e-6, atol=1e-9)
    print("Are the arrays equal to the given precision?", are_equal)
    
    haar_og = haar_transform(image)
    plt.imshow(haar_og)
    plt.show()
    image_rec = haar_transform(haar_og,inverse=True)
    plt.imshow(image_rec,cmap="grey")
    plt.show()
    are_equal = np.allclose(image_rec,image, rtol=1e-6, atol=1e-9)
    print("Are the arrays equal to the given precision?", are_equal)
# %%
