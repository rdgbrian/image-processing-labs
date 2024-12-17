import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from src.image_transforms import dct2d, haar_transform, hadamard_transform, walsh_transform
from src.utils import create_white_square, rgb2gray, save_transforms
from src.resize import crop_power2, reduce_image

# Define directories
image_dir = "images\\"
# image_names = ["crying-cat-meme.jpg", "nebula.jpg", "squirrel.jpg", "beach_sunset.jpg"]
image_names = ["random.png"]

output_dir = "outputs\\lab4_transforms"  # Save all transform outputs here

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to apply and save transforms for an image
def save_image_transforms(image_name,add_log=True):
    # Load and preprocess the image
    image_path = os.path.join(image_dir, image_name)
    cropped_image = crop_power2(image_path, save_path=None)  # Crop to power of 2
    cropped_image = np.array(cropped_image)
    cropped_image = rgb2gray(cropped_image)

    # Reduce the image size
    cropped_image = reduce_image(cropped_image, 1)
    size = cropped_image.shape[0]

    # Perform the transformations
    transforms = {
        "dct": dct2d(cropped_image),
        "walsh": walsh_transform(cropped_image),
        # "hadamard": hadamard_transform(cropped_image),
        "haar": haar_transform(cropped_image)
    }

    # Save each transform as an image
    for transform_name, transform_data in transforms.items():

        if(add_log):
            transform_data = (255 / np.log10(255)) * np.log10(1 + 255/(np.max(transform_data)) * np.abs(transform_data))  # log1p(x) = log(1 + x), safe for 0 values

        plt.imshow(transform_data, cmap='gray')
        plt.title(f"{image_name} - {transform_name.upper()} Transform")
        plt.axis('off')
        plt.savefig(f"{output_dir}\\{image_name.split('.')[0]}_{transform_name}_{size}x{size}.jpg", bbox_inches='tight')
        plt.close()

# Function to perform a sanity check on transforms
def sanity_check(image):
    for transform, func in zip([
        "DCT", "Walsh", "Hadamard", "Haar"],
        [dct2d, walsh_transform, hadamard_transform, haar_transform]):

        print(f"Performing sanity check for {transform} transform...")
        transform_og = func(image)
        plt.imshow(transform_og, cmap='gray')
        plt.title(f"{transform} Transform")
        plt.show()

        if transform == "DCT" or transform == "Haar":
            image_rec = func(transform_og, inverse=True)
        else:
            image_rec = func(transform_og)

        plt.imshow(image_rec, cmap='gray')
        plt.title(f"Reconstructed Image ({transform})")
        plt.show()

        are_equal = np.allclose(image_rec, image, rtol=1e-6, atol=1e-9)
        print(f"{transform} Transform: Are original and reconstructed images equal? {are_equal}")

# Main function to process all images
def main():
    for image_name in image_names:
        save_image_transforms(image_name,True)
        
    # Example sanity check on a created white square
    test_image = create_white_square(32, 8)
    sanity_check(test_image)

if __name__ == "__main__":
    main()
