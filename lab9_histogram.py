import os
import matplotlib.pyplot as plt
import numpy as np

from src.utils import rgb2gray
from src.resize import crop_power2
from src.histogram import histogram_equalization, histogram_dynamic_range_modification

# Define directories
image_dir = "images\\"
image_names = ["crying-cat-meme.jpg", "nebula.jpg", "squirrel.jpg", "beach_sunset.jpg","camel.jpg"]
output_dir = "outputs\\lab9_histogram_modifications"  # Save all outputs here

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to apply histogram equalization
def apply_histogram_equalization(image_name, num_bins_list):
    # Load and preprocess the image
    image_path = os.path.join(image_dir, image_name)
    cropped_image = crop_power2(image_path, save_path=None)
    cropped_image = np.array(cropped_image)
    cropped_image = rgb2gray(cropped_image)

    for num_bins in num_bins_list:
        # Apply histogram equalization
        equalized_image = histogram_equalization(cropped_image, num_bins=num_bins)

        # Save the result
        plt.imshow(equalized_image, cmap="gray")
        plt.title(f"Histogram Equalization (Bins={num_bins})")
        plt.axis('off')
        plt.savefig(f"{output_dir}\\{image_name.split('.')[0]}_equalized_bins{num_bins}.jpg", bbox_inches='tight')
        plt.close()

# Function to apply dynamic range modification
def apply_histogram_dynamic_range_modification(image_name):
    # Load and preprocess the image
    image_path = os.path.join(image_dir, image_name)
    cropped_image = crop_power2(image_path, save_path=None)
    cropped_image = np.array(cropped_image)
    cropped_image = rgb2gray(cropped_image)

    # Apply dynamic range modification
    modified_image = histogram_dynamic_range_modification(cropped_image)

    # Save the result
    plt.imshow(modified_image, cmap="gray")
    plt.title("Dynamic Range Modification")
    plt.axis('off')
    plt.savefig(f"{output_dir}\\{image_name.split('.')[0]}_dynamic_range.jpg", bbox_inches='tight')
    plt.close()

# Main function to process all images
def main():
    num_bins_list = [5, 10, 15, 20]  # Different bin sizes to test

    for image_name in image_names:
        apply_histogram_equalization(image_name, num_bins_list)
        apply_histogram_dynamic_range_modification(image_name)

if __name__ == "__main__":
    main()
