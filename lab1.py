import os
import numpy as np
import matplotlib.pyplot as plt

from src.resize import crop_power2
from src.utils import rgb2gray

# Define directories
image_dir = "images\\"
image_names = ["crying-cat-meme.jpg", "nebula.jpg", "squirrel.jpg", "beach_sunset.jpg","camel.jpg","random.png"]
output_dir = "outputs\\lab1_inverse"  # Save all outputs here

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to invert and save an image using matplotlib
def invert_and_save_image(image_name):
    # Load and preprocess the image
    image_path = os.path.join(image_dir, image_name)
    cropped_image = crop_power2(image_path, save_path=None)
    cropped_image = np.array(cropped_image)
    grayscale_image = rgb2gray(cropped_image)

    # Save the grayscale image using plt
    plt.imshow(grayscale_image, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"{image_name[:-4]}_grey.png"), bbox_inches='tight')
    plt.close()

    # Invert the image
    inverted_image = 255 - grayscale_image

    # Save the inverted image using plt
    plt.imshow(inverted_image, cmap="gray")
    plt.title("Inverted Image")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"{image_name[:-4]}_inv.png"), bbox_inches='tight')
    plt.close()

# Main function to process all images
def main():
    for image_name in image_names:
        invert_and_save_image(image_name)

if __name__ == "__main__":
    main()
