import numpy as np  # math library
import matplotlib.pyplot as plt
from src.resize import crop_power2, reduce_image
from src.utils import rgb2gray, create_white_square

# Define directories
image_dir = "images\\"
image_names = ["crying-cat-meme.jpg", "nebula.jpg", "squirrel.jpg", "beach_sunset.jpg"]
output_dir = "outputs\\lab2_reduce"  # Save all output images here

# Create the function to process and plot images
def lab2_plot(image_name):
    # Crop the image to the nearest power of 2
    image = crop_power2(image_dir + image_name, save_path=None)  # We won't save cropped image here
    image = np.array(image)

    # Convert the image to grayscale
    image = rgb2gray(image)

    size = image.shape[0]
    num_reduce = 1

    while size > 8:
        # Reduce the image size
        reduced_image = reduce_image(image, num_red=num_reduce)

        # Update the size for the next iteration
        size = reduced_image.shape[0]

        # Generate the output file name
        output_file = f"{output_dir}\\{image_name.split('.')[0]}_{size}x{size}.jpg"

        # Save the reduced image with matplotlib
        plt.imshow(reduced_image, cmap='gray')
        plt.title(f"{image_name} - Shape: {size}x{size}")
        plt.axis('off')
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()

        # Prepare the image for the next reduction
        image = reduced_image

# Process all images
def main():
    for i, name in enumerate(image_names):
        lab2_plot(name)

if __name__ == "__main__":
    main()
