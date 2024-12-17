import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.jpeg import jpeg_encode, block_to_code
from src.utils import rgb2gray
from src.resize import crop_power2, reduce_image

# Define directories
image_dir = "images\\"
image_names = ["crying-cat-meme.jpg", "squirrel.jpg", "beach_sunset.jpg","random.png"]
output_dir = "outputs\\lab11_jpeg_codes"  # Save outputs here

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to calculate JPEG code for an image
def calculate_jpeg_code(image_name):
    # Load and preprocess the image
    image_path = os.path.join(image_dir, image_name)
    cropped_image = crop_power2(image_path, save_path=None)
    cropped_image = np.array(cropped_image)
    cropped_image = rgb2gray(cropped_image)

    if image_name == "beach_sunset.jpg":
        cropped_image = reduce_image(cropped_image)
    
    if image_name == "random.png":
        rows, cols = 512, 512
        # Create a random array with values between 0 and 255
        cropped_image = np.random.randint(0, 256, size=(rows, cols), dtype=np.uint8)
    
    print(cropped_image.shape)
    # # Encode the image into JPEG binary code
    jpeg_code = jpeg_encode(cropped_image)

    return jpeg_code

# Function for a sanity check
def sanity_check():
    temp = [-35, 0, 0, 0, -2, -1, 0, 0, 4, 0, 0, 0, 0, 0, 1]
    temp_code = block_to_code(temp, prev_dc=-8)
    print("Example jpg block:", temp)
    print("Sanity Check Code:", temp_code)
    return temp_code

# Function to create a bar graph of JPEG code lengths
def create_bar_graph(image_names, jpeg_codes):
    code_lengths = [len(code) for code in jpeg_codes]

    plt.figure(figsize=(10, 6))
    plt.bar(image_names, code_lengths, color='skyblue', edgecolor='black')
    plt.xlabel("Image Names")
    plt.ylabel("JPEG Code Length")
    plt.title("JPEG Code Lengths per Image")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "jpeg_code_lengths_bar_graph.jpg"))
    plt.close()

# Main function to process all images
def main():
    jpeg_codes = []

    # for image_name in image_names:
    #     print(f"Processing {image_name}...")
    #     jpeg_code = calculate_jpeg_code(image_name)
    #     jpeg_codes.append(jpeg_code)

    # # Create bar graph for JPEG code lengths
    # create_bar_graph(image_names, jpeg_codes)

    # Run sanity check
    sanity_check()

if __name__ == "__main__":
    main()
