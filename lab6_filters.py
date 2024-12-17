import matplotlib.pyplot as plt
import numpy as np
import os
from src.filters import ideal_filter, butterworth_filter, apply_filter_in_frequency_domain
from src.utils import rgb2gray
from src.resize import crop_power2

# Define directories
image_dir = "images\\"
image_names = ["crying-cat-meme.jpg", "nebula.jpg", "squirrel.jpg", "beach_sunset.jpg"]
output_dir = "outputs\\lab6_filters"  # Save all filtered images here

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Thresholds and orders to use
thresholds = [10, 20, 40, 80]  # Four thresholds
orders = [1, 2, 4, 8]          # Four orders (for Butterworth)

def save_filtered_image(image, filter_name, image_name, threshold, order=None):
    """
    Save the filtered image to the output directory with a meaningful name.
    """
    order_str = f"_order{order}" if order else ""
    output_path = os.path.join(output_dir, f"{image_name.split('.')[0]}_{filter_name}_T{threshold}{order_str}.jpg")
    plt.imshow(image.real, cmap="gray")
    plt.title(f"{filter_name} T={threshold} {'Order=' + str(order) if order else ''}")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# Function to apply filters and save results
def apply_and_save_all_filters(image_name):
    # Load and preprocess the image
    image_path = os.path.join(image_dir, image_name)
    cropped_image = crop_power2(image_path, save_path=None)  # Crop to power of 2
    cropped_image = np.array(cropped_image)
    cropped_image = rgb2gray(cropped_image)

    # Apply Ideal Filters for different thresholds
    for threshold in thresholds:
        print(f"Applying Ideal Low Pass (T={threshold}) to {image_name}...")
        low_pass_filter = ideal_filter(cropped_image.shape, threshold, pass_type="low")
        filtered_image = apply_filter_in_frequency_domain(cropped_image, low_pass_filter)
        save_filtered_image(filtered_image, "Ideal_LowPass", image_name, threshold)

        print(f"Applying Ideal High Pass (T={threshold}) to {image_name}...")
        high_pass_filter = ideal_filter(cropped_image.shape, threshold, pass_type="high")
        filtered_image = apply_filter_in_frequency_domain(cropped_image, high_pass_filter)
        save_filtered_image(filtered_image, "Ideal_HighPass", image_name, threshold)

    # Apply Butterworth Filters for different thresholds and orders
    for threshold in thresholds:
        for order in orders:
            print(f"Applying Butterworth Low Pass (T={threshold}, Order={order}) to {image_name}...")
            low_pass_filter = butterworth_filter(cropped_image.shape, threshold, order, pass_type="low")
            filtered_image = apply_filter_in_frequency_domain(cropped_image, low_pass_filter)
            save_filtered_image(filtered_image, "Butterworth_LowPass", image_name, threshold, order)

            print(f"Applying Butterworth High Pass (T={threshold}, Order={order}) to {image_name}...")
            high_pass_filter = butterworth_filter(cropped_image.shape, threshold, order, pass_type="high")
            filtered_image = apply_filter_in_frequency_domain(cropped_image, high_pass_filter)
            save_filtered_image(filtered_image, "Butterworth_HighPass", image_name, threshold, order)

# Main function to process all images
def main():
    for image_name in image_names:
        print(f"Processing image: {image_name}...\n")
        apply_and_save_all_filters(image_name)

if __name__ == "__main__":
    main()
