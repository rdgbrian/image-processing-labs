import os
import matplotlib.pyplot as plt
import numpy as np

from src.utils import rgb2gray
from src.resize import crop_power2
from src.edge_operators import edge_detection_kirsch, edge_detection_sobel

# Define directories
image_dir = "images\\"
image_names = ["crying-cat-meme.jpg", "nebula.jpg", "squirrel.jpg", "beach_sunset.jpg"]
output_dir = "outputs\\lab10_edge_detection"  # Save all outputs here

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List of thresholds to apply
thresholds = [0.1, 0.2, 0.3, 0.4]

# Function to apply Kirsch edge detection
def apply_kirsch_edge_detection(image_name, thresholds):
    # Load and preprocess the image
    image_path = os.path.join(image_dir, image_name)
    cropped_image = crop_power2(image_path, save_path=None)
    cropped_image = np.array(cropped_image)
    cropped_image = rgb2gray(cropped_image)

    # Apply Kirsch edge detection
    kirsch_edges = edge_detection_kirsch(cropped_image)
    kirsch_edges = (kirsch_edges - kirsch_edges.min()) / (kirsch_edges.max() - kirsch_edges.min())

    # Apply different thresholds and save results
    for threshold in thresholds:
        thresholded_edges = kirsch_edges > threshold
        
        # Save the result
        output_path = os.path.join(
            output_dir, f"{image_name.split('.')[0]}_kirsch_edges_thresh_{threshold:.2f}.jpg"
        )
        plt.imshow(thresholded_edges, cmap="gray")
        plt.title(f"Kirsch Edge Detection (Threshold={threshold:.2f})")
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

# Function to apply Sobel edge detection
def apply_sobel_edge_detection(image_name, thresholds):
    # Load and preprocess the image
    image_path = os.path.join(image_dir, image_name)
    cropped_image = crop_power2(image_path, save_path=None)
    cropped_image = np.array(cropped_image)
    cropped_image = rgb2gray(cropped_image)

    # Apply Sobel edge detection
    sobel_edges = edge_detection_sobel(cropped_image)
    sobel_edges = (sobel_edges - sobel_edges.min()) / (sobel_edges.max() - sobel_edges.min())

    # Apply different thresholds and save results
    for threshold in thresholds:
        thresholded_edges = sobel_edges > threshold
        
        # Save the result
        output_path = os.path.join(
            output_dir, f"{image_name.split('.')[0]}_sobel_edges_thresh_{threshold:.2f}.jpg"
        )
        plt.imshow(thresholded_edges, cmap="gray")
        plt.title(f"Sobel Edge Detection (Threshold={threshold:.2f})")
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

# Main function to process all images
def main():
    for image_name in image_names:
        print(f"Processing {image_name} for multiple thresholds...")
        apply_kirsch_edge_detection(image_name, thresholds)
        apply_sobel_edge_detection(image_name, thresholds)
        print(f"Done processing {image_name}\n")

if __name__ == "__main__":
    main()
