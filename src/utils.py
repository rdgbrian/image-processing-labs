import numpy as np # math li
import matplotlib.pyplot as plt
from PIL import Image
import os

def rgb2gray(rgb): # Y' = 0.2989 R + 0.5870 G + 0.1140 B 
    """
    Function to turn an RGB image to gray scale
    rdg : an RGB image represented as a numpy array
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def mult_complex(c1, c2):
    real_part = c1[0] * c2[0] - c1[1] * c2[1]
    imaginary_part = c1[0] * c2[1] + c1[1] * c2[0]
    return np.array([real_part, imaginary_part])

def old_center_fourier(real_part, imag_part):
    """
    Centers the Fourier transform by manually shifting the zero-frequency component to the center.

    Parameters:
    real_part (np.array): The real part of the Fourier transform.
    imag_part (np.array): The imaginary part of the Fourier transform.

    Returns:
    tuple: Centered real and imaginary parts of the Fourier transform.
    """
    # Get the number of rows and columns in the arrays
    rows, cols = real_part.shape
    
    # Split the array into four quadrants and rearrange them to center the zero-frequency component
    centered_real = np.empty_like(real_part)
    centered_imag = np.empty_like(imag_part)

    # Top-left -> Bottom-right
    centered_real[:rows//2, :cols//2] = real_part[rows//2:, cols//2:]
    centered_imag[:rows//2, :cols//2] = imag_part[rows//2:, cols//2:]
    
    # Bottom-right -> Top-left
    centered_real[rows//2:, cols//2:] = real_part[:rows//2, :cols//2]
    centered_imag[rows//2:, cols//2:] = imag_part[:rows//2, :cols//2]

    # Top-right -> Bottom-left
    centered_real[:rows//2, cols//2:] = real_part[rows//2:, :cols//2]
    centered_imag[:rows//2, cols//2:] = imag_part[rows//2:, :cols//2]

    # Bottom-left -> Top-right
    centered_real[rows//2:, :cols//2] = real_part[:rows//2, cols//2:]
    centered_imag[rows//2:, :cols//2] = imag_part[:rows//2, cols//2:]

    return centered_real, centered_imag


def center_fourier(matrix):
    """
    Centers the Fourier transform by manually shifting the zero-frequency component to the center.

    Parameters:
    real_part (np.array): The real part of the Fourier transform.
    imag_part (np.array): The imaginary part of the Fourier transform.

    Returns:
    tuple: Centered real and imaginary parts of the Fourier transform.
    """
    # Get the number of rows and columns in the arrays
    rows, cols = matrix.shape
    
    # Split the array into four quadrants and rearrange them to center the zero-frequency component
    centered_matrix = np.empty_like(matrix)

    # Top-left -> Bottom-right
    centered_matrix[:rows//2, :cols//2] = matrix[rows//2:, cols//2:]
    
    # Bottom-right -> Top-left
    centered_matrix[rows//2:, cols//2:] = matrix[:rows//2, :cols//2]

    # Top-right -> Bottom-left
    centered_matrix[:rows//2, cols//2:] = matrix[rows//2:, :cols//2]

    # Bottom-left -> Top-right
    centered_matrix[rows//2:, :cols//2] = matrix[:rows//2, cols//2:]

    return centered_matrix


def power_spectrum(real, imaginary):
    return real**2 + imaginary**2
def fourier_spectrum(real, imaginary):
    return np.sqrt(real**2 + imaginary**2)

def create_white_square(n, square_size):
    # Create an n x n black background (all zeros)
    image = np.zeros((n, n))
    # Calculate the starting and ending points for the white square
    start = (n - square_size) // 2
    end = start + square_size
    # Create a white square (all ones) in the center of the black image
    image[start:end, start:end] = 255
    return image

def save_transforms(image_name, transform_dict, output_dir, add_log = True):
    """
    Save each transform in the dictionary to the output directory.

    :param image_name: Name of the original image file (used for naming outputs).
    :param transform_dict: Dictionary of transforms (e.g., {"dct": dct_image, ...}).
    :param output_dir: Directory to save the transformed images.
    """
    for name, transform in transform_dict.items():
        # Normalize the transform for visualization
        if(add_log):
            normalized = (255 / np.log10(255)) * np.log10(1 + 255/(np.max(transform)) * np.abs(transform))  # log1p(x) = log(1 + x), safe for 0 values

        else:
            normalized = (transform - transform.min()) / (transform.max() - transform.min()) * 255
        normalized = normalized.astype('uint8')

        # Create a Pillow image from the NumPy array
        transformed_image = Image.fromarray(normalized)

        # Save to the output directory
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_{name}.png")
        transformed_image.save(output_path)
        print(f"Saved {name} transform to {output_path}")

def save_image(image,out_path):
    
    plt.axis('off')  # Turns off the axis lines and labels
    plt.imshow(image,cmap="gray")
    plt.savefig(out_path)     
    plt.show()