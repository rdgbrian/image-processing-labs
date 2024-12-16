import numpy as np
import matplotlib.pyplot as plt
from src.fast_fourier_transform import fft2d
from src.utils import center_fourier


def ideal_filter(shape, cutoff, pass_type='low'):
    """
    Generate an Ideal Low-Pass or High-Pass Filter in the spatial domain.

    Parameters:
        shape (tuple): Shape of the filter (rows, columns).
        cutoff (float): Cutoff frequency.
        pass_type (str): 'low' for low-pass, 'high' for high-pass.
    
    Returns:
        2D numpy array: The ideal filter.
    """
    rows, cols = shape
    # Create a grid of coordinates
    u = np.arange(-rows//2, rows//2)
    v = np.arange(-cols//2, cols//2)
    U, V = np.meshgrid(u, v)
    
    # Calculate the Euclidean distance in the spatial domain
    D = np.sqrt(U**2 + V**2)

    # Create the ideal low-pass or high-pass filter
    if pass_type == 'low':
        H = np.where(D <= cutoff, 1, 0)
    elif pass_type == 'high':
        H = np.where(D > cutoff, 1, 0)
    else:
        raise ValueError("pass_type must be 'low' or 'high'")
    
    return H

def butterworth_filter(shape, cutoff, order, pass_type='low'):
    """
    Generate a Butterworth Low-Pass or High-Pass Filter in the spatial domain.

    Parameters:
        shape (tuple): Shape of the filter (rows, columns).
        cutoff (float): Cutoff frequency.
        order (int): The order of the Butterworth filter.
        pass_type (str): 'low' for low-pass, 'high' for high-pass.
    
    Returns:
        2D numpy array: The Butterworth filter.
    """
    rows, cols = shape
    # Create a grid of coordinates
    u = np.arange(-rows//2, rows//2)
    v = np.arange(-cols//2, cols//2)
    U, V = np.meshgrid(u, v)
    
    # Calculate the Euclidean distance in the spatial domain
    D = np.sqrt(U**2 + V**2)

    # Create the Butterworth filter
    if pass_type == 'low':
        H = 1 / (1 + (D / cutoff)**(2 * order))
    elif pass_type == 'high':
        H = 1 / (1 + (cutoff / D)**(2 * order))
    else:
        raise ValueError("pass_type must be 'low' or 'high'")
    
    return H

def gausian_fourier(shape, sigma):
    """
    Generate a 2D matrix of values computed from the function:
    e^(-(u^2 + v^2) * sigma^2 / 2), where (u, v) are coordinates centered at 0.

    Parameters:
        size (int): The size of the matrix (size x size). Must be even or odd.
        sigma (float): The standard deviation parameter for the function.

    Returns:
        np.ndarray: A 2D matrix of the computed values.
    """
    # if size <= 0 or not isinstance(size, int):
    #     raise ValueError("Size must be a positive integer.")

    rows, cols = shape
    # Create a grid of coordinates
    u = np.arange(-rows//2, rows//2)
    v = np.arange(-cols//2, cols//2)

    # Define the grid range centered at 0
    # mid = size // 2
    # u = np.arange(-mid, mid + 1) if size % 2 != 0 else np.arange(-mid, mid)
    # v = np.arange(-mid, mid + 1) if size % 2 != 0 else np.arange(-mid, mid)

    # Create a 2D grid of coordinates
    U, V = np.meshgrid(u, v)

    # Calculate the Gaussian function
    gaussian_matrix = np.exp(-((U**2 + V**2) * sigma**2) / 2)

    return gaussian_matrix

def lapacian_fourier(shape):
    """
    Generate a 2D matrix of values computed from the function:
    -(u^2 + v^2), where (u, v) are coordinates centered at 0.

    Parameters:
        shape 

    Returns:
        np.ndarray: A 2D matrix of the computed values.
    """
    # if size <= 0 or not isinstance(size, int):
    #     raise ValueError("Size must be a positive integer.")
    rows, cols = shape
    # Create a grid of coordinates
    u = np.arange(-rows//2, rows//2)
    v = np.arange(-cols//2, cols//2)

    # Define the grid range centered at 0
    # mid = size // 2
    # u = np.arange(-mid, mid + 1) if size % 2 != 0 else np.arange(-mid, mid)
    # v = np.arange(-mid, mid + 1) if size % 2 != 0 else np.arange(-mid, mid)

    # Create a 2D grid of coordinates
    U, V = np.meshgrid(u, v)

    # Calculate the quadratic function
    quadratic_matrix = -(U**2 + V**2)

    return quadratic_matrix

def lapacian_gausian_fourier(shape,sigma):
    return lapacian_fourier(shape) * gausian_fourier(shape, sigma)


# # Example usage
# size = 100  # Even-sized matrix
# sigma = 0.01
# matrix = lapacian_gausian_fourier(size, sigma)
# plt.imshow(matrix)
# plt.show()

def apply_filter_in_frequency_domain(image, filter_matrix):
    # Compute the FFT of the image
    fft_image, _ = fft2d(image, inverse=False)
    fft_image_centered = center_fourier(fft_image)
    fft_image_filtered_centered = fft_image_centered * filter_matrix
    fft_image_filtered = center_fourier(fft_image_filtered_centered)
    # Reconstruct the complex FFT and perform the inverse FFT
    filtered_image, _ = fft2d(fft_image_filtered, inverse=True)
    return filtered_image

# # Example Usage
# shape = (512, 512)
# cutoff = 50  # Cutoff frequency
# order = 2    # Butterworth filter order

# # Generate filters
# ideal_low_pass = ideal_filter(shape, cutoff, pass_type='low')
# ideal_high_pass = ideal_filter(shape, cutoff, pass_type='high')
# butterworth_low_pass = butterworth_filter(shape, cutoff, order, pass_type='low')
# butterworth_high_pass = butterworth_filter(shape, cutoff, order, pass_type='high')

# # Plotting
# fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# axs[0, 0].imshow(ideal_low_pass, cmap='gray')
# axs[0, 0].set_title("Ideal Low-Pass Filter")
# axs[0, 1].imshow(ideal_high_pass, cmap='gray')
# axs[0, 1].set_title("Ideal High-Pass Filter")
# axs[1, 0].imshow(butterworth_low_pass, cmap='gray')
# axs[1, 0].set_title("Butterworth Low-Pass Filter")
# axs[1, 1].imshow(butterworth_high_pass, cmap='gray')
# axs[1, 1].set_title("Butterworth High-Pass Filter")

# plt.show()
