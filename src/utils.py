import numpy as np # math li
import matplotlib.pyplot as plt

def center_fourier(real_part, imag_part):
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
