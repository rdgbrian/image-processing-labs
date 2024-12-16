import numpy as np # math li
import matplotlib.pyplot as plt
def dft2d(matrix, inverse=False, centered = False):
    """
    Compute the 2D Discrete Fourier Transform or its inverse.
    
    :param matrix: Input 2D array.
    :param inverse: If True, computes the inverse DFT. Default is False.
    :return: Transformed 2D array (frequency or spatial domain).
    """
    M, N = matrix.shape  # Get the dimensions of the input matrix
    dft_matrix = np.zeros((M, N), dtype=np.complex128)  # Initialize the output matrix
    factor = 1 / (M * N) if inverse else 1  # Scaling factor for the inverse transform
    sign = 1 if inverse else -1  # Sign of the exponent

    # sign_inv = 1 if centered else -1  # Sign of the exponent


    # Compute the 2D DFT or inverse DFT
    for u in range(M):
        for v in range(N):
            sum_value = 0
            for x in range(M):
                for y in range(N):
                    angle = sign * 2j * np.pi * ((u * x / M) + (v * y / N))
                    sum_value += matrix[x, y] * np.exp(angle)
            dft_matrix[u, v] = factor * sum_value
    
    return dft_matrix

# # Example usage
# input_matrix = np.array([[1, 2], [3, 4]], dtype=np.float64)  # A small 2x2 input
# dft_result = dft2d(input_matrix)  # Forward DFT
# idft_result = dft2d(dft_result, inverse=True)  # Inverse DFT

# print("Input Matrix:")
# print(input_matrix)
# print("\n2D DFT Result:")
# print(dft_result)
# print("\nReconstructed Matrix (Inverse DFT):")
# print(idft_result.real)  # Take the real part to avoid numerical noise



