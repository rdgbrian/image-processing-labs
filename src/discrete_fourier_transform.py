import numpy as np 
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
    sign_inv = 1 if inverse else -1  # Sign of the exponent
    sign_inv = -1 if centered else 1  # Sign of the exponent


    # Compute the 2D DFT or inverse DFT
    for u in range(M):
        for v in range(N):
            sum_value = 0
            for x in range(M):
                for y in range(N):
                    angle = sign_inv * 2j * np.pi * ((u * x / M) + (v * y / N))
                    sum_value += matrix[x, y] * sign_inv * np.exp(angle)
            dft_matrix[u, v] = factor * sum_value
    
    return dft_matrix



