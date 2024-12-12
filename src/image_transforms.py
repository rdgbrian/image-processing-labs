# %% 
import numpy as np
# %% 
def tau(u,v,N,M):
     if (u == v) and (u == 0):
          return 1/np.sqrt(N*M)
     else:
          return 2/np.sqrt(N*M)
     

def discrete_cosine_transform(img):

    N, M= img.shape[0], img.shape[1] #

    transform = np.zeros_like(img)


    input_shape = img.shape
    M = input_shape[1] # in x direction
    N = input_shape[0] # in y direction

    real = np.zeros(input_shape)
    for v in range(N):
        for u in range(M):
            for y in range(N): # y
                for x in range(M): # x 
                        real[v,u] += tau(u,v,N,M) * img[y,x] * np.cos((2*x+1)*u*np.pi/(2*M)) * np.cos((2*y+1)*v*np.pi/(2*N))
    return real


import numpy as np

# %% 
def hadamard_kernel(n):
    """
    Generates the Hadamard transform kernel of size n x n.

    Parameters:
        n (int): Size of the Hadamard matrix. Must be a power of 2.

    Returns:
        numpy.ndarray: The Hadamard transform kernel.
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of 2.")

    # Initialize the 1x1 Hadamard matrix
    H = np.array([[1]])

    # Recursive construction
    while H.shape[0] < n:
        H = np.block([
            [H, H],
            [H, -H]
        ])

    return H

# Example usage
n = 8  # Size of the Hadamard matrix
kernel = hadamard_kernel(n)
print(kernel)
# %% 


def b_i(z, i):
    """
    Returns the i-th bit of the binary representation of z.
    """
    return (z >> i) & 1

def h(x, u, n):
    """
    Computes the value of h(x, u) for given x, u, and n.
    
    Parameters:
        x (int): The x value in the kernel.
        u (int): The u value in the kernel.
        n (int): The dimension (number of bits).

    Returns:
        int: The product (-1)^(b_i(x) * b_{n-1-i}(u)) for i=0 to n-1.
    """
    product = 1
    for i in range(n):
        product *= (-1) ** (b_i(x, i) * b_i(u, n - 1 - i))
    return product

def walsh_kernel(n):
    """
    Computes the kernel matrix of the ordered Walsh transform for a given dimension.

    Parameters:
        n (int): The dimension (number of bits).

    Returns:
        np.ndarray: The kernel matrix of size 2^n x 2^n.
    """
    size = 2 ** n
    kernel = np.zeros((size, size), dtype=int)
    
    for x in range(size):
        for u in range(size):
            kernel[x, u] = h(x, u, n)
    
    return kernel

# Example usage
n = 3  # Number of bits (dimension)
kernel_matrix = walsh_kernel(n)
print("Kernel of the ordered Walsh transform:")
print(kernel_matrix)

# %%
