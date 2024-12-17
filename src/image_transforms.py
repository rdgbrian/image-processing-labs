# %% 
import numpy as np
# %% 
import numpy as np

def tau(u, v, N, M):
    """
    Normalization factor for DCT-II.
    """
    alpha_u = 1 / np.sqrt(N) if u == 0 else np.sqrt(2 / N)
    alpha_v = 1 / np.sqrt(M) if v == 0 else np.sqrt(2 / M)
    return alpha_u * alpha_v

def dct2d(img, inverse=False):
    """
    Perform 2D Discrete Cosine Transform (DCT-II) or its inverse.
    Args:
        img (ndarray): Input image as a 2D numpy array.
        inverse (bool): If True, compute the inverse DCT.
    Returns:
        ndarray: Transformed image as a 2D numpy array.
    """
    N, M = img.shape[0], img.shape[1]
    transform = np.zeros_like(img, dtype=float)

    if not inverse:
        for u in range(N):
            for v in range(M):
                for x in range(N):
                    for y in range(M):
                        transform[u, v] += (
                            tau(u, v, N, M)
                            * img[x, y]
                            * np.cos((2 * x + 1) * u * np.pi / (2 * M))
                            * np.cos((2 * y + 1) * v * np.pi / (2 * N))
                        )
    else:
        for x in range(N):
            for y in range(M):
                for u in range(N):
                    for v in range(M):
                        transform[x, y] += (
                            tau(u, v, N, M)
                            * img[u, v]
                            * np.cos((2 * x + 1) * u * np.pi / (2 * M))
                            * np.cos((2 * y + 1) * v * np.pi / (2 * N))
                        )

    return transform
# %% 
def hadamard_kernel(size):
    """
    Generates the Hadamard transform kernel of size n x n.

    Parameters:
        size (int): Size of the Hadamard matrix. Must be a power of 2.

    Returns:
        numpy.ndarray: The Hadamard transform kernel.
    """
    if size < 1 or (size & (size - 1)) != 0:
        raise ValueError("n must be a power of 2.")
    
    # Initialize the 1x1 Hadamard matrix
    H = np.array([[1]])

    # Recursive construction
    while H.shape[0] < size:
        H = np.block([
            [H, H],
            [H, -H]
        ])

    return H

# # Example usage
# n = 2**3  # Size of the Hadamard matrix
# kernel = hadamard_kernel(n)
# print(kernel)
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

def walsh_kernel(size):
    """
    Computes the kernel matrix of the ordered Walsh transform for a given dimension.

    Parameters:
        n (int): The dimension (number of bits).

    Returns:
        np.ndarray: The kernel matrix of size 2^n x 2^n.
    """
    if not (size > 0 and (size & (size - 1)) == 0):
        raise ValueError("Size must be a power of 2.")
    
    kernel = np.zeros((size, size), dtype=int)
    
    n = int(np.log2(size))
    for x in range(size):
        for u in range(size):
            kernel[x, u] = h(x, u, n)
    
    return kernel

# Example usage
# n = 2**3  # Number of bits (dimension)
# kernel_matrix = walsh_kernel(n)
# print("Kernel of the ordered Walsh transform:")
# print(kernel_matrix)

# %%

import numpy as np

def HA(r,m,x):
    if r == 0 and m==0:
        return 1
   
    value = 0
    if (m-1)/(2**r) <= x and x < (m-0.5)/(2**r):
        value = 2**(r/2)
    
    elif (m-0.5)/(2**r) <= x and x < (m)/(2**r):
        value = -2**(r/2)

    return value

def haar_kernel(size):
    """
    Creates the Haar transform kernel for a given size.

    Parameters:
        size (int): Size of the Haar transform (must be a power of 2).

    Returns:
        numpy.ndarray: The Haar transform kernel of shape (size, size).
    """
    if not (size > 0 and (size & (size - 1)) == 0):
        raise ValueError("Size must be a power of 2.")
    
    haar = np.zeros((size, size), dtype=float)

    haar[0] = np.ones((size),dtype=float)
    print(f"r={0},m={0}")
    
    u_count = 1
    for r in range(0,int(np.log2(size))):
        for m in range(1,2**r +1):
            for i in range(size):
                x = i/size
                haar[u_count,i] = HA(r,m,x)
            print(f"r={r},m={m}")
            u_count += 1


    return haar

# # Example usage
# n = 2**2  # Number of bits (dimension)
# kernel_matrix = haar_kernel(n)
# print("Kernel of the Haar transform:")
# print(kernel_matrix)


# %%


def hadamard_transform(img):
    N, M= img.shape[0], img.shape[1] #

    k = hadamard_kernel(N)
    k = (1/np.sqrt(N)) * k
    result = np.dot(img, k)
    result = np.dot(k,result)

    return result

def walsh_transform(img):
    N, M= img.shape[0], img.shape[1] #

    k = walsh_kernel(N)
    k = (1/np.sqrt(N)) * k
    result = np.dot(img, k)
    result = np.dot(k,result)

    return result

def haar_transform(img,inverse=False):
    N, M= img.shape[0], img.shape[1] #

    k = walsh_kernel(N)
    k = (1/np.sqrt(N)) * k
    if inverse:
        result = np.dot(img, k.T)
        result = np.dot(k,result)
    else:
        result = np.dot(img, k)
        result = np.dot(k.T,result)

    return result
