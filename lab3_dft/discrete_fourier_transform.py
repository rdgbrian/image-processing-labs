import numpy as np # math li
import matplotlib.pyplot as plt
def rgb2gray(rgb): # Y' = 0.2989 R + 0.5870 G + 0.1140 B 
    """
    Function to turn an RGB image to gray scale
    rdg : an RGB image represented as a numpy array
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def discrete_fourier_transform(img,centered=False):

    if centered:
        shift = -1.
    else:
        shift = 1.

    input_shape = img.shape
    
    M = input_shape[1] # in x direction
    N = input_shape[0] # in y direction

    real = np.zeros(input_shape)
    imaginary = np.zeros(input_shape)
    for v in range(M):
        for u in range(N):
            for y in range(M): # y
                for x in range(N): # x 
                            # (-1)**(i+j)
                        real[v,u] += img[y,x] * (shift)**(y+x) * np.cos(-2*np.pi*(u*x/input_shape[1]+v*y/input_shape[0]))
                        imaginary[v,u] += img[y,x] * (shift)**(y+x) *np.sin(-2*np.pi*(u*x/input_shape[1]+v*y/input_shape[0]))
    
    F = np.array([real,imaginary])
    return F

def inverse_discrete_fourier_transform(img,centered=False):

    if centered:
        shift = -1.
    else:
        shift = 1.

    input_shape = img.shape
    
    M = input_shape[1] # in x direction
    N = input_shape[0] # in y direction

    real = np.zeros(input_shape)
    imaginary = np.zeros(input_shape)
    for v in range(M):
        for u in range(N):
            for y in range(M): # y
                for x in range(N): # x 
                            # (-1)**(i+j)
                        real[u,v] += img[y,x] * (shift)**(y+x) * np.cos(2*np.pi*(u*x/input_shape[1]+v*y/input_shape[0]))
                        imaginary[u,v] += img[y,x] * (shift)**(y+x) * np.sin(2*np.pi*(u*x/input_shape[1]+v*y/input_shape[0]))
    real = real / (M*N)
    imaginary = imaginary / (M*N)
    
    return real, imaginary


# def discrete_fourier_transform(img,centered=False):

#     if centered:
#         shift = -1.
#     else:
#         shift = 1.

#     input_shape = img.shape
#     real = np.zeros(input_shape)
#     imaginary = np.zeros(input_shape)
#     for v in range(input_shape[0]):
#         for u in range(input_shape[0]):
#             for i in range(input_shape[0]): # y
#                 for j in range(input_shape[1]): # x 
#                             # (-1)**(i+j)
#                         real[u,v] += img[i,j] * (shift)**(i+j) * -np.cos(2*np.pi*(u*j/input_shape[1]+v*i/input_shape[0]))
#                         imaginary[u,v] += img[i,j] * (shift)**(i+j) *-np.sin(2*np.pi*(u*j/input_shape[1]+v*i/input_shape[0]))
#     return real, imaginary


def power_spectrum(real, imaginary):
    return real**2 + imaginary**2
def fourier_spectrum(real, imaginary):
    return np.sqrt(real**2 + imaginary**2)

def create_white_square(n, square_size):
    # Create an n x n black background (all zeros)
    image = np.zeros((n, n), dtype=np.uint8)
    # Calculate the starting and ending points for the white square
    start = (n - square_size) // 2
    end = start + square_size
    # Create a white square (all ones) in the center of the black image
    image[start:end, start:end] = 255
    return image



