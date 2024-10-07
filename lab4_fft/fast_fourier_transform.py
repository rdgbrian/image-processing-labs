
import numpy as np # math li
import matplotlib.pyplot as plt
from lab3_dft.discrete_fourier_transform import discrete_fourier_transform

def interp_partition(x,part):
    for char in reversed(part):
        if char == "e":
            x = 2*x
        if char == "o":
            x = 2*x+1
    
    return x

def partition_cord(parition_shape,partition_str):
    # input_shape = np.array(image_shape)/(2**(len(partition_str)/2))

    partition_string_x = partition_str[::2]
    partition_string_y = partition_str[1::2]

    N = int(parition_shape[0]) # y direction
    M = int(parition_shape[1]) # x direction
    partition_cords = []
    for y in range(N):
        partition_cords.append([])
        for x in range(M):
            partition_cords[y].append([interp_partition(y,partition_string_y),interp_partition(x,partition_string_x)])
                                      
    return partition_cords

def weight(top,bottom): # W^{top}_{bottom}
    # Wn = e^{j2pi/N}

    real = np.cos(-2*np.pi*top/bottom)
    imag = np.sin(-2*np.pi*top/bottom)
    return np.array([real,imag])

def partition(image,partition_str):
    input_shape = np.array(image.shape)/(2**(len(partition_str)/2))

    partition_string_x = partition_str[::2]
    partition_string_y = partition_str[1::2]

    N = int(input_shape[0]) # y direction
    M = int(input_shape[1]) # x direction
    sub_partition = np.zeros((N,M))

    for y in range(N):
        for x in range(M):
            sub_partition[y,x] = image[interp_partition(y,partition_string_y),interp_partition(x,partition_string_x)]
    return sub_partition

def mult_complex(c1, c2):
    real_part = c1[0] * c2[0] - c1[1] * c2[1]
    imaginary_part = c1[0] * c2[1] + c1[1] * c2[0]
    return np.array([real_part, imaginary_part])

# only works for N x N
def fast_fourier_transform(image,partition_str = ""): # partition eeoo
    input_shape = np.array(image.shape)/(2**(len(partition_str)/2))
    
    N = int(input_shape[0]) # y direction
    M = int(input_shape[1]) # x direction


    if N == 2 and M == 2: # partition is now a 2x2
        # partition_cord_n = partition_cord((N,M),partition_str)
        # print("F" + partition_str + f" {M}x{N}")
        # print(partition_cord_n)

        image_seg = partition(image,partition_str)
        dft2x2 = discrete_fourier_transform(image_seg)
        dft2x2 = dft2x2.transpose(1,2,0)

        return dft2x2

    # would store both the real and imaginary part
    Fee = fast_fourier_transform(image,partition_str=partition_str+"ee") # each should output matrix the size of 2 x N/2 x N/2 (2 at the end is for the real and imaginary part)
    Feo = fast_fourier_transform(image,partition_str=partition_str+"eo")
    Foe = fast_fourier_transform(image,partition_str=partition_str+"oe")
    Foo = fast_fourier_transform(image,partition_str=partition_str+"oo")
 
    F = np.zeros((M,N,2))

    for u in range(M//2):
        for v in range(N//2):
            F[v,u]            = Fee[v,u] + mult_complex(Feo[v,u],weight(v,N)) + mult_complex(Foe[v,u],weight(u,N)) + mult_complex(Foo[v,u],weight(u+v,N)) # + +++
            F[v,u+N//2]       = Fee[v,u] + mult_complex(Feo[v,u],weight(v,N)) - mult_complex(Foe[v,u],weight(u,N)) - mult_complex(Foo[v,u],weight(u+v,N)) # + +--
            F[v+N//2,u]       = Fee[v,u] - mult_complex(Feo[v,u],weight(v,N)) + mult_complex(Foe[v,u],weight(u,N)) - mult_complex(Foo[v,u],weight(u+v,N)) # + -+-
            F[v+N//2,u+N//2]  = Fee[v,u] - mult_complex(Feo[v,u],weight(v,N)) - mult_complex(Foe[v,u],weight(u,N)) + mult_complex(Foo[v,u],weight(u+v,N)) # + --+

    # partition_cord_n = partition_cord((N,M),partition_str)
    # print("F" + partition_str + f" {M}x{N}")
    # print(partition_cord_n)
    # F = F/4

    return F
