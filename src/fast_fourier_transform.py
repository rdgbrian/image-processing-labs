
import numpy as np # math li
import matplotlib.pyplot as plt
from src.discrete_fourier_transform import discrete_fourier_transform

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


def interp_partition(x,part):
    for char in reversed(part):
        if char == "e":
            x = 2*x
        if char == "o":
            x = 2*x+1
    
    return x

def partition_cord(parition_shape,partition_str): # give all the cordinated from the original image from the parition

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

def weight(top,bottom, inverse=False): # W^{top}_{bottom}
    # Wn = e^{j2pi/N}

    if inverse != False:
        sign = -1
    else:
        sign = 1

    real = np.cos(sign*2*np.pi*top/bottom)
    imag = np.sin(sign*2*np.pi*top/bottom)
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
def fast_fourier_transform(image,partition_str = "",inverse=False): # partition eeoo
    input_shape = np.array(image.shape)/(2**(len(partition_str)/2))
    
    N = int(input_shape[0]) # y direction
    M = int(input_shape[1]) # x direction


    if N == 2 and M == 2: # partition is now a 2x2
        # partition_cord_n = partition_cord((N,M),partition_str)
        # print("F" + partition_str + f" {M}x{N}")
        # print(partition_cord_n)

        image_seg = partition(image,partition_str)
        dft2x2 = discrete_fourier_transform(image_seg,inverse=inverse)
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
            F[v,u]            = Fee[v,u] + mult_complex(Feo[v,u],weight(v,N,inverse)) + mult_complex(Foe[v,u],weight(u,N,inverse)) + mult_complex(Foo[v,u],weight(u+v,N,inverse)) # + +++
            F[v,u+N//2]       = Fee[v,u] + mult_complex(Feo[v,u],weight(v,N,inverse)) - mult_complex(Foe[v,u],weight(u,N,inverse)) - mult_complex(Foo[v,u],weight(u+v,N,inverse)) # + +--
            F[v+N//2,u]       = Fee[v,u] - mult_complex(Feo[v,u],weight(v,N,inverse)) + mult_complex(Foe[v,u],weight(u,N,inverse)) - mult_complex(Foo[v,u],weight(u+v,N,inverse)) # + -+-
            F[v+N//2,u+N//2]  = Fee[v,u] - mult_complex(Feo[v,u],weight(v,N,inverse)) - mult_complex(Foe[v,u],weight(u,N,inverse)) + mult_complex(Foo[v,u],weight(u+v,N,inverse)) # + --+

    # partition_cord_n = partition_cord((N,M),partition_str)
    # print("F" + partition_str + f" {M}x{N}")
    # print(partition_cord_n)
    # F = F/4

    return F


temp_parition = {"relative_cord": (0,0), "parition_string": "ee", "dependency": [], "weight": []}

class fft_partition:
    def __init__(self,value,relative_cord,partition_string = "",dependents=None,weight_params_top = None):
        self.value = value
        self.relative_cord = relative_cord
        self.partition_string = partition_string

        self.weight_params_bottom = 2**(len(self.partition_string)/2)
    
        self.weight_params_top = [0,relative_cord[1],relative_cord[0],relative_cord[0] + relative_cord[1]]

        self.dependants = dependents
        self.image_cord = None
        if dependents is None:
            self.image_cord = partition_cord((2,2),self.partition_string)




class fft_info:
    def __init__(self):
        self.partitions = {}

    def __setitem__(self, key, value):
        self.partitions[key] = value

    def __getitem__(self, key):
        return self.partitions[key]

    def combine(self, other):
        """
        Combine the partitions of two fft_info objects.
        If a key exists in both, the value from `other` will overwrite the one in `self`.
        """
        combined = fft_info()  # Create a new instance for combined data
        combined.partitions = self.partitions.copy()  # Copy current partitions
        combined.partitions.update(other.partitions)  # Merge with the other partitions
        return combined
    
    def __iter__(self):
        return iter(self.partitions)

    def __repr__(self):
        return f"fft_info(partitions={self.partitions})"
    

def fast_fourier_transform(image,partition_str = "",save_info = False): # partition eeoo

    input_shape = np.array(image.shape)/(2**(len(partition_str)/2))
    temp = 2**(len(partition_str)/2)
    
    N = int(input_shape[0]) # y direction
    M = int(input_shape[1]) # x direction


    if N == 2 and M == 2: # partition is now a 2x2

        image_seg = partition(image,partition_str)
        dft2x2 = discrete_fourier_transform(image_seg)
        dft2x2 = dft2x2.transpose(1,2,0)
        
        temp_info = None
        if save_info:
            temp_info = fft_info()
            all_parts = [[0 for _ in range(M)] for _ in range(N)]
            for v in range(N):
                for u in range(M):
                    fft_part = fft_partition(dft2x2[v,u],(v,u),partition_string=partition_str)
                    all_parts[v][u] = fft_part
                    temp_info[partition_str] = all_parts
            # return dft2x2, temp_info
        return dft2x2, temp_info

    # would store both the real and imaginary part
    Fee, save_info_ee = fast_fourier_transform(image,partition_str=partition_str+"ee",save_info=save_info) # each should output matrix the size of 2 x N/2 x N/2 (2 at the end is for the real and imaginary part)
    Feo, save_info_eo = fast_fourier_transform(image,partition_str=partition_str+"eo",save_info=save_info)
    Foe, save_info_oe = fast_fourier_transform(image,partition_str=partition_str+"oe",save_info=save_info)
    Foo, save_info_oo = fast_fourier_transform(image,partition_str=partition_str+"oo",save_info=save_info)
    
    comb_save_info = None
    if save_info:
        comb_save_info = save_info_ee.combine(save_info_eo)
        comb_save_info = comb_save_info.combine(save_info_oe)
        comb_save_info = comb_save_info.combine(save_info_oo)
        all_parts = [[0 for _ in range(M)] for _ in range(N)]


    F = np.zeros((M,N,2))
    for v in range(N//2):
        for u in range(M//2):
            F[v,u]            = Fee[v,u] + mult_complex(Feo[v,u],weight(v,N)) + mult_complex(Foe[v,u],weight(u,N)) + mult_complex(Foo[v,u],weight(u+v,N)) # + +++
            F[v,u+N//2]       = Fee[v,u] + mult_complex(Feo[v,u],weight(v,N)) - mult_complex(Foe[v,u],weight(u,N)) - mult_complex(Foo[v,u],weight(u+v,N)) # + +--
            F[v+N//2,u]       = Fee[v,u] - mult_complex(Feo[v,u],weight(v,N)) + mult_complex(Foe[v,u],weight(u,N)) - mult_complex(Foo[v,u],weight(u+v,N)) # + -+-
            F[v+N//2,u+N//2]  = Fee[v,u] - mult_complex(Feo[v,u],weight(v,N)) - mult_complex(Foe[v,u],weight(u,N)) + mult_complex(Foo[v,u],weight(u+v,N)) # + --+
            
            if save_info:
                dependents = [comb_save_info[partition_str+"ee"][v][u], 
                              comb_save_info[partition_str+"eo"][v][u],
                              comb_save_info[partition_str+"oe"][v][u],
                              comb_save_info[partition_str+"oo"][v][u] ]
                all_parts[v][u] = fft_partition(F[v,u],(v,u),partition_string=partition_str,dependents=dependents)
                all_parts[v][u+N//2] = fft_partition(F[v,u+N//2] ,(v,u+N//2),partition_string=partition_str,dependents=dependents)
                all_parts[v+N//2][u] = fft_partition(F[v+N//2,u] ,(v+N//2,u),partition_string=partition_str,dependents=dependents)
                all_parts[v+N//2][u+N//2] = fft_partition(F[v+N//2,u+N//2],(v+N//2,u+N//2),partition_string=partition_str,dependents=dependents)

                comb_save_info[partition_str] = all_parts

    return F, comb_save_info 
