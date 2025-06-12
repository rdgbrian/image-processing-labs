
import numpy as np # math li
import matplotlib.pyplot as plt
from src.discrete_fourier_transform import dft2d

from src.utils import mult_complex


def interp_partition(x,part): # x here is either 0 or 1
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
    for x in range(N):
        partition_cords.append([])
        for y in range(M):
            partition_cords[x].append([interp_partition(x,partition_string_x)],interp_partition(y,partition_string_y))
                                      
    return partition_cords

def weight(top, bottom, inverse=False):  # W^{top}_{bottom}
    """
    Compute the FFT twiddle factor W^top_bottom as a single complex number.
    
    :param top: The numerator of the exponent.
    :param bottom: The denominator of the exponent.
    :param inverse: If True, use the inverse FFT (negative sign in the exponent).
    :return: A single complex number representing the twiddle factor.
    """
    sign = 1 if inverse else -1
    return np.exp(sign * 2j * np.pi * top / bottom)


def partition(image,partition_str):
    input_shape = np.array(image.shape)/(2**(len(partition_str)/2))

    partition_string_x = partition_str[::2]
    partition_string_y = partition_str[1::2]

    N = int(input_shape[0]) # y direction
    M = int(input_shape[1]) # x direction
    sub_partition = np.zeros((N,M),dtype=np.complex128)

    for y in range(N):
        for x in range(M):
            sub_partition[x,y] = image[interp_partition(x,partition_string_x),interp_partition(y,partition_string_y)]
    return sub_partition

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
    

def fft2d(image,partition_str = "",inverse = False, save_info = False, centered = False): # partition eeoo

    input_shape = np.array(image.shape)/(2**(len(partition_str)/2))
    
    N = int(input_shape[0]) # y direction
    M = int(input_shape[1]) # x direction


    if N == 2 and M == 2: # partition is now a 2x2

        image_seg = partition(image,partition_str)
        dft2x2 = dft2d(image_seg,inverse=inverse,centered=centered)
        # dft2x2 = dft2x2.transpose(1,2,0)
        
        temp_info = None
        if save_info:
            temp_info = fft_info()
            all_parts = [[0 for _ in range(M)] for _ in range(N)]
            for v in range(N):
                for u in range(M):
                    fft_part = fft_partition(dft2x2[u,v],(u,v),partition_string=partition_str)
                    all_parts[u][v] = fft_part
                    temp_info[partition_str] = all_parts
            # return dft2x2, temp_info
        return dft2x2, temp_info

    # would store both the real and imaginary part
    Fee, save_info_ee = fft2d(image,partition_str=partition_str+"ee",inverse=inverse,save_info=save_info,centered=centered) # each should output matrix the size of 2 x N/2 x N/2 (2 at the end is for the real and imaginary part)
    Feo, save_info_eo = fft2d(image,partition_str=partition_str+"eo",inverse=inverse,save_info=save_info,centered=centered)
    Foe, save_info_oe = fft2d(image,partition_str=partition_str+"oe",inverse=inverse,save_info=save_info,centered=centered)
    Foo, save_info_oo = fft2d(image,partition_str=partition_str+"oo",inverse=inverse,save_info=save_info,centered=centered)
    
    comb_save_info = None
    if save_info:
        comb_save_info = save_info_ee.combine(save_info_eo)
        comb_save_info = comb_save_info.combine(save_info_oe)
        comb_save_info = comb_save_info.combine(save_info_oo)
        all_parts = [[0 for _ in range(M)] for _ in range(N)]


    F = np.zeros((M,N),dtype=np.complex128)
    for v in range(N//2):
        for u in range(M//2):

            F[u,v]            = Fee[u,v] + (Feo[u,v]*weight(v,N,inverse)) + (Foe[u,v]*weight(u,N,inverse)) + (Foo[u,v]*weight(u+v,N,inverse)) # ++++
            F[u+N//2,v]       = Fee[u,v] + (Feo[u,v]*weight(v,N,inverse)) + (Foe[u,v]*weight(u+N//2,N,inverse)) + (Foo[u,v]*weight((u+N//2)+v,N,inverse)) # + +--
            F[u,v+N//2]       = Fee[u,v] + (Feo[u,v]*weight(v+N//2,N,inverse)) + (Foe[u,v]*weight(u,N,inverse)) + (Foo[u,v]*weight(u+(v+N//2),N,inverse)) # + -+-
            F[u+N//2,v+N//2]  = Fee[u,v] + (Feo[u,v]*weight(v+N//2,N,inverse)) + (Foe[u,v]*weight(u+N//2,N,inverse)) + (Foo[u,v]*weight((u+N//2)+(v+N//2),N,inverse)) # + --+

            if save_info: # for future visulization
                dependents = [comb_save_info[partition_str+"ee"][u][v], 
                              comb_save_info[partition_str+"eo"][u][v],
                              comb_save_info[partition_str+"oe"][u][v],
                              comb_save_info[partition_str+"oo"][u][v] ]
                all_parts[u][v] = fft_partition(F[u,v],(v,u),partition_string=partition_str,dependents=dependents)
                all_parts[u+N//2][v] = fft_partition(F[u+N//2,v],(u+N//2,v),partition_string=partition_str,dependents=dependents)
                all_parts[u][v+N//2] = fft_partition(F[u,v+N//2] ,(u,v+N//2),partition_string=partition_str,dependents=dependents)
                all_parts[u+N//2][v+N//2] = fft_partition(F[u+N//2,v+N//2],(u+N//2,v+N//2),partition_string=partition_str,dependents=dependents)

                comb_save_info[partition_str] = all_parts

    if inverse:
        F = F / 4

    return F, comb_save_info 
