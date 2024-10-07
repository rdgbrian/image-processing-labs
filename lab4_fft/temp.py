
import numpy as np # math li
import matplotlib.pyplot as plt
from lab3_dft import discrete_fourier_transform

def interp_partition(x,part):
    for char in reversed(part):
        if char == "e":
            x = 2*x
        if char == "o":
            x = 2*x+1
    
    return x


def fast_fourier_transform(image,partition): # partition eeoo
    base_case = True # 2 by 2
    if base_case:
        print()
        return 
    
    input_shape = image.shape # given the itteration we are on edit N
    N = input_shape[0] # y direction
    M = input_shape[1] # x direction

    # would store both the real and imaginary part
    Fee_real, Fee_imag = fast_fourier_transform(image,partition=partition+"ee") # each should output matrix the size of N/2 x N/2 x 2 (2 at the end is for the real and imaginary part)
    Feo_real, Feo_imag = fast_fourier_transform(image,partition=partition+"eo")
    Foe_real, Foe_imag = fast_fourier_transform(image,partition=partition+"oe")
    Foo_real, Foo_imag = fast_fourier_transform(image,partition=partition+"oo")
 
    F_real = np.zeros((N,M))
    F_imag = np.zeros((N,M))

    Wv = 


    for u in range(N/2):
        for v in range(M/2):
            F_real[u,v]          = +Fee_real[u,v]+Feo_real[u,v]+Foe_real[u,v]+Foo_real[u,v]
            F_real[u+N/2,v]      = +Fee_real[u,v]+Feo_real[u,v]-Foe_real[u,v]-Foo_real[u,v]
            F_real[u,v+M/2]      = +Fee_real[u,v]-Feo_real[u,v]+Foe_real[u,v]-Foo_real[u,v]
            F_real[u+N/2,v+N/2]  = +Fee_real[u,v]-Feo_real[u,v]-Foe_real[u,v]+Foo_real[u,v]

            
    return F




print(interp_partition(1,"eo"))