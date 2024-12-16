# %%
import matplotlib.pyplot as plt
import numpy as np
from src.discrete_fourier_transform import dft2d

from src.utils import fourier_spectrum, create_white_square, rgb2gray, center_fourier

# %%
# Example: 10x10 image with a 4x4 white square in the center
n = 32
square_size = 8
image = create_white_square(n, square_size)

# image = plt.imread("Lab1-Inverting_Image/crying-cat-meme.jpg")
print(image.shape) # show the dimensions of the image
plt.imshow(image)
plt.show()

# %%
dft_og = dft2d(image,centered=True)


dft_real_c = dft_og.real
dft_imaginary_c= dft_og.imag
spectrum = fourier_spectrum(dft_real_c, dft_imaginary_c)

dft_og_centered = center_fourier(dft_og)
spectrum_centered = fourier_spectrum(dft_og_centered.real,dft_og_centered.imag)

plt.imshow(spectrum_centered)
plt.show()
plt.imshow(spectrum)
plt.show()

bruh = center_fourier(dft_og_centered)

# %%
bruh_spectrum = fourier_spectrum(bruh.real,bruh.imag)
plt.imshow(bruh_spectrum)
plt.show()
# %%

og_temp = dft2d(dft_og,inverse=True)
plt.imshow(og_temp.real)
plt.show()

# %%

f = np.fft.fft2(image)
f_centered = np.fft.fftshift(f)
np_real = f.real
np_imag = f.imag

dft_og = dft2d(image,centered=True)
og_real = dft_og.real
og_imag = dft_og.imag



are_equal = np.allclose(np_real,og_real, rtol=1e-6, atol=1e-9)
print("Are the arrays equal to the given precision?", are_equal)
are_equal = np.allclose(np_imag,og_imag, rtol=1e-6, atol=1e-9)
print("Are the arrays equal to the given precision?", are_equal)



