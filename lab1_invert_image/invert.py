import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def rgb2gray(rgb): # Y' = 0.2989 R + 0.5870 G + 0.1140 B 
    """
    Function to turn an RGB image to gray scale
    rdg : an RGB image represented as a numpy array
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

image = plt.imread("Lab1-Inverting_Image/crying-cat-meme.jpg")
print(image.shape) # show the dimensions of the image
image_gray = rgb2gray(image) 
print(image.shape) # show new dimensions of the image

# show the gray scale image
plt.imshow(image_gray,cmap='gray')
plt.colorbar()
plt.savefig('grayscale_image.png')
plt.show()

# invert the image
image_gray_invert = 255 - image_gray
plt.imshow(image_gray_invert,cmap='gray')
plt.colorbar()
plt.savefig('invert_grayscale_image.png')
plt.show()

# original image
plt.imshow(image)
plt.colorbar()
plt.savefig('image.png')
plt.show()

# invert color image
invert_image = 255 - image
plt.imshow(image_gray_invert)
plt.colorbar()
plt.savefig('invert_image.png')
plt.show()