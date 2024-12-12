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

def reduce_image(img):
    new_shape = np.array(img.shape)//2
    new_img = np.zeros(new_shape)
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            sum_of_four = img[2*i,2*j] + img[2*i+1,2*j] + img[2*i,2*j+1] + img[2*i+1,2*j+1]
            new_img[i][j] = sum_of_four/4
    return new_img
    