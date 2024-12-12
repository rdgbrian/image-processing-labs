import matplotlib.pylab as plt
import numpy as np
from invert import rgb2gray

image = plt.imread("images/crying-cat-meme.jpg")
print(image.shape) # show the dimensions of the image
image_gray = rgb2gray(image) 
print(image.shape) # show new dimensions of the image


np.histogram(image_gray)