import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import math

def crop_power2(image_path, save_path=None):
    """
    Crop an image to the nearest square dimensions where each side is 2^n.

    Args:
        image_path (str): Path to the input image.
        save_path (str, optional): Path to save the cropped image. If None, it won't save.

    Returns:
        PIL.Image.Image: The cropped image.
    """
    # Open the image
    img = Image.open(image_path)
    width, height = img.size

    # Find the nearest power of 2 for both dimensions
    min_dim = min(width, height)
    nearest_power = 2**int(math.log2(min_dim))

    # Compute the cropping box
    left = (width - nearest_power) // 2
    top = (height - nearest_power) // 2
    right = left + nearest_power
    bottom = top + nearest_power

    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))

    # Save the cropped image if save_path is provided
    if save_path:
        cropped_img.save(save_path)

    cropped_img = np.array(cropped_img)
    return cropped_img

def reduce_image(img,num_red = 1):

    for i in range(num_red):
        new_shape = np.array(img.shape)//2
        new_img = np.zeros(new_shape)
        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                sum_of_four = img[2*i,2*j] + img[2*i+1,2*j] + img[2*i,2*j+1] + img[2*i+1,2*j+1]
                new_img[i][j] = sum_of_four/4
        
        img = new_img
    return img

