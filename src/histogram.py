# %%
import numpy as np
import matplotlib.pyplot as plt
from src.utils import fourier_spectrum, create_white_square, rgb2gray, center_fourier, save_transforms
from src.resize import crop_power2, reduce_image

def histogram_equalization(image, num_bins=5):
    flat_arr = image.flatten()
    flat_arr = flat_arr/255
    num_bins = 100

    hist, bins = np.histogram(flat_arr, bins=num_bins, range=[0, 1], density=False)
    # bin_values = np.array(list(range(num_bins+1)))/num_bins
    hist_freq = hist/flat_arr.size
    cdf = np.cumsum(hist_freq)
    new_freq = np.zeros_like(hist_freq)

    closest_values = bins[np.abs(cdf[:, None] - bins).argmin(axis=1)]

    new_arr = flat_arr.copy()

    for i, bin_value in enumerate(bins[1:]):
        new_arr[(bins[i] < new_arr) & (new_arr < bin_value)] = closest_values[i]
    new_arr1 = new_arr
    new_arr = (new_arr*255).astype(np.int16)
    new_arr = new_arr.reshape(image.shape)
    return new_arr

def histogram_dynamic_range_modification(arr, new_min=0, new_max=255):

    return (new_max-new_min) * (arr - arr.min())/(arr.max()-arr.min()) + new_min
# %%


image_dir = "images\\"
image_names = ["crying-cat-meme.jpg", "juliana.jpg"]
output_dir = "outputs\\lab6"

image_path = image_dir + image_names[1]
cropped_image = crop_power2(image_path)
cropped_image = rgb2gray(cropped_image)
