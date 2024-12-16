# %%
import numpy as np # math li
import matplotlib.pyplot as plt

from src.resize import crop_power2
from src.utils import rgb2gray
from src.resize import reduce_image
# from src.resize import rgb2gray

# %%
cropped_image = crop_power2("images\crying-cat-meme.jpg", save_path="output_image.jpg")
cropped_image = np.array(cropped_image)
cropped_image = rgb2gray(cropped_image)

plt.imshow(cropped_image,cmap="gray")
plt.show()
cropped_image = reduce_image(cropped_image,4)

plt.imshow(cropped_image,cmap="gray")
plt.show()
print(cropped_image.shape)

# %%
cropped_image_inv = 255 - cropped_image

plt.imshow(cropped_image_inv,cmap="gray")
plt.show()



# %%
