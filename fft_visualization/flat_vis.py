#%%

import numpy as np
#%%
def order_flat_2d_fft(n: int):
    """
    gives the special ordering for flatting the foureirs for a 2^n x 2^n image

    n (int): n = log(N) where the image is of size NxN
    quardrant: which subquardrant is the current subset
    """

    if n == 1: # when 2x2 sub fourier
        return np.array([[0,1],[2,3]])

    top_left = order_flat_2d_fft(n-1)

    top_right = order_flat_2d_fft(n-1) + 1 * top_left.size
    bottom_left = order_flat_2d_fft(n-1) + 2 * top_left.size
    bottom_right = order_flat_2d_fft(n-1) + 3 * top_left.size


    curr_order = np.block([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])


    return curr_order # list of ints 
import numpy as np

def order_flat_nd_fft(n: int, d: int) -> np.ndarray:
    """
    Generates the flattened ordering for an n-level FFT in d dimensions.
    
    Parameters:
    - n: int, number of FFT levels (i.e., size is 2^n in each dimension)
    - d: int, number of dimensions (e.g., 2 for 2D, 3 for 3D)

    Returns:
    - ndarray of shape [2^n, 2^n, ..., 2^n] with unique indices from 0 to (2^(n*d)) - 1
    """

    if n == 1:
        # Base case: all 2^d corner indices
        shape = [2] * d
        grid = np.indices(shape).reshape(d, -1).T  # shape: (2^d, d)
        
        # Flatten ordering: 0, 1, ..., 2^d - 1
        order = np.arange(2**d)
        
        # Reshape into d-dimensional array
        out = np.empty(shape, dtype=int)
        for idx, val in zip(grid, order):
            out[tuple(idx)] = val
        return out

    # Recursive case
    sub_order = order_flat_nd_fft(n - 1, d)
    sub_shape = sub_order.shape
    sub_size = sub_order.size

    # Create the output array by combining 2^d blocks
    blocks = []
    for i in range(2**d):
        offset = i * sub_size
        block = sub_order + offset
        blocks.append(block)

    # Now reshape and stack all 2^d blocks
    # Each axis gets doubled in size
    new_shape = [s * 2 for s in sub_shape]
    out = np.empty(new_shape, dtype=int)

    # Assign blocks to each corner
    for i, block in enumerate(blocks):
        # Compute binary coordinates for this block
        coords = [(i >> k) & 1 for k in reversed(range(d))]
        # Use slicing to insert the block
        slices = tuple(slice(c * s, (c + 1) * s) for c, s in zip(coords, sub_shape))
        out[slices] = block

    return out



# #%%
# # temp_order = order_flat_2d_fft(2)
# temp_order = order_flat_nd_fft(n=0,d=1)
# print(temp_order)
# print(temp_order.shape)
# # print(temp_order)
# # %%
