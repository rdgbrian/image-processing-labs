import numpy as np

def convolve(image, kernel, mode='reflect'):
    """
    Convolve an image with a kernel.
    
    Parameters:
    - image: 2D numpy array (grayscale image)
    - kernel: 2D numpy array (kernel or filter)
    - mode: The mode used for padding ('reflect', 'constant', 'nearest', etc.)
    
    Returns:
    - Convolved image
    """
    # Get image and kernel dimensions
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Determine padding for the kernel
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Apply padding to the image based on the specified mode
    if mode == 'reflect':
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')
    elif mode == 'constant':
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    elif mode == 'nearest':
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='nearest')
    else:
        raise ValueError(f"Unsupported padding mode: {mode}")

    # Prepare an empty output array
    output = np.zeros_like(image)

    # Perform the convolution operation
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region of interest (ROI) from the padded image
            roi = padded_image[i:i + kernel_height, j:j + kernel_width]
            
            # Perform element-wise multiplication and sum the result
            output[i, j] = np.sum(roi * kernel)
    
    return output


def edge_detection_kirsch(image,mode="constant"):
    """
    Apply the Kirsch operator to detect edges in an image.
    """
    # Define the 8 Kirsch masks
    kirsch_masks = [

        np.array([[5, 5, 5], 
                  [-3, 0, -3], 
                  [-3, -3, -3]]),  # north

        np.array([[5, 5, -3], 
                  [5, 0, -3], 
                  [-3, -3, -3]]),  # Northwest

        np.array([[5, -3, -3], 
                  [5, 0, -3], 
                  [5, -3, -3]]),  # west

        np.array([[-3, -3, -3], 
                  [5, 0, -3],
                  [5, 5, -3]]),  # southwest

        np.array([[-3, -3, -3], 
                  [-3, 0, -3], 
                  [5, 5, 5]]),  # south

        np.array([[-3, -3, -3],
                   [-3, 0, 5], 
                   [-3, 5, 5]]),  # Southeast
                   
        np.array([[-3, -3, 5], 
                  [-3, 0, 5], 
                  [-3, -3, 5]]),  # east

        np.array([[-3, 5, 5], 
                  [-3, 0, 5], 
                  [-3, -3, -3]]),  # northeast 
        ]
    
    # Apply each mask and take the maximum response
    edge_magnitudes = [convolve(image, mask, mode=mode) for mask in kirsch_masks]
    kirsch_result = np.max(edge_magnitudes, axis=0)
    return kirsch_result


def edge_detection_sobel(image,mode="constant"):
    """
    Apply the Sobel operator to detect edges in an image.
    """
    # Define Sobel masks
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]])  # Horizontal edges
    sobel_y = np.array([[-1, -2, -1], 
                        [0, 0, 0], 
                        [1, 2, 1]])  # Vertical edges
    
    # Convolve the image with each Sobel mask
    grad_x = convolve(image, sobel_x, mode=mode)
    grad_y = convolve(image, sobel_y, mode=mode)
    
    # Compute the gradient magnitude
    sobel_result = np.sqrt(grad_x**2 + grad_y**2)
    return sobel_result


# Example usage
if __name__ == "__main__":
    # Example grayscale image
    image = np.array([[0, 0, 0, 0],
                      [0, 20, 20, 0],
                      [0, 20, 20, 0],
                      [0, 0, 0, 0]], dtype=np.float32)

    # Apply Kirsch operator
    kirsch_edges = edge_detection_kirsch(image)
    print("Kirsch edges:\n", kirsch_edges)

    # Apply Sobel operator
    sobel_edges = edge_detection_sobel(image)
    print("Sobel edges:\n", sobel_edges)
