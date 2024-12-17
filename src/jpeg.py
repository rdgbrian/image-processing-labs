from src.image_transforms import dct2d

# %%
import numpy as np
from PIL import Image
from src.jpeg_tables import DC_CODE, AC_CODE, category


Y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


def zigzag_scan(matrix):
    """Applies zigzag scan to an 8x8 matrix."""
    rows, cols = matrix.shape
    result = []
    for s in range(rows + cols - 1):
        if s % 2 == 0:
            x = min(s, rows - 1)
            y = s - x
            while x >= 0 and y < cols:
                result.append(matrix[x, y])
                x -= 1
                y += 1
        else:
            y = min(s, cols - 1)
            x = s - y
            while y >= 0 and x < rows:
                result.append(matrix[x, y])
                x += 1
                y -= 1
    return result

def blockify(image, block_size=8):
    """Breaks the image into non-overlapping 8x8 blocks"""
    height, width = image.shape
    blocks = []
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i+block_size, j:j+block_size]
            blocks.append(block)
    return np.array(blocks)

def dct2(block):
    """2D Discrete Cosine Transform"""
    # print(block.shape)
    return dct2d(block) #scipy.fftpack.dct(scipy.fftpack.dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    """Inverse 2D Discrete Cosine Transform"""
    # print(block.shape)
    return dct2d(block,inverse=True) #scipy.fftpack.idct(scipy.fftpack.idct(block.T, norm='ortho').T, norm='ortho')

def quantize(block, quantization_matrix):
    """Quantizes a block based on a quantization matrix"""
    return np.round(block / quantization_matrix)

def dequantize(block, quantization_matrix):
    """Dequantizes a block based on a quantization matrix"""
    return block * quantization_matrix

def dct_quantize_matrix(image, quantization_matrix=Y_quantization_matrix):
    """Encodes an image using a basic JPEG-like compression"""
    blocks = blockify(image)

    dct_blocks = np.array([dct2(block) for block in blocks])
    quantized_blocks = np.array([quantize(block, quantization_matrix) for block in dct_blocks])
    
    return quantized_blocks

def inv_dct_quantize_matrix(quantized_blocks, image_shape,quantization_matrix=Y_quantization_matrix):
    """Decodes the quantized blocks to reconstruct the image"""
    dequantized_blocks = np.array([dequantize(block, quantization_matrix) for block in quantized_blocks])
    idct_blocks = np.array([idct2(block) for block in dequantized_blocks])

    height, width = image_shape
    reconstructed_image = np.zeros((height, width))
    block_size = 8
    idx = 0
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            reconstructed_image[i:i+block_size, j:j+block_size] = idct_blocks[idx]
            idx += 1

    return reconstructed_image

def int_to_binary_ones_complement(n: int) -> str:
    """
    Converts an integer to its binary representation as a string.
    Uses one's complement for negative integers.
    
    Parameters:
    n (int): The integer to convert.
    
    Returns:
    str: Binary representation as a string.
    """
    if n >= 0:
        return bin(n)[2:]  # For positive integers, just use the binary representation.
    else:
        # Get the binary of the absolute value
        binary_positive = bin(abs(n))[2:]
        # Calculate one's complement by flipping bits
        ones_complement = ''.join('1' if bit == '0' else '0' for bit in binary_positive)
        return ones_complement


def block_to_code(zig_block, prev_dc=0):
    code = ""
    # do dc code first
    dc_coeff = zig_block[0]
    dc_coeff = dc_coeff - prev_dc

    cat = category(dc_coeff)
    base_code, length = DC_CODE[cat]

    temp_value_binary = int_to_binary_ones_complement(dc_coeff)

    code += base_code + temp_value_binary

    run = 0
    for ac_coeff in zig_block[1:]:
        if run > 15:
            break
        if ac_coeff == 0:
            run += 1
        else:
            cat = category(ac_coeff)
            base_code, length = AC_CODE[(run, cat)]
            temp_value_binary = int_to_binary_ones_complement(ac_coeff)
            code = code + base_code + temp_value_binary
            run = 0
    
    code += "1010" # end of block
    return code

def blocks_to_code(zig_blocks):
    code = ""
    prev_dc = 0
    for zig_block in zig_blocks:
        code = code + block_to_code(zig_block,prev_dc=prev_dc)
        prev_dc = zig_block[0]  
    
    return code 

def jpeg_encode(image):
    blocks = dct_quantize_matrix(image)
    blocks = blocks.astype(int)
    zig_blocks = [zigzag_scan(block) for block in blocks]
    code = blocks_to_code(zig_blocks)
    return code
