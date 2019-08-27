import PIL
import os
import cv2
import numpy as np
from skimage import exposure

import hsfm.io

# Adjust max pixel count to avoid decompression bomb warning
# PIL.Image.warnings.simplefilter("ignore", PIL.Image.DecompressionBombWarning)
PIL.Image.MAX_IMAGE_PIXELS = 1000000000

def rescale_image(image_file_name, scale_factor):
    img = PIL.Image.open(image_file_name)

    width = int(img.width / scale_factor)
    height = int(img.height / scale_factor)

    rescaled_img = img.resize((width, height), resample=PIL.Image.BICUBIC)
    return rescaled_img
    
    
def clahe_equalize_image(img_gray):
    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_gray_clahe = clahe.apply(img_gray)
    return img_gray_clahe
    
def img_linear_stretch(img_gray):
    p_min, p_max = np.percentile(img_gray, (0.1, 99.9))
    img_rescale = exposure.rescale_intensity(img_gray, in_range=(p_min, p_max))
    return img_rescale