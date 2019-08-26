import PIL
import os
import cv2
import numpy as np
from skimage import exposure

import hsfm.io

def rescale_image(image_file_name, scale_factor):
    img = PIL.Image.open(img_fn)

    width = int(img.width / scale_factor)
    height = int(img.height / scale_factor)

    rescaled_img = img.resize((width, height), resample=Image.BICUBIC)
    
    file_path, file_name, file_extension = hsfm.io.split_file(image_file_name)
    out_fn = os.path.join(file_path, file_name +'_sub'+str(scale_factor)+file_extension)
    rescaled_img.save(out_fn)
    
    
def clahe_equalize_image(img_gray):
    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_gray_clahe = clahe.apply(img_gray)
    return img_gray_clahe
    
def img_linear_stretch(img_gray):
    p_min, p_max = np.percentile(img_gray, (0.1, 99.9))
    img_rescale = exposure.rescale_intensity(img_gray, in_range=(p_min, p_max))
    return img_rescale