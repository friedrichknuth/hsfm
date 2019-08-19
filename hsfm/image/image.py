import PIL
import os

import hsfm.io

def rescale_image(image_file_name, scale_factor):
    img = PIL.Image.open(img_fn)

    width = int(img.width / scale_factor)
    height = int(img.height / scale_factor)

    rescaled_img = img.resize((width, height), resample=Image.BICUBIC)
    
    file_path, file_name, file_extension = hsfm.io.split_file(image_file_name)
    out_fn = os.path.join(file_path, file_name +'_sub'+str(scale_factor)+file_extension)
    rescaled_img.save(out_fn)