import os
import cv2

import hsfm.io
import hsfm.core
import hsfm.image


def preprocess_images(csv_file_name, 
                      template_directory,
                      image_type='pid_tiff', 
                      out_dir='data/images',
                      subset=None, 
                      scale=None):
                      
    """
    Function to preprocess images from NAGAP archive in batch.
    """
    # TODO
    # - Add qc output functions to evaluate how well the principle
    #   point was detected and image cropped accordingly.
                      
    hsfm.io.create_dir(out_dir)             
                      
    left_template = os.path.join(template_directory,'L.jpg')
    top_template = os.path.join(template_directory,'T.jpg')
    right_template = os.path.join(template_directory,'R.jpg')
    bottom_template = os.path.join(template_directory,'B.jpg')
    
    window_left = [5000,7000,250,1500]
    window_right = [5000,6500,12000,12391]
    window_top = [0,500,6000,7200]
    window_bottom = [11000,11509,6000,7200]
    
    df = hsfm.core.select_images_for_download(csv_file_name, subset)
    targets = dict(zip(df[image_type], df['fileName']))
    
    for pid, file_name in targets.items():
        img = hsfm.core.download_image(pid)
        
        side = hsfm.core.evaluate_image_frame(img)
        
        img_gray = hsfm.core.convert_image_to_grayscale(img)
        img_gray_clahe = hsfm.image.clahe_equalize_image(img_gray)
        
        left_slice, top_slice, right_slice, bottom_slice = hsfm.core.slice_image_frame(img_gray_clahe)
        
        left_slice_padded = hsfm.core.pad_image(left_slice)
        top_slice_padded = hsfm.core.pad_image(top_slice)
        right_slice_padded = hsfm.core.pad_image(right_slice)
        bottom_slice_padded = hsfm.core.pad_image(bottom_slice)
    
        left_fiducial = hsfm.core.get_fiducials(left_slice_padded, 
                                                left_template, 
                                                window_left, 
                                                position = 'left')
        top_fiducial = hsfm.core.get_fiducials(top_slice_padded, 
                                               top_template, 
                                               window_top, 
                                               position = 'top')
        right_fiducial = hsfm.core.get_fiducials(right_slice_padded, 
                                                 right_template, 
                                                 window_right, 
                                                 position = 'right')
        bottom_fiducial = hsfm.core.get_fiducials(bottom_slice_padded, 
                                                  bottom_template, 
                                                  window_bottom, 
                                                  position = 'bottom')
                                                  
        principal_point = hsfm.core.principal_point(left_fiducial,
                                                    top_fiducial,
                                                    right_fiducial,
                                                    bottom_fiducial)
                                                    
                                                    
        cropped = hsfm.core.crop_about_principal_point(img, principal_point)
        img_rot = hsfm.core.rotate_camera(cropped, side=side)
        
        out = os.path.join(out_dir, file_name+'.tif')
        
        cv2.imwrite(out,img_rot)
    
    return out_dir