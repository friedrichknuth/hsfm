import os
import cv2
import sys
import glob

import hsfm.io
import hsfm.core
import hsfm.image

def rescale_images(image_directory, 
                   extension='.tif',
                   scale_factor=8):
    
    image_files  = sorted(glob.glob(os.path.join(image_directory,'*'+ extension)))
    
    for image_file in image_files:
        
        file_path, file_name, file_extension = hsfm.io.split_file(image_file)
        output_directory = hsfm.io.create_dir(file_path+'_sub'+str(scale_factor))
        output_file = os.path.join(output_directory, 
                                   file_name +'_sub'+str(scale_factor)+file_extension)
        
        rescaled_img = hsfm.image.rescale_image(image_file, scale_factor)
        
        rescaled_img.save(output_file)
    
    print("Rescaled images available at ", output_directory)
    
    return sorted(glob.glob(os.path.join(output_directory,'*'+ extension)))

def rescale_tsai_cameras(camera_directory,
                         extension='.tsai',
                         scale_factor=8):
                         
    pitch = "pitch = 1"
    new_pitch = "pitch = "+str(scale_factor)
    
    camera_files  = sorted(glob.glob(os.path.join(camera_directory,'*'+ extension)))
                 
    for camera_file in camera_files:
        
        file_path, file_name, file_extension = hsfm.io.split_file(camera_file)
        output_directory = hsfm.io.create_dir(file_path+'_sub'+str(scale_factor))
        output_file = os.path.join(output_directory, 
                                   file_name +'_sub'+str(scale_factor)+file_extension)
                                   
        
        hsfm.io.replace_string_in_file(camera_file, output_file, pitch, new_pitch)
        
    print("Rescaled cameras available at ", output_directory)
    
    return sorted(glob.glob(os.path.join(output_directory,'*'+ extension)))

def batch_generate_cameras(image_directory,
                           camera_positions_file_name,
                           reference_dem_file_name,
                           focal_length_mm,
                           output_directory='./data/cameras',
                           subset=None):
                           
                           
    
    image_list = sorted(glob.glob(os.path.join(image_directory, '*.tif')))
    df = calculate_heading_from_metadata(camera_positions_file_name, subset=subset)
    
    
    if len(image_list) != len(df):
        print('Mismatch between metadata entries in camera position file and available images.')
        sys.exit(1)
    
    for i,v in enumerate(image_list):
        image_file_name = v
        camera_lat_lon_center_coordinates = (df['Latitude'].iloc[i], df['Longitude'].iloc[i])
        heading = df['heading'].iloc[i]
        
        print(camera_lat_lon_center_coordinates)
        
        hsfm.asp.generate_camera(image_file_name,
                                 camera_lat_lon_center_coordinates,
                                 reference_dem_file_name,
                                 focal_length_mm,
                                 heading,
                                 output_directory=output_directory)
        
    pass
    
    
def calculate_heading_from_metadata(camera_positions_file_name, subset=None):
    df = hsfm.core.select_images_for_download(camera_positions_file_name, subset)
    lons = df['Longitude'].values
    lats = df['Latitude'].values
    
    headings = []
    for i, v in enumerate(lats):
        try:
            p0_lon = lons[i]
            p0_lat = lats[i]

            p1_lon = lons[i+1]
            p1_lat = lats[i+1]
        
            heading = hsfm.geospatial.calculate_heading(p0_lon,p0_lat,p1_lon,p1_lat)
            headings.append(heading)
    
        except:
            # When the loop reaches the last element, 
            # assume that the final image is oriented 
            # the same as the previous, i.e. the flight 
            # direction did not change
            headings.append(heading)
        
    df['heading'] = headings
    
    return df
    
    
def preprocess_images(camera_positions_file_name, 
                      template_directory,
                      image_type='pid_tiff', 
                      output_directory='data/images',
                      subset=None, 
                      scale=None):
                      
    """
    Function to preprocess images from NAGAP archive in batch.
    """
    # TODO
    # - Add qc output functions to evaluate how well the principle
    #   point was detected and image cropped accordingly.
                      
    hsfm.io.create_dir(output_directory)             
                      
    left_template = os.path.join(template_directory,'L.jpg')
    top_template = os.path.join(template_directory,'T.jpg')
    right_template = os.path.join(template_directory,'R.jpg')
    bottom_template = os.path.join(template_directory,'B.jpg')
    
    window_left = [5000,7000,250,1500]
    window_right = [5000,6500,12000,12391]
    window_top = [0,500,6000,7200]
    window_bottom = [11000,11509,6000,7200]
    
    df = hsfm.core.select_images_for_download(camera_positions_file_name, subset)
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
        
        out = os.path.join(output_directory, file_name+'.tif')
        
        cv2.imwrite(out,img_rot)
    
    return output_directory