from osgeo import gdal
import os
import cv2
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hsfm.io
import hsfm.core
import hsfm.image
import hsfm.plot
import hsfm.utils

def rescale_images(image_directory, 
                   extension='.tif',
                   scale=8,
                   verbose=False):
    
    image_files  = sorted(glob.glob(os.path.join(image_directory,'*'+ extension)))
    
    for image_file in image_files:
        
        file_path, file_name, file_extension = hsfm.io.split_file(image_file)
        output_directory = hsfm.io.create_dir(file_path+'_sub'+str(scale))
        output_file = os.path.join(output_directory, 
                                   file_name +'_sub'+str(scale)+file_extension)
        
        hsfm.utils.rescale_geotif(image_file,
                                  output_file_name=output_file,
                                  scale=scale,
                                  verbose=verbose)

    return output_directory
#     return sorted(glob.glob(os.path.join(output_directory,'*'+ extension)))

def rescale_tsai_cameras(camera_directory,
                         extension='.tsai',
                         scale=8):
                         
    pitch = "pitch = 1"
    new_pitch = "pitch = "+str(scale)
    
    camera_files  = sorted(glob.glob(os.path.join(camera_directory,'*'+ extension)))
                 
    for camera_file in camera_files:
        
        file_path, file_name, file_extension = hsfm.io.split_file(camera_file)
        output_directory = hsfm.io.create_dir(file_path+'_sub'+str(scale))
        output_file = os.path.join(output_directory, 
                                   file_name +'_sub'+str(scale)+file_extension)
                                   
        
        hsfm.io.replace_string_in_file(camera_file, output_file, pitch, new_pitch)
        
#     return output_directory
#     return sorted(glob.glob(os.path.join(output_directory,'*'+ extension)))

def batch_generate_cameras(image_directory,
                           camera_positions_file_name,
                           reference_dem_file_name,
                           focal_length_mm,
                           output_directory='data/cameras',
                           print_asp_call=False,
                           verbose=False,
                           subset=None,
                           manual_heading_selection=False):
    # TODO
    # Embed hsfm.utils.pick_headings() within calculate_heading_from_metadata() and launch only for images where the heading could not be determined.  
    
    image_list = sorted(glob.glob(os.path.join(image_directory, '*.tif')))
    if manual_heading_selection == False:
        df = calculate_heading_from_metadata(camera_positions_file_name, subset=subset)
    else:
        df = hsfm.utils.pick_headings(image_directory, camera_positions_file_name, subset, delta=0.01)
    
    
    if len(image_list) != len(df):
        print('Mismatch between metadata entries in camera position file and available images.')
        sys.exit(1)
    
    for i,v in enumerate(image_list):
        image_file_name = v
        camera_lat_lon_center_coordinates = (df['Latitude'].iloc[i], df['Longitude'].iloc[i])
        heading = df['heading'].iloc[i]
        
        hsfm.asp.generate_camera(image_file_name,
                                 camera_lat_lon_center_coordinates,
                                 reference_dem_file_name,
                                 focal_length_mm,
                                 heading,
                                 print_asp_call=print_asp_call,
                                 verbose=verbose,
                                 output_directory=output_directory)
        
    
    
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

def download_images_to_disk(camera_positions_file_name, 
                            subset=None, 
                            output_directory='output_data/raw_images',
                            image_type='pid_tiff'):
                            
    hsfm.io.create_dir(output_directory)
    df = hsfm.core.select_images_for_download(camera_positions_file_name, subset)
    targets = dict(zip(df[image_type], df['fileName']))
    for pid, file_name in targets.items():
        print('Downloading',file_name, image_type)
        img = hsfm.core.download_image(pid)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        out = os.path.join(output_directory, file_name+'.tif')
        cv2.imwrite(out,img_gray)
        final_output = hsfm.utils.optimize_geotif(out)
        os.remove(out)
        os.rename(final_output, out)
    
    return output_directory
    
def preprocess_images(template_directory,
                      camera_positions_file_name=None,
                      image_directory=None,
                      image_type='pid_tiff', 
                      output_directory='output_data/images',
                      subset=None, 
                      scale=None,
                      qc=False):
                      
    """
    Function to preprocess images from NAGAP archive in batch.
    """
    # TODO
    # - Make io faster with gdal
    # - Generalize for other types of images
    # - Add affine transformation
                      
    hsfm.io.create_dir(output_directory)
    
    templates = hsfm.core.gather_templates(template_directory)         
                      
    intersections =[]
    file_names = []
    
    if camera_positions_file_name:
        df = hsfm.core.select_images_for_download(camera_positions_file_name, subset)
        targets = dict(zip(df[image_type], df['fileName']))
        for pid, file_name in targets.items():
            print('Processing',file_name)
            img = hsfm.core.download_image(pid)
            img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            intersection_angle = hsfm.core.preprocess_image(img_gray,
                                                            file_name,
                                                            templates, 
                                                            qc=qc,
                                                            output_directory=output_directory)
            intersections.append(intersection_angle)
            file_names.append(file_name)
    
    elif image_directory:
        image_files = sorted(glob.glob(os.path.join(image_directory,'*.tif')))
        for image_file in image_files:
            file_path, file_name, file_extension = hsfm.io.split_file(image_file)
            print('Processing',file_name)
            
            ## TODO
            ## - Make io faster with gdal
            # src = gdal.Open(image_file)
            # arr = src.ReadAsArray()
            # if len(arr.shape) == 3:
            #     img = arr.reshape(arr.shape[1], arr.shape[2], arr.shape[0]).shape
            #     img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #
            # else:
            #     img_gray = arr
                
            img = cv2.imread(image_file)
            img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            intersection_angle = hsfm.core.preprocess_image(img_gray, 
                                                            file_name,
                                                            templates, 
                                                            qc=qc,
                                                            output_directory=output_directory)
            intersections.append(intersection_angle)
            file_names.append(file_name)
        
    if qc == True:
        hsfm.plot.plot_intersection_angles_qc(intersections, file_names)
    
    return output_directory