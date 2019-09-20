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
                   scale_factor=8):
    
    image_files  = sorted(glob.glob(os.path.join(image_directory,'*'+ extension)))
    
    for image_file in image_files:
        
        file_path, file_name, file_extension = hsfm.io.split_file(image_file)
        output_directory = hsfm.io.create_dir(file_path+'_sub'+str(scale_factor))
        output_file = os.path.join(output_directory, 
                                   file_name +'_sub'+str(scale_factor)+file_extension)
        
        rescaled_img = hsfm.image.rescale_image(image_file, scale_factor)
        
        rescaled_img.save(output_file)
    
#     return output_directory
#     return sorted(glob.glob(os.path.join(output_directory,'*'+ extension)))

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
    

def detect_fiducials_and_principal_point(windows, 
                                         templates, 
                                         grayscale_unit8_image_array,
                                         noisify=None):
    img_gray = grayscale_unit8_image_array

    window_left   = windows[0]
    window_top    = windows[1]
    window_right  = windows[2]
    window_bottom = windows[3]

    # enhance local contrast
    img_gray_clahe = hsfm.image.clahe_equalize_image(img_gray)

    # pull out slices according to window
    slices = hsfm.core.slice_image_frame(img_gray_clahe,windows)
    
    # pad each slice so that the template can be fully moved over a given fiducial marker
    padded_slices = hsfm.core.pad_image_frame_slices(slices)
    
    if noisify == 'left':
        padded_slices[0] = hsfm.core.noisify_template(padded_slices[0])
    elif noisify == 'top':
        padded_slices[1] = hsfm.core.noisify_template(padded_slices[1])
    elif noisify == 'right':
        padded_slices[2] = hsfm.core.noisify_template(padded_slices[2])
    elif noisify == 'bottom':
        padded_slices[3] = hsfm.core.noisify_template(padded_slices[3])
          
    # detect fiducial markers
    fiducials = hsfm.core.detect_fiducials(padded_slices, windows, templates)

    # detect principal point
    principal_point = hsfm.core.principal_point(fiducials[0],
                                                fiducials[1],
                                                fiducials[2],
                                                fiducials[3])
    return fiducials, principal_point
       
def preprocess_images(camera_positions_file_name, 
                      template_directory,
                      image_type='pid_tiff', 
                      output_directory='data/images',
                      subset=None, 
                      scale=None,
                      qc=False):
                      
    """
    Function to preprocess images from NAGAP archive in batch.
    """
    # TODO
    # - Reduce redundancy with preprocess_images_from_directory function
    # - Handle image io where possible with gdal instead of opencv in order to
    #   optimize io from url and tiling and compression of final output.
    # - Generalize for other types of images
    # - Add affine transformation
                      
    hsfm.io.create_dir(output_directory)             
                      
    left_template = os.path.join(template_directory,'L.jpg')
    top_template = os.path.join(template_directory,'T.jpg')
    right_template = os.path.join(template_directory,'R.jpg')
    bottom_template = os.path.join(template_directory,'B.jpg')
    templates = [left_template, top_template, right_template, bottom_template]
    
    intersections =[]
    file_names = []
    
    df = hsfm.core.select_images_for_download(camera_positions_file_name, subset)
    targets = dict(zip(df[image_type], df['fileName']))
    
    for pid, file_name in targets.items():
        print('Processing',file_name)
        img = hsfm.core.download_image(pid)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        window_left = [5000,7000,250,1500]
        window_top = [0,500,6000,7200]
        window_right = [5000,6500,12000,img_gray.shape[1]]
        window_bottom = [11000,img_gray.shape[0],6000,7200]
        windows = [window_left, window_top, window_right, window_bottom]
        
        side = hsfm.core.evaluate_image_frame(img)
        
        fiducials, principal_point = detect_fiducials_and_principal_point(windows, 
                                                                          templates, 
                                                                          img_gray)
     
        # QC routine
        intersection_angle = hsfm.core.determine_intersection_angle(fiducials)
        print('Principal point intersection angle:',intersection_angle)
        
        intersections.append(intersection_angle)
        file_names.append(file_name)
        
        if intersection_angle > 90.1 or intersection_angle < 89.9:
            print("Warning: intersection angle at principle point is not within orthogonality limits.")
            print('Re-attempting fiducial marker detection.')
            print("Processing left fiducial.")
            fiducials, principal_point = detect_fiducials_and_principal_point(windows, 
                                                                              templates, 
                                                                              img_gray,
                                                                              noisify='left')
            intersection_angle = hsfm.core.determine_intersection_angle(fiducials)
            print('New intersection angle:',intersection_angle)
            if intersection_angle > 90.1 or intersection_angle < 89.9:
                print("Processing top fiducial.")
                
                fiducials, principal_point = detect_fiducials_and_principal_point(windows, 
                                                                                  templates, 
                                                                                  img_gray,
                                                                                  noisify='top')
                intersection_angle = hsfm.core.determine_intersection_angle(fiducials)
                print('New intersection angle:',intersection_angle)
                if intersection_angle > 90.1 or intersection_angle < 89.9:
                    print("Processing right fiducial.")
                    fiducials, principal_point = detect_fiducials_and_principal_point(windows, 
                                                                                      templates, 
                                                                                      img_gray,
                                                                                      noisify='right')
                    intersection_angle = hsfm.core.determine_intersection_angle(fiducials)
                    print('New intersection angle:',intersection_angle)
                    if intersection_angle > 90.1 or intersection_angle < 89.9:
                        print("Processing bottom fiducial.")
                        fiducials, principal_point = detect_fiducials_and_principal_point(windows, 
                                                                                          templates, 
                                                                                          img_gray,
                                                                                          noisify='bottom')
                        intersection_angle = hsfm.core.determine_intersection_angle(fiducials)
                        print('New intersection angle:',intersection_angle)
                        print("Unable to improve result", file_name)
            
        if intersection_angle < 90.1 and intersection_angle > 89.9:
            cropped = hsfm.core.crop_about_principal_point(img, principal_point)
            img_rot = hsfm.core.rotate_camera(cropped, side=side)
            out = os.path.join(output_directory, file_name+'.tif')
            cv2.imwrite(out,img_rot)
            final_output = hsfm.utils.optimize_geotif(out)
            os.remove(out)
            os.rename(final_output, out)
            
            
        if qc == True:
            hsfm.plot.plot_principal_point_and_fiducial_locations(img,
                                                                  left_fiducial,
                                                                  top_fiducial,
                                                                  right_fiducial,
                                                                  bottom_fiducial,
                                                                  principal_point,
                                                                  file_name,
                                                                  output_directory='qc/image_preprocessing/')
            
    if qc == True:
        df = pd.DataFrame({"Angle off mean":intersections,"filename":file_names}).set_index("filename")
        df_mean = df - df.mean()
        fig, ax = plt.subplots(1, figsize=(10, 10))
        df_mean.plot.bar(grid=True,ax=ax)
        plt.show()
        fig.savefig('qc/image_preprocessing/principal_point_intersection_angle_off_mean.png')
        plt.close()
        print("Mean rotation off 90 degree intersection at principal point:",(df.mean() - 90).values[0])
        print("Further QC plots for principal point and fiducial marker detection available under qc/image_preprocessing/")
        
    
    return output_directory
    
def preprocess_images_from_directory(image_directory,
                                     template_directory,
                                     output_directory='data/images',
                                     subset=None, 
                                     scale=None,
                                     qc=False):
                      
    """
    Function to preprocess images from NAGAP archive in batch.
    """
    # TODO
    # - Reduce redundancy with preprocess_images function
                      
    hsfm.io.create_dir(output_directory)             
                      
    left_template = os.path.join(template_directory,'L.jpg')
    top_template = os.path.join(template_directory,'T.jpg')
    right_template = os.path.join(template_directory,'R.jpg')
    bottom_template = os.path.join(template_directory,'B.jpg')
    templates = [left_template, top_template, right_template, bottom_template]
    
    intersections =[]
    file_names = []
    
    image_files = sorted(glob.glob(os.path.join(image_directory,'*.tif')))
    
    for image_file in image_files:
        file_path, file_name, file_extension = hsfm.io.split_file(image_file)
        print('Processing',file_name)
        
        img = cv2.imread(image_file)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        window_left = [5000,7000,250,1500]
        window_top = [0,500,6000,7200]
        window_right = [5000,6500,12000,img_gray.shape[1]]
        window_bottom = [11000,img_gray.shape[0],6000,7200]
        windows = [window_left, window_top, window_right, window_bottom]
    
        side = hsfm.core.evaluate_image_frame(img)
    
        fiducials, principal_point = detect_fiducials_and_principal_point(windows, 
                                                                          templates, 
                                                                          img_gray)
 
        # QC routine
        intersection_angle = hsfm.core.determine_intersection_angle(fiducials)
        print('Principal point intersection angle:',intersection_angle)
    
        intersections.append(intersection_angle)
        file_names.append(file_name)
    
        if intersection_angle > 90.1 or intersection_angle < 89.9:
            print("Warning: intersection angle at principle point is not within orthogonality limits.")
            print('Re-attempting fiducial marker detection.')
            print("Processing left fiducial.")
            fiducials, principal_point = detect_fiducials_and_principal_point(windows, 
                                                                              templates, 
                                                                              img_gray,
                                                                              noisify='left')
            intersection_angle = hsfm.core.determine_intersection_angle(fiducials)
            print('New intersection angle:',intersection_angle)
            if intersection_angle > 90.1 or intersection_angle < 89.9:
                print("Processing top fiducial.")
            
                fiducials, principal_point = detect_fiducials_and_principal_point(windows, 
                                                                                  templates, 
                                                                                  img_gray,
                                                                                  noisify='top')
                intersection_angle = hsfm.core.determine_intersection_angle(fiducials)
                print('New intersection angle:',intersection_angle)
                if intersection_angle > 90.1 or intersection_angle < 89.9:
                    print("Processing right fiducial.")
                    fiducials, principal_point = detect_fiducials_and_principal_point(windows, 
                                                                                      templates, 
                                                                                      img_gray,
                                                                                      noisify='right')
                    intersection_angle = hsfm.core.determine_intersection_angle(fiducials)
                    print('New intersection angle:',intersection_angle)
                    if intersection_angle > 90.1 or intersection_angle < 89.9:
                        print("Processing bottom fiducial.")
                        fiducials, principal_point = detect_fiducials_and_principal_point(windows, 
                                                                                          templates, 
                                                                                          img_gray,
                                                                                          noisify='bottom')
                        intersection_angle = hsfm.core.determine_intersection_angle(fiducials)
                        print('New intersection angle:',intersection_angle)
                        print("Unable to improve result", file_name)
        
        if intersection_angle < 90.1 and intersection_angle > 89.9:
            cropped = hsfm.core.crop_about_principal_point(img, principal_point)
            img_rot = hsfm.core.rotate_camera(cropped, side=side)
            out = os.path.join(output_directory, file_name+'.tif')
            cv2.imwrite(out,img_rot)
            final_output = hsfm.utils.optimize_geotif(out)
            os.remove(out)
            os.rename(final_output, out)
        
        
        if qc == True:
            hsfm.plot.plot_principal_point_and_fiducial_locations(img,
                                                                  left_fiducial,
                                                                  top_fiducial,
                                                                  right_fiducial,
                                                                  bottom_fiducial,
                                                                  principal_point,
                                                                  file_name,
                                                                  output_directory='qc/image_preprocessing/')
        
    if qc == True:
        df = pd.DataFrame({"Angle off mean":intersections,"filename":file_names}).set_index("filename")
        df_mean = df - df.mean()
        fig, ax = plt.subplots(1, figsize=(10, 10))
        df_mean.plot.bar(grid=True,ax=ax)
        plt.show()
        fig.savefig('qc/image_preprocessing/principal_point_intersection_angle_off_mean.png')
        plt.close()
        print("Mean rotation off 90 degree intersection at principal point:",(df.mean() - 90).values[0])
        print("Further QC plots for principal point and fiducial marker detection available under qc/image_preprocessing/")
    

    return output_directory