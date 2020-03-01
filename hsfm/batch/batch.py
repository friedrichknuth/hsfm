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

"""
Wrappers around other hsfm functions for batch processing. 
Inputs are general a folder contaning multiple files or a csv listing
multiple urls.
"""

def prepare_ba_run(input_directory,
                   output_directory,
                   scale):
    
    
    camera_solve_directory = os.path.join(output_directory, 'camera_solve')
    bundle_adjust_directory = os.path.join(output_directory,'ba')
    images_directory = os.path.join(output_directory,'images'+'_sub'+str(scale))
    gcp_directory = os.path.join(input_directory,'gcp')

    hsfm.io.batch_rename_files(
            camera_solve_directory,
            file_extension=str(scale)+'.match',
            destination_file_path=bundle_adjust_directory)

    overlap_list = hsfm.core.create_overlap_list_from_match_files(camera_solve_directory,
                                                                  images_directory,
                                                                  output_directory)

    if not os.path.exists(os.path.join(bundle_adjust_directory,'overlaplist.txt')):
        gcp_directory = os.path.join(input_directory,'gcp')
        overlap_list = hsfm.core.create_overlap_list(gcp_directory,
                                                     images_directory,
                                                     output_directory=output_directory)
        
    return os.path.join(bundle_adjust_directory,'overlaplist.txt')
    
    
    
def prepare_stereo_run(output_directory):
    
    bundle_adjust_directory = os.path.join(output_directory, 'ba')
    stereo_input_directory = os.path.join(output_directory, 'stereo/stereo_inputs')
    stereo_output_directory = os.path.join(output_directory, 'stereo/stereo_run')

    hsfm.io.batch_rename_files(
        bundle_adjust_directory,
        file_extension='tsai',
        destination_file_path=stereo_input_directory)

    hsfm.io.batch_rename_files(
        bundle_adjust_directory,
        file_extension='clean.match',
        destination_file_path=stereo_input_directory)



def rescale_images(image_directory,
                   output_directory,
                   extension='.tif',
                   scale=8,
                   verbose=False):
    
    output_directory = os.path.join(output_directory, 'images'+'_sub'+str(scale))
    hsfm.io.create_dir(output_directory)
    
    image_files  = sorted(glob.glob(os.path.join(image_directory,'*'+ extension)))
    
    for image_file in image_files:
        
        file_path, file_name, file_extension = hsfm.io.split_file(image_file)
        output_file = os.path.join(output_directory, 
                                   file_name +'_sub'+str(scale)+file_extension)
        
        hsfm.utils.rescale_geotif(image_file,
                                  output_file_name=output_file,
                                  scale=scale,
                                  verbose=verbose)

    return os.path.relpath(output_directory)
#     return sorted(glob.glob(os.path.join(output_directory,'*'+ extension)))

def rescale_tsai_cameras(camera_directory,
                         output_directory,
                         extension='.tsai',
                         scale=8):

    output_directory = os.path.join(output_directory, 'cameras'+'_sub'+str(scale))
    hsfm.io.create_dir(output_directory)
    
    pitch = "pitch = 1"
    new_pitch = "pitch = "+str(scale)
    
    camera_files  = sorted(glob.glob(os.path.join(camera_directory,'*'+ extension)))
                 
    for camera_file in camera_files:
        
        file_path, file_name, file_extension = hsfm.io.split_file(camera_file)
        output_file = os.path.join(output_directory, 
                                   file_name +'_sub'+str(scale)+file_extension)
                                   
        
        hsfm.io.replace_string_in_file(camera_file, output_file, pitch, new_pitch)
        
    return os.path.relpath(output_directory)
#     return sorted(glob.glob(os.path.join(output_directory,'*'+ extension)))
    
    
def batch_generate_cameras(image_directory,
                           camera_positions_file_name,
                           reference_dem_file_name,
                           focal_length_mm,
                           output_directory,
                           pixel_pitch_mm=0.02,
                           verbose=False,
                           subset=None,
                           manual_heading_selection=False,
                           reverse_order=False):
                           
    """
    Function to generate cameras in batch.
                           
    Note:
        - Specifying subset as a tuple indicates selecting a range of values, while supplying
          a list allows for single or multiple specific image selection.
    """
    
    # TODO
    # - Embed hsfm.utils.pick_headings() within calculate_heading_from_metadata() and launch for            images where the heading could not be determined with high confidence (e.g. if image
    #   potentially part of another flight line, or at the end of current flight line with no
    #   subsequent image to determine flight line from.)
    # - provide principal_point_px to hsfm.core.initialize_cameras on a per image basis
    # put gcp generation in a seperate batch routine
    
    image_list = sorted(glob.glob(os.path.join(image_directory, '*.tif')))
    image_list = hsfm.core.subset_input_image_list(image_list, subset=subset)
    
    if reverse_order:
        image_list = image_list[::-1]
    
    if manual_heading_selection == False:
        df = hsfm.batch.calculate_heading_from_metadata(camera_positions_file_name,
                                                        output_directory, 
                                                        subset=subset,
                                                        reverse_order=reverse_order)
    else:
        df = hsfm.utils.pick_headings(image_directory, camera_positions_file_name, subset, delta=0.01)
    
    if len(image_list) != len(df):
        print('Mismatch between metadata entries in camera position file and available images.')
        sys.exit(1)
    
    for i,v in enumerate(image_list):
        image_file_name = v
        camera_lat_lon_center_coordinates = (df['Latitude'].iloc[i], df['Longitude'].iloc[i])
        heading = df['heading'].iloc[i]
        
        gcp_directory = hsfm.core.prep_and_generate_gcp(image_file_name,
                                                        camera_lat_lon_center_coordinates,
                                                        reference_dem_file_name,
                                                        focal_length_mm,
                                                        heading,
                                                        output_directory)
        
    
        # principal_point_px is needed to initialize the cameras in the next step.
        img_ds = gdal.Open(image_file_name)
        image_width_px = img_ds.RasterXSize
        image_height_px = img_ds.RasterYSize
        principal_point_px = (image_width_px / 2, image_height_px /2 )
    
    focal_length_px = focal_length_mm / pixel_pitch_mm
    
    # should be using principal_point_px on a per image basis
    intial_cameras_directory = hsfm.core.initialize_cameras(camera_positions_file_name, 
                                                            reference_dem_file_name,
                                                            focal_length_px,
                                                            principal_point_px,
                                                            output_directory)
    
    output_directory = hsfm.asp.generate_ba_cameras(image_directory,
                                                    gcp_directory,
                                                    intial_cameras_directory,
                                                    output_directory,
                                                    subset=subset) 
    return output_directory


def calculate_heading_from_metadata(camera_positions_file_name,
                                    output_directory,
                                    subset=None,
                                    reverse_order     = False,
                                    for_metashape     = False,
                                    reference_dem     = None,
                                    flight_altitude_m = 1500,
                                    sorting_column = 'fileName'):
    # TODO
    # - Add flightline seperation function
    # - Generalize beyond NAGAP keys
    df = hsfm.core.select_images_for_download(camera_positions_file_name, subset)
    df = df.sort_values(by=[sorting_column])
    if reverse_order:
        df = df.sort_values(by=[sorting_column], ascending=False)
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
    
    if for_metashape:
        df['yaw']             = df['heading'].round()
        df['pitch']           = 1.0
        df['roll']            = 1.0
        df['image_file_name'] = df['fileName']+'.tif'
        
        if reference_dem:
            df['alt']             = hsfm.geospatial.sample_dem(lons, lats, reference_dem)
            df['alt']             = df['alt'] + flight_altitude_m
            df['alt']             = df['alt'].max()
        
        else:
            df['alt']             = flight_altitude_m
            
        df['lon']             = df['Longitude'].round(6)
        df['lat']             = df['Latitude'].round(6)
        df['lon_acc']         = 1000
        df['lat_acc']         = 1000
        df['alt_acc']         = 1000
        df['yaw_acc']         = 50
        df['pitch_acc']       = 50
        df['roll_acc']        = 50
    
        df = df[['image_file_name',
                 'lon',
                 'lat',
                 'alt',
                 'lon_acc',
                 'lat_acc',
                 'alt_acc',
                 'yaw',
                 'pitch',
                 'roll',
                 'yaw_acc',
                 'pitch_acc',
                 'roll_acc']]
        df.to_csv(os.path.join(output_directory,'metashape_metadata.csv'),index=False)
        
        return df
    
    else:
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
        img_gray = hsfm.core.download_image(pid)
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
                      qc=False,
                      invisible_fiducial=None,
                      crop_from_pp_dist = 11250):
                      
    """
    Function to preprocess images from NAGAP archive in batch.
    """
    # TODO
    # - Make io faster with gdal
    # - Generalize for other types of images
    # - Add affine transformation (if needed)
                      
    hsfm.io.create_dir(output_directory)
    
    templates = hsfm.core.gather_templates(template_directory)         
                      
    intersections =[]
    file_names = []
    
    if camera_positions_file_name:
        df = hsfm.core.select_images_for_download(camera_positions_file_name, subset)
        targets = dict(zip(df[image_type], df['fileName']))
        for pid, file_name in targets.items():
            print('Processing',file_name)
            img_gray = hsfm.core.download_image(pid)
            intersection_angle = hsfm.core.preprocess_image(img_gray,
                                                            file_name,
                                                            templates, 
                                                            qc=qc,
                                                            output_directory=output_directory,
                                                            invisible_fiducial=invisible_fiducial,
                                                            crop_from_pp_dist=crop_from_pp_dist)
            intersections.append(intersection_angle)
            file_names.append(file_name)
    
    elif image_directory:
        image_files = sorted(glob.glob(os.path.join(image_directory,'*.tif')))
        for image_file in image_files:
            file_path, file_name, file_extension = hsfm.io.split_file(image_file)
            print('Processing',file_name)
            
            ## TODO
            ## - Potentially make io faster with gdal
            # src = gdal.Open(image_file)
            # arr = src.ReadAsArray()
            # if len(arr.shape) == 3:
            #     img = arr.reshape(arr.shape[1], arr.shape[2], arr.shape[0]).shape
            #     img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #
            # else:
            #     img_gray = arr
                
            img_gray = cv2.imread(image_file)
            
            intersection_angle = hsfm.core.preprocess_image(img_gray, 
                                                            file_name,
                                                            templates, 
                                                            qc=qc,
                                                            output_directory=output_directory,
                                                            invisible_fiducial=invisible_fiducial,
                                                            crop_from_pp_dist=crop_from_pp_dist)
            intersections.append(intersection_angle)
            file_names.append(file_name)
        
    if qc == True:
        hsfm.plot.plot_intersection_angles_qc(intersections, file_names)
    
    return output_directory

def plot_match_overlap(match_files_directory, images_directory, output_directory='qc/matches/'):
    
    out = os.path.split(match_files_directory)[-1]
    output_directory = os.path.join(output_directory,out)
    hsfm.io.create_dir(output_directory)
    
    matches=sorted(glob.glob(os.path.join(match_files_directory,'*.csv')))
    images=sorted(glob.glob(os.path.join(images_directory,'*.tif')))
    
    df_combined, keys = hsfm.qc.match_files_to_combined_df(matches)
        
    fig_size_y = len(matches)*3
    fig, ax = plt.subplots(len(keys),2,figsize=(10,fig_size_y),sharex='col',sharey=True)
    for i,v in enumerate(keys):
        
        left_title = v.split('__')[0]
        right_title = v.split('__')[1]
        
        ax[i][0].scatter(df_combined.xs(keys[i])['x1'], df_combined.xs(keys[i])['y1'],color='r',marker='.')
        ax[i][1].scatter(df_combined.xs(keys[i])['x2'], df_combined.xs(keys[i])['y2'],color='r',marker='.')
        
        left_image = hsfm.io.retrieve_match(left_title, images)
        left_image = gdal.Open(left_image)
        left_image = left_image.ReadAsArray()
        clim = np.percentile(left_image, (2,98))
        ax[i][0].imshow(left_image, clim=clim, cmap='gray')
        
        right_image = hsfm.io.retrieve_match(right_title, images)
        right_image = gdal.Open(right_image)
        right_image = right_image.ReadAsArray()
        clim = np.percentile(right_image, (2,98))
        ax[i][1].imshow(right_image, clim=clim, cmap='gray')
        
        ax[i][0].set_title(left_title)
        ax[i][1].set_title(right_title)
        
        ax[i][0].set_aspect('equal')
        ax[i][1].set_aspect('equal')
    
    
    plt.tight_layout()
    out = os.path.join(output_directory,'match_plot.png')
    plt.savefig(out)
    return out