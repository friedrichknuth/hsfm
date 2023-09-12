from osgeo import gdal
import os
import cv2
import sys
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import concurrent.futures
import psutil
import pathlib
import shutil
import time
import geopandas as gpd
import traceback
from pathlib import Path

import hipp
import hsfm

"""
Wrappers around other hsfm functions for batch processing. 
Inputs are general a folder contaning multiple files or a csv listing
multiple urls.
"""

def prepare_ba_run(input_directory,
                   output_directory,
                   scale):
    
    
    camera_solve_directory = os.path.join(output_directory, 'cam_solve')
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
    
    
    image_files  = sorted(glob.glob(os.path.join(image_directory,'*'+ extension)))
    
#     n = len(psutil.Process().cpu_affinity())
#     pool = concurrent.futures.ThreadPoolExecutor(max_workers=n)
    
#     parallel_data = {pool.submit(hsfm.utils.rescale_geotif,
#                                  image_file,
#                                  output_directory=output_directory,
#                                  scale=scale): \
#                      image_file for image_file in image_files}
    
#     for future in concurrent.futures.as_completed(parallel_data):
#         r = future.result()
#         if verbose:
#             print(r)

    for image_file in image_files:
        
        hsfm.utils.rescale_geotif(image_file,
                                  output_directory=output_directory,
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
    # - Embed hsfm.utils.pick_headings() within calculate_heading_from_metadata() and launch for
    #   images where the heading could not be determined with high confidence (e.g. if image
    #   potentially part of another flight line, or at the end of current flight line with no
    #   subsequent image to determine flight line from.)
    # - provide principal_point_px to hsfm.core.initialize_cameras on a per image basis
    #   put gcp generation in a seperate batch routine
    
    image_list = sorted(glob.glob(os.path.join(image_directory, '*.tif')))
    image_list = hsfm.core.subset_input_image_list(image_list, subset=subset)
    
    if reverse_order:
        image_list = image_list[::-1]
    
    if manual_heading_selection == False:
        df = hsfm.batch.calculate_heading_from_metadata(camera_positions_file_name,
                                                        output_directory=output_directory, 
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


def calculate_heading_from_metadata(df,
                                    subset                         = None,
                                    reverse_order                  = False,
                                    output_directory               = None,
                                    for_metashape                  = False,
                                    reference_dem                  = None,
                                    flight_altitude_above_ground_m = 1500,
                                    file_base_name_column          = 'fileName',
                                    longitude_column               = 'Longitude',
                                    latitude_column                = 'Latitude'):
    # TODO
    # - Add flightline seperation function
    if not isinstance(df, type(pd.DataFrame())):
        df = pd.read_csv(df)
        
    if not isinstance(subset, type(None)):
        df = hsfm.core.subset_images_for_download(df, subset)
        
    df = df.sort_values(by=[file_base_name_column])
    if reverse_order:
        df = df.sort_values(by=[file_base_name_column], ascending=False)
    lons = df[longitude_column].values
    lats = df[latitude_column].values
    
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
            
    df = df.sort_values(by=[file_base_name_column], ascending=True)   
    df['heading'] = headings
    
    if for_metashape:
        
        df['yaw']             = df['heading'].round()
        df['pitch']           = 1.0
        df['roll']            = 1.0
        df['image_file_name'] = df[file_base_name_column]+'.tif'
        
        if reference_dem:
            df['alt']             = hsfm.geospatial.sample_dem(lons, lats, reference_dem)
            df['alt']             = df['alt'] + flight_altitude_above_ground_m
            df['alt']             = df['alt'].max()
        
        else:
            df['alt']             = flight_altitude_above_ground_m
            
        df['lon']             = df[longitude_column].round(6)
        df['lat']             = df[latitude_column].round(6)
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
                 
        if output_directory:
            hsfm.io.create_dir(output_directory)
            df.to_csv(os.path.join(output_directory,'metashape_metadata.csv'),index=False)
        
        return df
    
    else:
        return df

def download_images_to_disk(image_metadata, 
                            output_directory = 'output_data/raw_images',
                            image_type = 'pid_tiff',
                            image_file_name_column = 'fileName',
                            image_extension = '.tif'):
                            
    if not isinstance(image_metadata, type(pd.DataFrame())):
        df = pd.read_csv(image_metadata)
    else:
        df = image_metadata
    
    hsfm.io.create_dir(output_directory)
    
    
    targets = dict(zip(df[image_type], df[image_file_name_column]))
    for pid, file_name in targets.items():
        print('Downloading',file_name, image_type)
        img_gray = hsfm.core.download_image(pid)
        out = os.path.join(output_directory, file_name+image_extension)
        cv2.imwrite(out,img_gray)
        final_output = hsfm.utils.optimize_geotif(out)
        os.remove(out)
        os.rename(final_output, out)
    
    return output_directory
        

def EE_pre_process_images(
        apiKey,
        project_name,
        bounds,
        ee_project_name,
        year,
        month,
        day,
#         pixel_pitch               = 0.025,
        buffer_m                  = 40000,
        threshold_px              = 50,
        missing_proxy             = None,
        download_images           = True,
        preprocess_images         = True,
        template_parent_dir       = None,
        output_directory          = '../',
        ee_query_max_results      = 50000,
        ee_query_label            = 'test_download',
        EE_find_matching_template = False,
        invert_color              = False,
        prime_for_download_later = False,
#         overwrite                 = False,
    ):
    """
    Download and preprocess images from the EE archive.
    Assumes you already have fiducial marker proxy template files available (see examples in the HIPP repo).
    Requires an api key to be passed in, which can be retrieved using hipp.dataquery.EE_login.
    
    Args:
        project_name ([type]): [description]
        bounds ([type]): Tuple or list in order: ULLON, ULLAT, LRLON, LRLAT
        ee_project_name ([type], optional): [description]. Defaults to None.
        year ([type], optional): [description]. Defaults to None.
        month ([type], optional): [description]. Defaults to None.
        day ([type], optional): [description]. Defaults to None.
        pixel_pitch ([type], optional): [description]. Defaults to None.
        focal_length ([type], optional): [description]. Defaults to None.
        buffer_m (int, optional): [description]. Defaults to 2000.
        threshold_px (int, optional): [description]. Defaults to 50.
        missing_proxy ([type], optional): [description]. Defaults to None.
        keep_raw (bool, optional): [description]. Defaults to True.
        download_images (bool, optional): [description]. Defaults to True.
        image_square_dim ([type], optional): [description]. Defaults to None.
        template_parent_dir ([type], optional): [description]. Defaults to None.
        output_directory (str, optional): [description]. Defaults to '../'.
        ee_query_max_results (int, optional): [description]. Defaults to 50000.
        ee_query_label (string): The label required by the EE api. Defaults to "test_download" only because thats a previously used value that makes downloading easy for me personally.
        skip_download (bool, optional): Skip download step of this process. Useful if you have called this function before (with the same arguments) and something bad happened. Defaults to False
    """

    print('\n')
    print('EE Project ID:',ee_project_name)
    print('Date:', '-'.join([str(year),
                             str(month).zfill(2),
                             str(day).zfill(2)]))
    output_directory = os.path.join(output_directory, project_name, 'input_data')

    #TODO fix behavior of this query - you can get more than the maxResults parameter returned
    # Also remove the head() hack that fixes the issue
    ULLON, ULLAT, LRLON, LRLAT = bounds
    ee_results_df = hipp.dataquery.EE_pre_select_images(
        apiKey,
        xmin = LRLON,
        ymin = LRLAT,
        xmax = ULLON,
        ymax = ULLAT,
        startDate = f"{year}-{month}-{day}",
        endDate = f"{year}-{month}-{day}",
        maxResults   = ee_query_max_results
    ).head(ee_query_max_results)

    ee_results_df = ee_results_df[ee_results_df['project'] == ee_project_name]
    
    if len(ee_results_df['entityId'].tolist()) == 1:
          print('Skipping record with only 1 image')
    elif len(ee_results_df['entityId'].tolist()) > 1:
        download_directory = os.path.join(output_directory, 
                                          f"EE_{year}", 
                                          str(month).zfill(2), str(day).zfill(2))

        # If you do not download the images, 
        # you still need to retrieve the directories and pixel_pitch
        raw_images_directory_name = 'raw_images'
        r = hipp.dataquery.EE_download_images_to_disk(
            apiKey,
            ee_results_df['entityId'].tolist(),
            ee_query_label,
            download_directory,
            images_directory_suffix=raw_images_directory_name,
            invert_color=invert_color,
            overwrite = download_images,
            prime_for_download_later = prime_for_download_later,
        )
        raw_images_directory, calibration_reports_directory, pixel_pitch = r

        if prime_for_download_later:
            print('Requests primed for download later')
            return None
        
        if raw_images_directory:
            preprocessed_images_directory = raw_images_directory.replace('raw_images', 'cropped_images')
            qc_directory = preprocessed_images_directory.replace("cropped_images", "preprocess_qc")

            if pathlib.Path(raw_images_directory).is_dir() and preprocess_images:
                hipp.batch.preprocess_with_fiducial_proxies(
                    raw_images_directory,
                    template_parent_dir,
                    output_directory=preprocessed_images_directory,
                    verbose=True,
                    missing_proxy=missing_proxy,
                    threshold_px=threshold_px,
                    qc_df=True,
                    qc_df_output_directory=os.path.join(qc_directory, 'proxy_detection_data_frames'),
                    qc_plots=True,
                    qc_plots_output_directory=os.path.join(qc_directory, 'proxy_detection'),
                    EE_find_matching_template = EE_find_matching_template
                )

                # Generate metashape_metadata.csv file for all images

            metashape_metadata_df = ee_results_df[[
                'entityId',
                'centerLon',
                'centerLat',
                'altitudesFeet', #needs conversion to meters
                'focalLength'
            ]]

            convert_column_names = {
                'entityId': 'image_file_name',
                'centerLon': 'lon',
                'centerLat': 'lat',
                'altitudesFeet': 'alt',
                'focalLength': 'focal_length'
            }

            metashape_metadata_df = metashape_metadata_df.rename(convert_column_names, axis=1)

            #convert feet to meters
            FEET_IN_1_METER = 3.28084
            metashape_metadata_df['alt'] = metashape_metadata_df['alt'] / FEET_IN_1_METER

            # Add default values
            metashape_metadata_df['lon_acc'] = 1000
            metashape_metadata_df['lat_acc'] = 1000
            metashape_metadata_df['alt_acc'] = 1000
            metashape_metadata_df['yaw'] = 0
            metashape_metadata_df['pitch'] = 0 
            metashape_metadata_df['roll'] = 0
            metashape_metadata_df['yaw_acc'] = 180
            metashape_metadata_df['pitch_acc'] = 20
            metashape_metadata_df['roll_acc'] = 20
            metashape_metadata_df['pixel_pitch'] = pixel_pitch

            target_column_names_in_order = [
                'image_file_name',
                'lon',
                'lat',
                'alt',
                'lon_acc', #default 1000
                'lat_acc', #default 1000
                'alt_acc', #default 1000
                'yaw', #default 0
                'pitch', #default 0
                'roll', #default 0
                'yaw_acc', #default 180
                'pitch_acc', #default 20
                'roll_acc', #default 20
                'focal_length',
                'pixel_pitch'
            ]

            #order the columns appropriately
            metashape_metadata_df = metashape_metadata_df[target_column_names_in_order]


            def estimate_image_footprint_from_ee_results(df, arcsecond_to_meters=111111):
                points1 = gpd.points_from_xy(x=df['NElon'], y=df['NElat'])
                points2 = gpd.points_from_xy(x=df['NWlon'], y=df['NWlat'])
                lat_distances = points1.distance(points2)

                points1 = gpd.points_from_xy(x=df['NElon'], y=df['NElat'])
                points2 = gpd.points_from_xy(x=df['SElon'], y=df['SElat'])
                lon_distances = points1.distance(points2)
                max_edge_length = np.concatenate([lat_distances, lon_distances]).max()
                return max_edge_length * arcsecond_to_meters

            qc = False
            if not buffer_m:
                qc = True
                buffer_m = estimate_image_footprint_from_ee_results(ee_results_df)

            # Run coarse (crude) regional cluster detection 
            hsfm.core.determine_image_clusters(
                metashape_metadata_df,
                pixel_pitch = pixel_pitch,
                output_directory = raw_images_directory.replace("raw_images", "sfm"),
                buffer_m         = buffer_m,
                image_file_name_column= 'image_file_name',
                qc=qc,
                image_metadata_longitude_column = 'lon',
                image_metadata_latitude_column = 'lat',
                image_metadata_altitude_column = 'alt',
            )
            return None


    


def NAGAP_pre_process_images(project_name,
                             bounds,
                             roll                = None,
                             year                = None,
                             month               = None,
                             day                 = None,
                             pixel_pitch         = 0.02,
                             focal_length        = None,
                             buffer_m            = 40000,
                             threshold_px        = 50,
                             missing_proxy       = None,
                             keep_raw            = True,
                             download_images     = True,
                             preprocess_images   = True,
                             image_square_dim    = None,
                             stretch_histogram   = True,
                             clahe_enhancement   = True,
                             template_parent_dir = '../input_data/fiducials/nagap',
                             nagap_metadata_csv  = '../input_data/nagap_image_metadata.csv',
                             output_directory    = '../'):
        
    output_directory = os.path.join(output_directory, project_name, 'input_data')
    
    template_dirs = sorted(glob.glob(os.path.join(template_parent_dir, '*')))
    template_types = []
    for i in template_dirs:
        template_types.append(i.split('/')[-1])
        
    df = hipp.dataquery.NAGAP_pre_select_images(nagap_metadata_csv, 
                                                bounds = bounds,
                                                roll   = roll,
                                                year   = year,
                                                month  = month,
                                                day    = day)
    df = df[df['fiducial_proxy_type'].isin(template_types)]
    
    
    if isinstance(roll, type(None)):
        rolls = sorted(list(set(df['Roll'].values)))
    else:
        rolls = [roll,]

    for roll in rolls:
        df_roll = df[df['Roll']  == roll].copy()
        print('Processing roll:', roll, sep = "\n")
        out_dir_roll = os.path.join(output_directory,roll)
        
        if len(list(set(df_roll['Month'].values))) >=1:
            for month in sorted(list(set(df_roll['Month'].values))):
                print('Processing month:', month, sep = "\n")
                try:
                    out_dir_month = os.path.join(out_dir_roll,str(int(month)).zfill(2))
                except:
                    out_dir_month = os.path.join(out_dir_roll,str(month).zfill(2))
                df_month = df_roll[df_roll['Month']  == month].copy()
                if len(list(set(df_month['Day'].values))) >=1:
                    for day in sorted(list(set(df_month['Day'].values))):
                        print('Processing day:', day, sep = "\n")
                        try:
                            out_dir_day = os.path.join(out_dir_month,str(int(day)).zfill(2))
                        except:
                            out_dir_day = os.path.join(out_dir_month,str(day).zfill(2))
                        df_day = df_month[df_month['Day']  == day].copy()
                        hsfm.batch.NAGAP_pre_process_set(df_day,
                                                         template_types,
                                                         template_dirs,
                                                         out_dir_day,
                                                         pixel_pitch     = pixel_pitch,
                                                         focal_length    = focal_length,
                                                         missing_proxy   = missing_proxy,
                                                         buffer_m        = buffer_m,
                                                         threshold_px    = threshold_px,
                                                         keep_raw        = keep_raw,
                                                         download_images = download_images,
                                                         preprocess_images = preprocess_images,
                                                         image_square_dim = image_square_dim,
                                                         stretch_histogram = stretch_histogram,
                                                         clahe_enhancement = clahe_enhancement,
                                                        )
                
                # in case no day specified in metadata
                else:
                    try:
                        out_dir_month = os.path.join(output_directory,roll,str(int(month)).zfill(2),'dd')
                    except:
                        out_dir_month = os.path.join(output_directory,roll,str(month).zfill(2),'dd')
                    hsfm.batch.NAGAP_pre_process_set(df_month,
                                                     template_types,
                                                     template_dirs,
                                                     out_dir_month,
                                                     pixel_pitch   = pixel_pitch,
                                                     focal_length  = focal_length,
                                                     missing_proxy = missing_proxy,
                                                     buffer_m      = buffer_m,
                                                     threshold_px  = threshold_px,
                                                     keep_raw      = keep_raw,
                                                     download_images = download_images,
                                                     preprocess_images = preprocess_images,
                                                     image_square_dim = image_square_dim,
                                                     stretch_histogram = stretch_histogram,
                                                     clahe_enhancement = clahe_enhancement,
                                                    )
        # in case no month specified in metadata                
        else:
            out_dir_roll = os.path.join(output_directory,roll,'mm','dd')
            hsfm.batch.NAGAP_pre_process_set(df_roll,
                                             template_types,
                                             template_dirs,
                                             out_dir_roll,
                                             pixel_pitch   = pixel_pitch,
                                             focal_length  = focal_length,
                                             missing_proxy = missing_proxy,
                                             buffer_m      = buffer_m,
                                             threshold_px  = threshold_px,
                                             keep_raw      = keep_raw,
                                             download_images = download_images,
                                             preprocess_images = preprocess_images,
                                             image_square_dim = image_square_dim,
                                             stretch_histogram = stretch_histogram,
                                             clahe_enhancement = clahe_enhancement,
                                            )
                    

                    
def NAGAP_pre_process_set(df,
                          template_types,
                          template_dirs,
                          output_directory,
                          pixel_pitch         = None,
                          focal_length        = None,
                          missing_proxy       = None,
                          buffer_m            = 40000,
                          threshold_px        = 50,
                          keep_raw            = True,
                          download_images     = True,
                          preprocess_images   = True,
                          image_square_dim    = None,
                          stretch_histogram   = True,
                          clahe_enhancement   = True):
    
    # TODO
    # check if image directory already contains raw images, else skip
                          
        for i,v in enumerate(template_types):
            df_tmp = df[df['fiducial_proxy_type']  == v].copy()
            if not df_tmp.empty and len(df_tmp.index) > 2:
                print(str(len(df_tmp)),'images found with marker type',v)
                image_directory = os.path.join(output_directory,v+'_raw_images')
                if download_images:
                    hipp.dataquery.NAGAP_download_images_to_disk(
                        df_tmp,
                        output_directory=image_directory)
                if preprocess_images:
                    template_directory = template_dirs[i]
                    image_square_dim = hipp.batch.preprocess_with_fiducial_proxies(
                                                  image_directory,
                                                  template_directory,
                                                  threshold_px = threshold_px,
                                                  image_square_dim = image_square_dim,
                                                  output_directory=os.path.join(output_directory,
                                                                                v+'_cropped_images'),
                                                  missing_proxy = missing_proxy,

                                                  qc_df_output_directory=os.path.join(output_directory,
                                                                                      'qc',
                                                                                      v+'_proxy_detection_data_frames'),
                                                  qc_plots_output_directory=os.path.join(output_directory,
                                                                                         'qc', 
                                                                                         v+'_proxy_detection_plots'),
                                                  stretch_histogram = stretch_histogram,
                                                  clahe_enhancement = clahe_enhancement,

                    )
                if not keep_raw:
                    shutil.rmtree(image_directory)

                if not focal_length:
                    focal_length = df_tmp['focal_length'].values[0]
                hsfm.core.determine_image_clusters(df_tmp,
#                                                    image_square_dim = image_square_dim,
                                                   pixel_pitch      = pixel_pitch,
                                                   focal_length     = focal_length,
                                                   output_directory = os.path.join(output_directory,'sfm'),
                                                   buffer_m         = buffer_m)
              
              ## TODO make regional cluster detection optional
#             hsfm.core.prepare_metashape_metadata(df_tmp,
#                                                  output_directory                = os.path.join(output_directory,'sfm'),
#                                                  focal_length                    = focal_length,
#                                                  pixel_pitch                     = pixel_pitch,
#                                                  image_file_name_column          = 'fileName',
#                                                  image_metadata_longitude_column = 'Longitude',
#                                                  image_metadata_latitude_column  = 'Latitude',
#                                                  image_metadata_altitude_column  = 'Altitude',
#             )

            elif len(df_tmp.index) <= 2 and len(df_tmp.index) !=0:
                print("Only",str(len(df_tmp)),'images found. Skipping.')
                

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
    
def pick_camera_locations(image_directory, 
                          camera_positions_file_name,
                          center_lon, 
                          center_lat,
                          image_file_column_name = 'fileName',
                          latitude_column_name = 'Latitude',
                          longitude_column_name = 'Longitude',
                          delta=0.030):
    
    df = pd.read_csv(camera_positions_file_name)
    
    image_file_paths = sorted(glob.glob(os.path.join(image_directory, '*.tif')))
    
    for i in image_file_paths:
        x, y, image_file_basename = hsfm.utils.pick_camera_location(i, 
                                                                    center_lon, 
                                                                    center_lat, 
                                                                    dx = delta,
                                                                    dy = delta)
        
        df.loc[(df[image_file_column_name] == image_file_basename),longitude_column_name] = x
        df.loc[(df[image_file_column_name] == image_file_basename),latitude_column_name]  = y

    return df

def run_metashape(project_name,
                  images_path,
                  images_metadata_file,
                  reference_dem,
                  output_path,
                  pixel_pitch,
                  focal_length            = None,
                  plot_LE90_CE90          = True,
                  camera_model_xml_files  = None,
                  image_matching_accuracy = 1,
                  densecloud_quality      = 1,
                  output_DEM_resolution   = None,
                  generate_ortho          = False,
                  dem_align_all           = True,
                  rotation_enabled        = False,
                  metashape_licence_file  = None,
                  verbose                 = False,
                  iteration               = 0,
                  cleanup                 = False,
                  overwrite               = False):
    
    now = datetime.now()
    
    output_path = output_path.rstrip('/') + str(iteration)
    bundle_adjusted_metadata_file = Path(output_path,"bundle_adjusted_metadata.csv").as_posix()
    aligned_bundle_adjusted_metadata_file = Path(output_path,"aligned_bundle_adjusted_metadata.csv").as_posix()
    
    if os.path.exists(reference_dem):
        pass
    else:
        print("\nCan't find reference DEM at",output_path)
        sys.exit(0) 
    
    if not isinstance(metashape_licence_file, type(None)):
        hsfm.metashape.authentication(metashape_licence_file)
        
    out = hsfm.metashape.images2las(project_name,
                                    images_path,
                                    images_metadata_file,
                                    output_path,
                                    focal_length            = focal_length,
                                    pixel_pitch             = pixel_pitch,
                                    camera_model_xml_files  = camera_model_xml_files,
                                    image_matching_accuracy = image_matching_accuracy,
                                    densecloud_quality      = densecloud_quality,
                                    rotation_enabled        = rotation_enabled,
                                    overwrite               = overwrite)
    
    metashape_project_file, point_cloud_file = out
    
    ba_cameras_df, unaligned_cameras_df = hsfm.metashape.update_ba_camera_metadata(metashape_project_file,
                                                                                   images_metadata_file,
                                                                                   bundle_adjusted_metadata_file=bundle_adjusted_metadata_file)

    x_offset, y_offset, z_offset = hsfm.core.compute_point_offsets(images_metadata_file, 
                                                                   bundle_adjusted_metadata_file)


    ba_CE90, ba_LE90 = hsfm.geospatial.CE90(x_offset,y_offset), hsfm.geospatial.LE90(z_offset)

    if plot_LE90_CE90:
        hsfm.plot.plot_offsets(ba_LE90,
                               ba_CE90,
                               x_offset, 
                               y_offset, 
                               z_offset,
                               title = 'Initial vs Bundle Adjusted',
                               plot_file_name = os.path.join(output_path, 'qc_ba_ce90le90.png'))

    if not output_DEM_resolution:
        print('No DEM output resolution specified.')
        print('Using Ground Sample Distance from metashape report.') 
        metashape_report = metashape_project_file.replace('.psx','_report.pdf')
        output_DEM_resolution = hsfm.core.get_DEM_resolution_from_report_GSD(metashape_report)
    
    elif not output_DEM_resolution and focal_length:
        print('No DEM output resolution specified.')
        print('Using Ground Sample Distance from mean camera altitude above ground to estimate.') 
        output_DEM_resolution = hsfm.core.estimate_DEM_resolution_from_GSD(images_metadata_file, 
                                                                           pixel_pitch,
                                                                           focal_length)
        
        output_DEM_resolution = densecloud_quality * output_DEM_resolution
        print('DEM resolution factored by densecloud quality setting:',output_DEM_resolution)
    elif not output_DEM_resolution and not focal_length:
        print('No DEM output resolution specified. No focal length specified.')
        print('Cannot compute GSD to estimate an optimal DEM resolution without a focal length.')
        print('Setting output DEM resolution to 10 m. You can regrid the las file to a higher resolution as desired.')
        output_DEM_resolution = 10
        
    epsg_code = 'EPSG:'+ hsfm.geospatial.get_epsg_code(reference_dem)
    dem = hsfm.asp.point2dem(point_cloud_file,
                             '--nodata-value','-9999',
                             '--tr',str(output_DEM_resolution),
                             '--t_srs', epsg_code,
                             verbose=verbose)

    if ba_CE90 > 0.01 or ba_LE90 > 0.01:

        # if the camera positions do not change after bundle adjustment, then
        # the cameras, DEM, and ortho should all be in the right place. 
        # further attempted alignment is unlikely to change the result,
        # if it was unsuccessful to begin with.

        clipped_reference_dem = os.path.join(output_path,'reference_dem_clip.tif')
        
        # if the reference dem is smaller in extent than the to be aligned dem - don't clip it
        # clipping is done to improve processing time as loading a large high-res reference dem
        # into memory is costly at various succeeding steps
        large_to_small_order = hsfm.geospatial.compare_dem_extent(dem, reference_dem)
        if large_to_small_order == (reference_dem, dem):
            reference_dem = hsfm.utils.clip_reference_dem(dem,
                                                          reference_dem,
                                                          output_file_name = clipped_reference_dem,
                                                          buff_size        = 2000,
                                                          verbose = verbose)

        aligned_dem_file, transform =  hsfm.asp.pc_align_p2p_sp2p(dem, 
                                                                  reference_dem,
                                                                  output_path,
                                                                  verbose = verbose)

        print("Elapsed time", str(datetime.now() - now))

        hsfm.core.metadata_transform(bundle_adjusted_metadata_file,
                                     transform,
                                     output_file_name=aligned_bundle_adjusted_metadata_file)

        x_offset, y_offset, z_offset  = hsfm.core.compute_point_offsets(bundle_adjusted_metadata_file,
                                                                        aligned_bundle_adjusted_metadata_file)

        tr_ba_CE90, tr_ba_LE90 = hsfm.geospatial.CE90(x_offset,y_offset), hsfm.geospatial.LE90(z_offset)

        if plot_LE90_CE90:
            hsfm.plot.plot_offsets(tr_ba_LE90,
                                   tr_ba_CE90,
                                   x_offset, 
                                   y_offset, 
                                   z_offset,
                                   title = 'Bundle Adjusted vs Transformed',
                                   plot_file_name = os.path.join(output_path, 'qc_tr_ba_ce90le90.png'))

        output = [bundle_adjusted_metadata_file, 
                  ba_CE90, 
                  ba_LE90, 
                  aligned_dem_file,
                  transform, 
                  aligned_bundle_adjusted_metadata_file, 
                  tr_ba_CE90, 
                  tr_ba_LE90]
        
        if dem_align_all:
            dem_align_output_path,_,_ = hsfm.io.split_file(aligned_dem_file)
            dem_difference_file , aligned_dem_file = hsfm.utils.dem_align_custom(reference_dem,
                                                                                 aligned_dem_file,
                                                                                 verbose = verbose)
        
        return output



    else:
        dem_align_output_path,_,_ = hsfm.io.split_file(dem)
        dem_difference_file , aligned_dem_file = hsfm.utils.dem_align_custom(reference_dem,
                                                                             dem,
                                                                             verbose = verbose)
        if generate_ortho:
            ortho_output_path,_,_ = hsfm.io.split_file(dem)
            project_file = ortho_output_path + project_name + ".psx"
            hsfm.metashape.images2ortho(project_file)
        
        output = [bundle_adjusted_metadata_file, 
                  ba_CE90, 
                  ba_LE90, 
                  dem,
                  None, 
                  None, 
                  None, 
                  None]
        
        return output
    
    


    
def metaflow(project_name,
             images_path,
             images_metadata_file,
             reference_dem,
             output_path,
             pixel_pitch             = None,
             focal_length            = None,
             plot_LE90_CE90          = True,
             camera_model_xml_files  = None,
             image_matching_accuracy = 1,
             densecloud_quality      = 2,
             output_DEM_resolution   = None,
             generate_ortho          = False,
             dem_align_all           = True,
             metashape_licence_file  = None,
             verbose                 = False,
             cleanup                 = False,
             check_subsets           = True,
             attempts_to_adjust_cams = 0,
             overwrite               = False):

    if not isinstance(metashape_licence_file, type(None)):
        hsfm.metashape.authentication(metashape_licence_file)
        
    # read from metadata file if not specified
    if not focal_length and not camera_model_xml_files:
        try:
            df_tmp        = pd.read_csv(images_metadata_file)
            focal_lengths = df_tmp['focal_length'].values
            if len(set(focal_lengths)) == 1:
                focal_length = focal_lengths[0]
                print('Focal length:', focal_length)
            else:
                print('Multiple focal lengths provided in metadata csv file.')
                print('hsfm.metashape.images2las will read the focal length')
                print('provided for each camera from the meteadata csv file.')
                pass
        except:
            print('No focal length specified in metadata csv file.')
            pass
    if not pixel_pitch and not camera_model_xml_files:
        try:
            df_tmp        = pd.read_csv(images_metadata_file)
            pixel_pitches = df_tmp['pixel_pitch'].values
            if len(set(pixel_pitches)) == 1:
                pixel_pitch = pixel_pitches[0]
                print('Pixel Pitch:', pixel_pitch)
            else:
                print('Multiple pixel pitches provided in metadata csv file.')
                print('hsfm.metashape.images2las will read the pixel pitch')
                print('provided for each camera from the meteadata csv file.')
                pass
        except:
            print('No pixel pitch specified in metadata csv file.')
            pass
        
    # determine if there are subset clusters of images that do not overlap and/or unaligned images  
    if check_subsets:
        print('Checking for subsets using match points detected with metashape...')
        metashape_project_file, point_cloud_file = hsfm.metashape.images2las(project_name,
                                                                         images_path,
                                                                         images_metadata_file,
                                                                         output_path,
                                                                         focal_length            = focal_length,
                                                                         pixel_pitch             = pixel_pitch,
                                                                         camera_model_xml_files  = camera_model_xml_files,
                                                                         image_matching_accuracy = 2,
                                                                         densecloud_quality      = 4,
                                                                         keypoint_limit          = 80000,
                                                                         tiepoint_limit          = 8000,
                                                                         rotation_enabled        = True,
                                                                         export_point_cloud      = False,
                                                                            overwrite            = overwrite)

        subsets = hsfm.metashape.determine_clusters(metashape_project_file)
        ba_cameras_df, unaligned_cameras_df = hsfm.metashape.update_ba_camera_metadata(metashape_project_file,
                                                                                       images_metadata_file)

        if len(subsets) > 1:
            print('Detected and processing', len(subsets), 'subsets...')
            print(len(subsets), 'image cluster subsets detected')
            image_file_names = list(ba_cameras_df['image_file_name'].values)
            cameras_sub_clusters_dfs = []
            for sub in subsets:
                tmp    = hsfm.core.select_strings_with_sub_strings(image_file_names, sub)
                ## might be better to restart with the original input positions as subset could be displaced
                ## after matching without actually connecting to the other cluster.
                tmp_df = pd.read_csv(images_metadata_file)
                tmp_df = tmp_df[tmp_df['image_file_name'].isin(tmp)].reset_index(drop=True)
#                 tmp_df = ba_cameras_df[ba_cameras_df['image_file_name'].isin(tmp)].reset_index(drop=True)
                cameras_sub_clusters_dfs.append(tmp_df)

            sub_counter = 0
            for sub_df in cameras_sub_clusters_dfs:

                sub_output_path = os.path.join(output_path,'sub_cluster'+str(sub_counter))
                p = pathlib.Path(sub_output_path)
                p.mkdir(parents=True, exist_ok=True)

                sub_images_metadata_file = os.path.join(sub_output_path,'metashape_metadata.csv')
                sub_df.to_csv(sub_images_metadata_file, index = False)

                try:
                    hsfm.batch.metaflow(project_name+'_sub_cluster'+str(sub_counter),
                                    images_path,
                                    sub_images_metadata_file,
                                    reference_dem,
                                    os.path.join(sub_output_path,'metashape'),
                                    pixel_pitch,
                                    focal_length            = focal_length,
                                    plot_LE90_CE90          = plot_LE90_CE90,
                                    camera_model_xml_files  = camera_model_xml_files,
                                    image_matching_accuracy = image_matching_accuracy,
                                    densecloud_quality      = densecloud_quality,
                                    output_DEM_resolution   = output_DEM_resolution,
                                    generate_ortho          = generate_ortho,
                                    dem_align_all           = dem_align_all,
                                    metashape_licence_file  = metashape_licence_file,
                                    verbose                 = verbose,
                                    cleanup                 = cleanup,
                                    check_subsets           = False,
                                    attempts_to_adjust_cams = attempts_to_adjust_cams,
                                       overwrite            = overwrite)
                except:
                    pass

                sub_counter = sub_counter+1
        else:
            # initial run at low res with rotation enabled to ensure DEM does not flip and cameras
            # in roughly correct position before iterative refinement in for loop below.
            out = hsfm.batch.run_metashape(project_name,
                                           images_path,
                                           images_metadata_file,
                                           reference_dem,
                                           output_path,
                                           pixel_pitch,
                                           focal_length            = focal_length,
                                           plot_LE90_CE90          = plot_LE90_CE90,
                                           camera_model_xml_files  = camera_model_xml_files,
                                           image_matching_accuracy = image_matching_accuracy,
                                           densecloud_quality      = densecloud_quality,
                                           output_DEM_resolution   = output_DEM_resolution,
                                           generate_ortho          = generate_ortho,
                                           dem_align_all           = dem_align_all,
                                           rotation_enabled        = True,
                                           metashape_licence_file  = metashape_licence_file,
                                           verbose                 = verbose,
                                           iteration               = 0,
                                           cleanup                 = cleanup,
                                           overwrite            = overwrite)

            bundle_adjusted_metadata_file,\
            ba_CE90,\
            ba_LE90,\
            aligned_dem_file,\
            transform,\
            aligned_bundle_adjusted_metadata_file,\
            tr_ba_CE90,\
            tr_ba_LE90 = out
            
            if attempts_to_adjust_cams > 0:
                for i in np.arange(1,attempts_to_adjust_cams+1,1):
                    if ba_CE90 > 0.01 or ba_LE90 > 0.01:
                        out = hsfm.batch.run_metashape(project_name,
                                                       images_path,
                                                       aligned_bundle_adjusted_metadata_file,
                                                       reference_dem,
                                                       output_path,
                                                       pixel_pitch,
                                                       focal_length            = focal_length,
                                                       plot_LE90_CE90          = plot_LE90_CE90,
                                                       camera_model_xml_files  = camera_model_xml_files,
                                                       image_matching_accuracy = image_matching_accuracy,
                                                       densecloud_quality      = densecloud_quality,
                                                       output_DEM_resolution   = output_DEM_resolution,
                                                       generate_ortho          = generate_ortho,
                                                       dem_align_all           = dem_align_all,
                                                       rotation_enabled        = False,
                                                       metashape_licence_file  = metashape_licence_file,
                                                       verbose                 = verbose,
                                                       iteration               = i,
                                                       cleanup                 = cleanup,
                                                       overwrite            = overwrite)

                        bundle_adjusted_metadata_file,\
                        ba_CE90,\
                        ba_LE90,\
                        aligned_dem_file,\
                        transform,\
                        aligned_bundle_adjusted_metadata_file,\
                        tr_ba_CE90,\
                        tr_ba_LE90 = out
                
        if len(unaligned_cameras_df) > 3:
            # launch seperate metaflow for unaligned images
            print("Processing unaligned cameras seperately...")

            unaligned_output_path          = os.path.join(output_path,
                                                          'unaligned_subset')
            unaligned_images_metadata_file = os.path.join(output_path,
                                                          'unaligned_subset',
                                                          'metashape_metadata.csv')

            p = pathlib.Path(unaligned_output_path)
            p.mkdir(parents=True, exist_ok=True)

            unaligned_cameras_df.to_csv(unaligned_images_metadata_file, index = False)


            hsfm.batch.metaflow(project_name+'_unaligned',
                                images_path,
                                unaligned_images_metadata_file,
                                reference_dem,
                                os.path.join(unaligned_output_path,'metashape'),
                                pixel_pitch,
                                focal_length            = focal_length,
                                plot_LE90_CE90          = plot_LE90_CE90,
                                camera_model_xml_files  = camera_model_xml_files,
                                image_matching_accuracy = image_matching_accuracy,
                                densecloud_quality      = densecloud_quality,
                                output_DEM_resolution   = output_DEM_resolution,
                                generate_ortho          = generate_ortho,
                                dem_align_all           = dem_align_all,
                                metashape_licence_file  = metashape_licence_file,
                                verbose                 = verbose,
                                cleanup                 = cleanup,
                                check_subsets           = check_subsets,
                                attempts_to_adjust_cams = attempts_to_adjust_cams,
                                overwrite            = overwrite)
        if cleanup == True:
            las_files = glob.glob(os.path.join(output_path,'**/*.las'), recursive=True)
            for i in las_files:
                os.remove(i)
                
    else:
        out = hsfm.batch.run_metashape(project_name,
                                       images_path,
                                       images_metadata_file,
                                       reference_dem,
                                       output_path,
                                       pixel_pitch,
                                       focal_length            = focal_length,
                                       plot_LE90_CE90          = plot_LE90_CE90,
                                       camera_model_xml_files  = camera_model_xml_files,
                                       image_matching_accuracy = image_matching_accuracy,
                                       densecloud_quality      = densecloud_quality,
                                       output_DEM_resolution   = output_DEM_resolution,
                                       generate_ortho          = generate_ortho,
                                       dem_align_all           = dem_align_all,
                                       rotation_enabled        = True,
                                       metashape_licence_file  = metashape_licence_file,
                                       verbose                 = verbose,
                                       iteration               = 0,
                                       cleanup                 = cleanup,
                                       overwrite            = overwrite)

        bundle_adjusted_metadata_file,\
        ba_CE90,\
        ba_LE90,\
        aligned_dem_file,\
        transform,\
        aligned_bundle_adjusted_metadata_file,\
        tr_ba_CE90,\
        tr_ba_LE90 = out
        
        if attempts_to_adjust_cams > 0:
            for i in np.arange(1,attempts_to_adjust_cams+1,1):
                if ba_CE90 > 0.01 or ba_LE90 > 0.01:
                    out = hsfm.batch.run_metashape(project_name,
                                                   images_path,
                                                   aligned_bundle_adjusted_metadata_file,
                                                   reference_dem,
                                                   output_path,
                                                   pixel_pitch,
                                                   focal_length            = focal_length,
                                                   plot_LE90_CE90          = plot_LE90_CE90,
                                                   camera_model_xml_files  = camera_model_xml_files,
                                                   image_matching_accuracy = image_matching_accuracy,
                                                   densecloud_quality      = densecloud_quality,
                                                   output_DEM_resolution   = output_DEM_resolution,
                                                   generate_ortho          = generate_ortho,
                                                   dem_align_all           = dem_align_all,
                                                   rotation_enabled        = False,
                                                   metashape_licence_file  = metashape_licence_file,
                                                   verbose                 = verbose,
                                                   iteration               = i,
                                                   cleanup                 = cleanup,
                                                   overwrite            = overwrite)

                    bundle_adjusted_metadata_file,\
                    ba_CE90,\
                    ba_LE90,\
                    aligned_dem_file,\
                    transform,\
                    aligned_bundle_adjusted_metadata_file,\
                    tr_ba_CE90,\
                    tr_ba_LE90 = out

        if cleanup == True:
            las_files = glob.glob(os.path.join(output_path,'**/*.las'), recursive=True)
            for i in las_files:
                os.remove(i)

def batch_process(project_name,
                  reference_dem,
                  input_directory         ='../',
                  pixel_pitch             = None,
                  output_DEM_resolution   = None,
                  generate_ortho          = False,
                  dem_align_all           = True,
                  image_matching_accuracy = 1,
                  densecloud_quality      = 2,
                  metashape_licence_file  = '/opt/metashape-pro/uw_agisoft.lic',
                  verbose                 = True,
                  cleanup                 = True,
                  attempts_to_adjust_cams = 0,
                  check_subsets           = True,
                  overwrite               = False):
    
    output_directory = os.path.join(input_directory, project_name, 'input_data')
    
    if os.path.isdir(output_directory):
        pass
    else:
        print("\nCan't find directory",output_directory)
        sys.exit(0) 
        
    image_files = os.path.join(output_directory,'*','*','*','*cropped_images','*.tif')
    image_files = sorted(glob.glob(image_files))
    
    input_directories = os.path.join(output_directory,'*','*','*','sfm/regional_cl*')
    batches = sorted(glob.glob(input_directories))
    camera_model_xml_files = sorted(glob.glob(os.path.join(output_directory,'camera_models','*.xml')))
    if len(camera_model_xml_files) ==0:
        print('No camera models found in ', os.path.join(output_directory,'camera_models'))
        print('Proceeding without them...')
        camera_model_xml_files = None
    
    if os.path.exists(reference_dem):
        pass
    else:
        print("\nCan't find reference DEM at",output_path)
        sys.exit(0) 
    for i in batches:
        
        try:
            print('\n\n'+i)
    
            now = datetime.now()
    
            cluster_project_name = project_name+'_'+os.path.basename(i)
    
            images_metadata_file = os.path.join(i,'metashape_metadata.csv')
            output_path          = os.path.join(i,'metashape')
            
            if not os.path.isfile(images_metadata_file):
                raise FileNotFoundError(images_metadata_file,' not found.\nSkipping')
    
    
            hsfm.batch.metaflow(cluster_project_name,
                                image_files,
                                images_metadata_file,
                                reference_dem,
                                output_path,
                                pixel_pitch,
                                camera_model_xml_files  = camera_model_xml_files,
                                output_DEM_resolution   = output_DEM_resolution,
                                generate_ortho          = generate_ortho,
                                dem_align_all           = dem_align_all,
                                image_matching_accuracy = image_matching_accuracy,
                                densecloud_quality      = densecloud_quality,
                                metashape_licence_file  = metashape_licence_file,
                                verbose                 = verbose,
                                cleanup                 = cleanup,
                                attempts_to_adjust_cams = attempts_to_adjust_cams,
                                check_subsets           = check_subsets,
                                overwrite               = overwrite)
        except Exception as e:
            print('FAIL:', i,'\n')
            print(traceback.format_exc())

        print("Elapsed time", str(datetime.now() - now), '\n')
        print("DONE")

