from osgeo import gdal
import os
import cv2
import sys
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


import hsfm.io
import hsfm.core
import hsfm.image
import hsfm.plot
import hsfm.utils
import hsfm.metashape

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


def calculate_heading_from_metadata(df,
                                    subset            = None,
                                    reverse_order     = False,
                                    output_directory  = None,
                                    for_metashape     = False,
                                    reference_dem     = None,
                                    flight_altitude_m = 1500,
                                    sorting_column = 'fileName'):
    # TODO
    # - Add flightline seperation function
    # - Generalize beyond NAGAP keys
    if subset:
        df = hsfm.core.subset_images_for_download(df, subset)
        
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
            
    df = df.sort_values(by=[sorting_column], ascending=True)   
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
                 
        if output_directory:
            hsfm.io.create_dir(output_directory)
            df.to_csv(os.path.join(output_directory,'metashape_metadata.csv'),index=False)
        
        return df
    
    else:
        return df

def download_images_to_disk(camera_positions_file_name, 
                            subset=None, 
                            output_directory='output_data/raw_images',
                            image_type='pid_tiff'):
                            
    hsfm.io.create_dir(output_directory)
    df = pd.read_csv(camera_positions_file_name)
    
    if not isinstance(subset,type(None)):
        df = hsfm.core.subset_images_for_download(df, subset)
    
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
                      output_directory='input_data/images',
                      subset=None, 
                      scale=None,
                      qc=False,
                      invisible_fiducial=None,
                      crop_from_pp_dist = 11250,
                      manually_pick_fiducials=False,
                      side = None):
                      
    """
    Function to preprocess images from NAGAP archive in batch.
    
    side : 'left', 'right', top', 'bottom' # side opposite flight direction in image
    """
    # TODO
    # - Make io faster with gdal
    # - Generalize for other types of images
    # - Add affine transformation (if needed)
                      
    hsfm.io.create_dir(output_directory)
    
    templates = hsfm.core.gather_templates(template_directory)         
                      
    intersections =[]
    file_names = []
    
    if not isinstance(camera_positions_file_name,type(None)):
        df = pd.read_csv(camera_positions_file_name)
        df = hsfm.core.subset_images_for_download(df, subset)
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
                                                            crop_from_pp_dist=crop_from_pp_dist,
                                                            manually_pick_fiducials=manually_pick_fiducials,
                                                            side = side)
            intersections.append(intersection_angle)
            file_names.append(file_name)
    
    elif not isinstance(image_directory,type(None)):
        image_files = sorted(glob.glob(os.path.join(image_directory,'*.tif')))
        for image_file in image_files:
            file_path, file_name, file_extension = hsfm.io.split_file(image_file)
            print('Processing',file_name)
            img_gray = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            
            intersection_angle = hsfm.core.preprocess_image(img_gray, 
                                                            file_name,
                                                            templates, 
                                                            qc=qc,
                                                            output_directory=output_directory,
                                                            invisible_fiducial=invisible_fiducial,
                                                            crop_from_pp_dist=crop_from_pp_dist,
                                                            manually_pick_fiducials=manually_pick_fiducials,
                                                            side = side)
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
                  focal_length,
                  pixel_pitch,
                  output_dem_resolution   = 0.5,
                  image_matching_accuracy = 1,
                  densecloud_quality      = 1,
                  rotation_enabled        = False,
                  generate_ortho          = False,
                  metashape_licence_file  = None,
                  verbose                 = False,
                  iteration               = 0):
    
    # TODO auto determine optimal output_dem_resolution from point cloud density
    # TODO plot LE90 CE90 as qc
    
    now = datetime.now()
    
    output_path = output_path.rstrip('/') + str(iteration)
    bundle_adjusted_metadata_file = os.path.join(output_path, project_name + "_bundle_adjusted_metadata.csv")
    aligned_bundle_adjusted_metadata_file = os.path.join(output_path, project_name + "_aligned_bundle_adjusted_metadata.csv")
    
    
    if not isinstance(metashape_licence_file, type(None)):
        hsfm.metashape.authentication(metashape_licence_file)
        
    out = hsfm.metashape.images2las(project_name,
                                    images_path,
                                    images_metadata_file,
                                    output_path,
                                    focal_length            = focal_length,
                                    pixel_pitch             = pixel_pitch,
                                    image_matching_accuracy = image_matching_accuracy,
                                    densecloud_quality      = densecloud_quality,
                                    rotation_enabled        = rotation_enabled)
    
    metashape_project_file, point_cloud_file = out
    
    
    hsfm.metashape.update_ba_camera_metadata(metashape_project_file,
                                             images_metadata_file,
                                             output_file_name=bundle_adjusted_metadata_file)
    
    x_offset, y_offset, z_offset = hsfm.core.compute_point_offsets(images_metadata_file, 
                                                                    bundle_adjusted_metadata_file)
    

    ba_CE90, ba_LE90 = hsfm.geospatial.CE90(x_offset,y_offset), hsfm.geospatial.LE90(z_offset)
    hsfm.plot.plot_offsets(ba_LE90,
                           ba_CE90,
                           x_offset, 
                           y_offset, 
                           z_offset,
                           title = 'Initial vs Bundle Adjusted',
                           plot_file_name = os.path.join(output_path, 'qc_ba_ce90le90.png'))
    
    if ba_CE90 < 0.01 and ba_LE90 < 0.01:
        if generate_ortho:
            hsfm.metashape.las2dem(project_name,
                                   output_path)

            hsfm.metashape.images2ortho(project_name,
                                        output_path)
    
    
    epsg_code = 'EPSG:'+ hsfm.geospatial.get_epsg_code(reference_dem)
    dem = hsfm.asp.point2dem(point_cloud_file,
                             '--nodata-value','-9999',
                             '--tr',str(output_dem_resolution),
                             '--threads', '10',
                             '--t_srs', epsg_code,
                             verbose=verbose)
    
    clipped_reference_dem = os.path.join(output_path,'reference_dem_clip.tif')
    
    large_to_small_order = hsfm.geospatial.compare_dem_extent(dem, reference_dem)
    if large_to_small_order == (reference_dem, dem):
        reference_dem = hsfm.utils.clip_reference_dem(dem,
                                                      reference_dem,
                                                      output_file_name = clipped_reference_dem,
                                                      buff_size        = 2000,
                                                      verbose = verbose)
    
#     if ba_CE90 < 0.01 and ba_LE90 < 0.01:
#         hsfm.utils.dem_align_custom(reference_dem,
#                                     dem,
#                                     output_path,
#                                     verbose = verbose)

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
    
    hsfm.plot.plot_offsets(tr_ba_LE90,
                           tr_ba_CE90,
                           x_offset, 
                           y_offset, 
                           z_offset,
                           title = 'Bundle Adjusted vs Transformed',
                           plot_file_name = os.path.join(output_path, 'qc_tr_ba_ce90le90.png'))
    
    hsfm.utils.dem_align_custom(reference_dem,
                            aligned_dem_file,
                            output_path,
                            verbose = verbose)
    
#     if tr_ba_CE90 > 0.01 and tr_ba_LE90 > 0.01:
#         hsfm.utils.dem_align_custom(reference_dem,
#                                 aligned_dem_file,
#                                 output_path,
#                                 verbose = verbose)
    
    output = [bundle_adjusted_metadata_file, 
              ba_CE90, 
              ba_LE90, 
              aligned_dem_file,
              transform, 
              aligned_bundle_adjusted_metadata_file, 
              tr_ba_CE90, 
              tr_ba_LE90]
    
    return output


    
def metaflow(project_name,
             images_path,
             images_metadata_file,
             reference_dem,
             output_path,
             focal_length,
             pixel_pitch,
             image_matching_accuracy = 1,
             densecloud_quality      = 1,
             output_dem_resolution   = 0.5,
             metashape_licence_file  = None,
             verbose                 = False,
             cleanup                 = False):
    
    out = hsfm.batch.run_metashape(project_name,
                                   images_path,
                                   images_metadata_file,
                                   reference_dem,
                                   output_path,
                                   focal_length,
                                   pixel_pitch,
                                   output_dem_resolution   = output_dem_resolution,
                                   image_matching_accuracy = image_matching_accuracy,
                                   densecloud_quality      = densecloud_quality,
                                   rotation_enabled        = True,
                                   generate_ortho          = False,
                                   metashape_licence_file  = metashape_licence_file,
                                   verbose                 = verbose,
                                   iteration               = 0)
    
    bundle_adjusted_metadata_file,\
    ba_CE90,\
    ba_LE90,\
    aligned_dem_file,\
    transform,\
    aligned_bundle_adjusted_metadata_file,\
    tr_ba_CE90,\
    tr_ba_LE90 = out
    
    for i in np.arange(1,4,1):
        if ba_CE90 > 0.01 or ba_LE90 > 0.01:
            out = hsfm.batch.run_metashape(project_name,
                                           images_path,
                                           aligned_bundle_adjusted_metadata_file,
                                           reference_dem,
                                           output_path,
                                           focal_length,
                                           pixel_pitch,
                                           output_dem_resolution   = output_dem_resolution,
                                           image_matching_accuracy = image_matching_accuracy,
                                           densecloud_quality      = densecloud_quality,
                                           rotation_enabled        = False,
                                           generate_ortho          = False,
                                           metashape_licence_file  = metashape_licence_file,
                                           verbose                 = verbose,
                                           iteration               = i)
            
            bundle_adjusted_metadata_file,\
            ba_CE90,\
            ba_LE90,\
            aligned_dem_file,\
            transform,\
            aligned_bundle_adjusted_metadata_file,\
            tr_ba_CE90,\
            tr_ba_LE90 = out
            
#     out = hsfm.batch.run_metashape(project_name,
#                                    images_path,
#                                    bundle_adjusted_metadata_file,
#                                    reference_dem,
#                                    output_path,
#                                    focal_length,
#                                    pixel_pitch,
#                                    output_dem_resolution   = output_dem_resolution,
#                                    image_matching_accuracy = image_matching_accuracy,
#                                    densecloud_quality      = densecloud_quality,
#                                    rotation_enabled        = True,
#                                    generate_ortho          = False,
#                                    metashape_licence_file  = metashape_licence_file,
#                                    verbose                 = verbose,
#                                    iteration               = '_final')
            
    if cleanup == True:
        las_files = glob.glob('./**/*.las', recursive=True)
        for i in las_files:
            os.remove(i)
    
    
    
    