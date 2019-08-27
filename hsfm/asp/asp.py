from osgeo import gdal
import os
import glob
import utm

import hsfm.io
import hsfm.core
import hsfm.utils


def generate_cam_gem_corner_coordinates_string(corners_gdf):
    for n,p in enumerate(corner_points.geometry):
        lon_c = corner_points.loc[n].geometry.x
        lat_c = corner_points.loc[n].geometry.y
        corner_points_xy.append(str(lon_c))
        corner_points_xy.append(str(lat_c))
    corner_points_xy = ','.join(corner_points_xy)
    return corner_points_xy

def calculate_corner_coordinates(camera_lat_lon_wgs84_center_coordinates,
                                 focal_length_mm,
                                 image_width_px,
                                 image_height_px,
                                 heading):
                                 
    # TODO
    # - Investigate why the order of UL, UR, LR, LL does not match the way
    #   cam_gen traverses the iamge 0,0 w,0, w,h, 0,h
    
    # This assumes the principal point is at the image center 
    # i.e. half the image width and height                             
    half_width_m, half_height_m = hsfm.core.calculate_distance_principal_point_to_image_edge(focal_length_mm,
                                                                                         image_width_px,
                                                                                         image_height_px)
    
    # Convert camera center coordinates to utm
    u = utm.from_latlon(camera_lat_lon_wgs84_center_coordinates[0], camera_lat_lon_wgs84_center_coordinates[1])
    
    camera_utm_lat = u[1]
    camera_utm_lon = u[0]
    # Calculate upper left, upper right, lower right, lower left corner coordinates as (lat,lon)
    UL, UR, LR, LL = hsfm.trig.calculate_corner(camera_utm_lat,camera_utm_lon,half_width_m, half_height_m, heading)

    # Calculate corner coordinates in UTM
    # corners = [UL, UR, LR, LL] # this should be right
    corners = [LR, UR, UL, LL]
    corner_points_wgs84 = []
    for coordinate in corners:
        coordinate_wgs84 = utm.to_latlon(coordinate[0],coordinate[1],u[2],u[3])
        lat = coordinate_wgs84[0]
        lon = coordinate_wgs84[1]
        corner_points_wgs84.append(str(lon))
        corner_points_wgs84.append(str(lat))
    corner_coordinates_string = ','.join(corner_points_wgs84)
    
    return corner_coordinates_string
    
def generate_camera(image_file_name,
                    camera_lat_lon_center_coordinates,
                    reference_dem,
                    focal_length_mm,
                    heading,
                    out_dir = './data/cameras',
                    pixel_pitch=0.02,
                    scale = 1,
                    verbose=True):
    
    # Get the image base name to name the output camera
    image_base_name = os.path.splitext(os.path.split(image_file_name)[-1])[0]
    
    # Read in the image and get the dimensions and principal point at image center
    img_ds = gdal.Open(image_file_name)
    image_width_px = img_ds.RasterXSize
    image_height_px = img_ds.RasterYSize
    principal_point_px = (image_width_px / 2, image_height_px /2 )
    
    # Calculate the focal length in pixel coordinates
    focal_length_px = focal_length_mm / pixel_pitch
    
    # Calculate corner coordinates string
    corner_coordinates = calculate_corner_coordinates(camera_lat_lon_center_coordinates,
                                                      focal_length_mm,
                                                      image_width_px,
                                                      image_height_px,
                                                      heading)
    out = os.path.join(out_dir,image_base_name+'.tsai')
    
    call =[
        'cam_gen', image_file_name,
        '--reference-dem', reference_dem,
        '--focal-length', str(focal_length_px),
        '--optical-center', str(principal_point_px[0]), str(principal_point_px[1]),
        '--pixel-pitch', str(scale),
        '--refine-camera',
        '-o', out,
        '--lon-lat-values',corner_coordinates
    ]
    
    print(*call)
    
    hsfm.utils.run_command(call, verbose=verbose)
    
    return out

def bundle_adjust_custom(image_files_directory, 
                         camera_files_directory, 
                         output_directory_prefix):
    
    input_image_files  = sorted(glob.glob(os.path.join(image_files_directory,'*.tif')))
    input_camera_files  = sorted(glob.glob(os.path.join(camera_files_directory,'*.tsai')))
    
    ba_dir = os.path.split(output_directory_prefix)[0]
    
    log_directory = os.path.join(ba_dir,'log')
    hsfm.io.create_dir(log_directory)
    
    call =['bundle_adjust',
           '--threads', '1',
           '--disable-tri-ip-filter',
           '--force-reuse-match-files',
           '--skip-rough-homography',
           '-t', 'nadirpinhole',
           '--ip-inlier-factor', '1',
           '--ip-uniqueness-threshold', '0.9',
           '--ip-per-tile','4000',
           '--datum', 'wgs84',
           '--inline-adjustments',
           '--camera-weight', '0.0',
           '--num-iterations', '500',
           '--num-passes', '3']
           
    call.extend(input_image_files)
    call.extend(input_camera_files)
    call.extend(['-o', output_directory_prefix])
    
    hsfm.utils.run_command(call, 
                           verbose=False, 
                           log_directory=log_directory)
                           
    print('Bundle adjust results saved in', ba_dir)
    return ba_dir


def parallel_stereo_custom(first_image, 
                           second_image,
                           first_camera,
                           second_camera, 
                           stereo_output_directory_prefix):
    

    stereo_output_directory = os.path.split(stereo_output_directory_prefix)[0]
    
    log_directory = os.path.join(stereo_output_directory,'log')
    hsfm.io.create_dir(log_directory)
    
    call =['parallel_stereo',
           '--force-reuse-match-files',
           '--stereo-algorithm', '2',
           '-t', 'nadirpinhole',
           '--skip-rough-homography',
           '--ip-inlier-factor', '1',
           '--ip-per-tile','2000',
           '--ip-uniqueness-threshold', '0.9']
           
    call.extend([first_image,second_image])
    call.extend([first_camera,second_camera])
    call.extend([stereo_output_directory_prefix])
    
    hsfm.utils.run_command(call, 
                           verbose=False, 
                           log_directory=log_directory)
                           
    print('Bundle adjust results saved in', stereo_output_directory)
    return stereo_output_directory