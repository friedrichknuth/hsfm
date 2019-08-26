from osgeo import gdal
import os
import utm

import hsfm.core
import hsfm.io

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
    print(camera_utm_lat,camera_utm_lon,half_width_m, half_height_m, heading)
    UL, UR, LR, LL = hsfm.trig.calculate_corner(camera_utm_lat,camera_utm_lon,half_width_m, half_height_m, heading)

    # Calculate corner coordinates in UTM
    corners = [UL, UR, LR, LL]
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
    
    hsfm.utils.run_command(call, verbose=verbose)
    
    return out

