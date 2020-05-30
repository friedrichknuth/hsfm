import shutil
import numpy as np
import pandas as pd
from urllib.request import urlopen
import cv2
from skimage import exposure
import glob
import os
from osgeo import gdal
import utm
import itertools

import hsfm.image
import hsfm.utils
import hsfm.plot
import hsfm.geospatial
import bare

"""
Core data wrangling and preprocessing functions. 
"""

# TODO
# - break this up into seperate libraries and classes to better
#   accomodate other imagery and generealize upstream as much as possible.
    
def get_gcp_polygon(fn):
    
    file_name = os.path.splitext(os.path.split(fn)[-1])[0]
    
    df = pd.read_csv(fn, header=None, sep=' ')
    df = df[[1,2]]
    df.columns=['lat','lon']
    
    gdf = hsfm.geospatial.df_points_to_polygon_gdf(df)
    gdf['camera'] = file_name
    return gdf

def create_overlap_list(gcp_directory,
                        image_directory,
                        output_directory):
    
    output_directory = os.path.join(output_directory, 'ba')
    
    hsfm.io.create_dir(output_directory)
    
    
    filename_out = os.path.join(output_directory,'overlaplist.txt')
    if os.path.exists(filename_out):
        os.remove(filename_out)
    
    gcp_files = glob.glob(os.path.join(gcp_directory,'*.gcp'))
    image_files = glob.glob(os.path.join(image_directory,'*.tif'))
    
    footprints = []
    for fn in gcp_files:
        gdf = get_gcp_polygon(fn)
        footprints.append(gdf)
        
    pairs=[]
    for a, b in itertools.combinations(footprints, 2):
        result = hsfm.geospatial.compare_footprints(a, b)
        if result == 1:
            c = hsfm.io.retrieve_match(a['camera'].values[0] , image_files)
            d = hsfm.io.retrieve_match(b['camera'].values[0] , image_files)
            pairs.append((c,d))

    pairs = sorted(list(set(pairs)))
    for i in pairs:
        with open(filename_out, 'a') as out:
            out.write(i[0] + ' '+ i[1]+'\n')
    return filename_out


def create_overlap_list_from_match_files(match_files_directory,
                                         image_directory,
                                         output_directory,
                                         suffix='.match'):
    
    output_directory = os.path.join(output_directory, 'ba')
    
    hsfm.io.create_dir(output_directory)
    
    
    filename_out = os.path.join(output_directory,'overlaplist.txt')
    if os.path.exists(filename_out):
        os.remove(filename_out)
    
    match_files = sorted(glob.glob(os.path.join(match_files_directory, '*' + suffix)))
    
    match_files
    pairs = []
    for match_file in match_files:
        img1_fn, img2_fn = bare.core.parse_image_names_from_match_file_name(match_file, 
                                                                            image_directory, 
                                                                            'tif')
        pairs.append((img1_fn, img2_fn))
        
    # creates full set from .match and clean.match pairs
    pairs = sorted(list(set(pairs)))
    for i in pairs:
        with open(filename_out, 'a') as out:
            out.write(i[0] + ' '+ i[1]+'\n')
            
    return filename_out

def determine_flight_lines(df, cutoff_angle = 30):
    
    df = hsfm.batch.calculate_heading_from_metadata(df)
    
    df['next_heading'] = df['heading'].shift(-1)
    df['heading_diff'] = abs(df['next_heading'] - df['heading'])
    df['heading_diff'] = df['heading_diff'].fillna(0)
    df = df.reset_index()
    
    flights_tmp = []
    tmp_df = pd.DataFrame()
    
    for row in df.iterrows():
        if row[1]['heading_diff'] < cutoff_angle:
            tmp_df = pd.concat([tmp_df, pd.DataFrame(row[1])],axis=1)
        else:
            tmp_df = pd.concat([tmp_df, pd.DataFrame(row[1])],axis=1)
            tmp_df = tmp_df.T
            flights_tmp.append(tmp_df)
            tmp_df = pd.DataFrame()
    tmp_df = tmp_df.T
    if not tmp_df.empty:
        flights_tmp.append(tmp_df)

    flights = []
    for i,v in enumerate(flights_tmp):
        if len(v) == 1:
            tmp_df = pd.concat([flights_tmp[i-1],v])
            flights.pop()
            flights.append(tmp_df)
        else:
            flights.append(v)
    
    return flights
    
    
def evaluate_image_frame(grayscale_unit8_image_array,frame_size=0.07):
    
    x = grayscale_unit8_image_array.shape[1]
    y = grayscale_unit8_image_array.shape[0]
    
    img = grayscale_unit8_image_array
    
    window = [0,y,0,x]
    
    slice_left_top = frame_size
    slice_right_bottom = 1-frame_size
    
    x_slice_left = int(x * slice_left_top)
    x_slice_right = int(x * slice_right_bottom)
    y_slice_top = int(y * slice_left_top)
    y_slice_bottom = int(y * slice_right_bottom)
    
    left =   img[0:y,              0:x_slice_left]
    top =    img[0:y_slice_top,    0:x]
    right =  img[0:y,              x_slice_right:x] 
    bottom = img[y_slice_bottom:y, 0:x]
    
    stats = {'left':np.median(left), 
             'right':np.median(right), 
             'top': np.median(top), 
             'bottom':np.median(bottom)}

    side = min(stats, key=lambda key: stats[key])
    
    return side

def calculate_corner_coordinates(camera_lat_lon_wgs84_center_coordinates,
                                 reference_dem,
                                 focal_length_mm,
                                 image_width_px,
                                 image_height_px,
                                 heading):

    # This assumes the principal point is at the image center 
    # i.e. half the image width and height                             
    half_width_m, half_height_m = calculate_distance_principal_point_to_image_edge(focal_length_mm,
                                                                                   image_width_px,
                                                                                   image_height_px)
    
    # Convert camera center coordinates to utm
    u = utm.from_latlon(camera_lat_lon_wgs84_center_coordinates[0], camera_lat_lon_wgs84_center_coordinates[1])
    camera_utm_lat = u[1]
    camera_utm_lon = u[0]
    
    # Calculate upper left, upper right, lower right, lower left corner coordinates as (lat,lon)
    UL, UR, LR, LL = hsfm.trig.calculate_corner(camera_utm_lat,camera_utm_lon,half_width_m, half_height_m, heading)

    #Calculate corner coordinates in UTM
    corners = [UL, UR, LR, LL]
    corner_points_wgs84 = []
    corner_lons = []
    corner_lats = []
    
    for coordinate in corners:
        coordinate_wgs84 = utm.to_latlon(coordinate[0],coordinate[1],u[2],u[3])
        lat = coordinate_wgs84[0]
        lon = coordinate_wgs84[1]
        corner_lons.append(lon)
        corner_lats.append(lat)
        
    corner_elevations = hsfm.geospatial.sample_dem(corner_lons, corner_lats, reference_dem)
    
    return corner_lons, corner_lats, corner_elevations
    
def prep_and_generate_gcp(image_file_name,
                          camera_lat_lon_center_coordinates,
                          reference_dem,
                          focal_length_mm,
                          heading,
                          output_directory,
                          pixel_pitch_mm=0.02):
    
    # Get the image base name to name the output camera
    image_base_name = os.path.splitext(os.path.split(image_file_name)[-1])[0]
    
    # Read in the image and get the dimensions and principal point at image center
    img_ds = gdal.Open(image_file_name)
    image_width_px = img_ds.RasterXSize
    image_height_px = img_ds.RasterYSize
    principal_point_px = (image_width_px/2, image_height_px/2)
    
    # Calculate corner coordinates and elevations
    corner_lons, corner_lats, corner_elevations = calculate_corner_coordinates(camera_lat_lon_center_coordinates,
                                                                               reference_dem,
                                                                               focal_length_mm,
                                                                               image_width_px,
                                                                               image_height_px,
                                                                               heading)
    output_directory = generate_gcp(corner_lons,
                                    corner_lats,
                                    corner_elevations,
                                    image_file_name,
                                    image_width_px,
                                    image_height_px,
                                    output_directory=output_directory)
    
    return output_directory
                                      
                                      
def generate_gcp(corner_lons, 
                 corner_lats, 
                 corner_elevations, 
                 image_file_name,
                 image_width_px,
                 image_height_px,
                 output_directory):
    
    output_directory = os.path.join(output_directory, 'gcp')
    
    # TODO
    # - add seperate rescale function based on image subsample factor
    
    hsfm.io.create_dir(output_directory)
    
    file_path, file_name, file_extension = hsfm.io.split_file(image_file_name)
    
    df = pd.DataFrame()
    df['lat'] = corner_lats
    df['lon'] = corner_lons
    df['ground_elevation'] = corner_elevations
    df['sigmas1'] = 1
    df['sigmas2'] = 1
    df['sigmas3'] = 1
    df['image'] = image_file_name
    df['corners1'] = [0,image_width_px,image_width_px,0]
    df['corners2'] = [0,0,image_height_px,image_height_px]
    df['sigmas4'] = 1
    df['sigmas5'] = 1
    
    out = os.path.join(output_directory,file_name+'.gcp')
    
    df.to_csv(out, sep=' ', header=False)
    
    return output_directory

def initialize_cameras(camera_positions_file_name, 
                       reference_dem_file_name,
                       focal_length_px,
                       principal_point_px,
                       output_directory,
                       subset=None,
                       altitude=1500):
    
    # TODO
    # - integrate elevation interpolation function to handle no data values
    # - get raster crs and convert points to crs of input raster before interpolation
    
    output_directory = os.path.join(output_directory, 'initial_cameras')
    hsfm.io.create_dir(output_directory)
    
    df = hsfm.core.select_images_for_download(camera_positions_file_name)
    lons = df['Longitude'].values
    lats = df['Latitude'].values
    elevations = hsfm.geospatial.sample_dem(lons,lats, reference_dem_file_name)
    df['elevation'] = elevations 
    df['elevation'] = df['elevation'] + altitude
    df['elevation'] = df['elevation'].max()
    gdf = hsfm.geospatial.df_xyz_coords_to_gdf(df,lon='Longitude',lat='Latitude')
    
    gdf = gdf.to_crs({'init':'epsg:4978'})
    gdf = hsfm.geospatial.extract_gpd_geometry(gdf)
    
    for index, row in gdf.iterrows():
        image_base_name = row['fileName']
        out = os.path.join(output_directory,image_base_name+'.tsai')
        with open(out, 'w') as f:

            C0 = str(row['x'])
            C1 = str(row['y'])
            C2 = str(row['z'])

            line0 = 'VERSION_4\n'
            line1 = 'PINHOLE\n'
            line2 = 'fu = ' + str(focal_length_px) +'\n'
            line3 = 'fv = ' + str(focal_length_px) +'\n'
            line4 = 'cu = ' + str(principal_point_px[0]) +'\n'
            line5 = 'cv = ' + str(principal_point_px[1]) +'\n'
            line6 = 'u_direction = 1 0 0\n'
            line7 = 'v_direction = 0 1 0\n'
            line8 = 'w_direction = 0 0 1\n'
            line9 = ' '.join(['C =',C0,C1,C2,'\n'])
            line10 = 'R = 1 0 0 0 1 0 0 0 1\n'
            line11 = 'pitch = 1\n'
            line12 = 'NULL\n'

            f.writelines([line0,line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11,line12])
    return output_directory
            
def rotate_camera(cropped_grayscale_unit8_image_array, side=None):

    img = cropped_grayscale_unit8_image_array

    if side == 'left':
        # rotate image 90 degrees counter clockwise
        img = np.rot90(img)
    
    elif side == 'top':
        # rotate image 180 degrees counter clockwise
        img = np.rot90(img)
        img = np.rot90(img)
    
    elif side == 'right':
        # rotate image 270 degrees counter clockwise
        img = np.rot90(img)
        img = np.rot90(img)
        img = np.rot90(img)
    
    return img
    
def subset_input_image_list(image_list, subset=None):
    
    """
    Function to subset list of images. 
    
    Use tuple to specify a range of image suffixes.
    Use list to specify a list of image suffixes.
    """
    if subset == None:
        return image_list
    
    else:
        image_list_df = pd.DataFrame(image_list,columns=['image_file_path'])
        image_list_tmp = []
        for image_file in image_list:
            image_list_tmp.append(os.path.splitext(os.path.split(image_file)[-1])[0])
        image_list_df['image_name'] = image_list_tmp
        image_list_df['image_index_number'] = image_list_df['image_name'].str[11:].apply(int)
        
        if isinstance(subset, tuple):
            image_list_df = image_list_df[image_list_df['image_index_number'].between(subset[0],subset[1])]
        elif isinstance(subset, list):
            image_list_df = image_list_df[image_list_df['image_index_number'].isin(subset)]
        
        subset_image_list = image_list_df['image_file_path'].to_list()
        return subset_image_list
    
def pre_select_target_images(input_csv, prefix, image_suffix_list,output_directory=None):
    df = pd.read_csv(input_csv)
    df = df[df['fileName'].str.contains(prefix)]
    image_suffix_list = list(map(str, image_suffix_list))
    image_suffix_list = [x.rjust(3,'0') for x in image_suffix_list]
    df = df[df['fileName'].str.endswith(tuple(image_suffix_list), na=False)]
    if output_directory:
        hsfm.io.create_dir(output_directory)
        output_file_name = os.path.join(output_directory, 'targets.csv')
        df.to_csv(output_file_name,index=False)
    else:
        output_file_name = 'input_data/targets.csv'
        df.to_csv(output_file_name,index=False)
    return output_file_name
    
    
def subset_images_for_download(df, subset=None):
    
    """
    Function to convert input metadata csv to dataframe.
    
    Use tuple to specify a range of image suffixes.
    Use list to specify a list of image suffixes.
    """
    # TODO
    # - Add option to subset with list if range not suitable
    df = df.drop_duplicates()
    if subset == None:
        return df
    else:
        df['image_index_number'] = df['fileName'].str[11:].apply(int)
        if isinstance(subset, tuple):
            df = df[df['image_index_number'].between(subset[0],subset[1])]
        elif isinstance(subset, list):
            df = df[df['image_index_number'].isin(subset)]
    return df
    
def download_image(pid):
    base_url = 'https://arcticdata.io/metacat/d1/mn/v2/object/'
    url = base_url+pid
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    return image
    
def slice_image_frame(grayscale_unit8_image_array, windows):
    window_left   = windows[0]
    window_top    = windows[1]
    window_right  = windows[2]
    window_bottom = windows[3]
    
    img_gray = grayscale_unit8_image_array
    
    left_slice = img_gray[window_left[0]:window_left[1],window_left[2]:window_left[3]]
    top_slice = img_gray[window_top[0]:window_top[1],window_top[2]:window_top[3]]
    right_slice = img_gray[window_right[0]:window_right[1],window_right[2]:window_right[3]]
    bottom_slice = img_gray[window_bottom[0]:window_bottom[1],window_bottom[2]:window_bottom[3]]
    
    slices = [left_slice, top_slice, right_slice, bottom_slice]
    return slices

def gather_templates(template_directory):
    left_template = os.path.join(template_directory,'L.jpg')
    top_template = os.path.join(template_directory,'T.jpg')
    right_template = os.path.join(template_directory,'R.jpg')
    bottom_template = os.path.join(template_directory,'B.jpg')
    templates = [left_template, top_template, right_template, bottom_template]
    return templates
    
def pick_fiducials_manually(image_file_name=None, image_array=None, qc=False, output_directory='data/images'):
    if isinstance(image_array, np.ndarray):
        hsfm.io.create_dir('tmp/')
        temp_out = os.path.join('tmp/', 'temporary_image.tif')
        cv2.imwrite(temp_out,image_array)
        temp_out_optimized = hsfm.utils.optimize_geotif(temp_out)
        os.remove(temp_out)
        os.rename(temp_out_optimized, temp_out)
        
        image_file_name = temp_out
          
    condition = True
    while condition == True:
        # currently only works from file on disk
        principal_point, intersection_angle, fiducials = hsfm.utils.pick_fiducials(image_file_name)
        if intersection_angle > 90.1 or intersection_angle < 89.9:
            print('Intersection angle at principle point is not within orthogonality limits.')
            print('Try again.')
        else:
            if isinstance(image_array, np.ndarray):
                shutil.rmtree('tmp/')
            return principal_point, intersection_angle, fiducials
            condition = False 
    
    
def preprocess_image(image_array, 
                     file_name, 
                     templates, 
                     qc=False, 
                     output_directory='data/images', 
                     image_file_name=None,
                     invisible_fiducial=None,
                     crop_from_pp_dist=11250,
                     manually_pick_fiducials=False,
                     side = None):
    """
    side = 'left','top','right' #Determines position of frame opposite to flight direction. 
                                #If none determines side of frame with largest black border.
    """
                     
    # TODO clean this up
    
    img_gray = image_array
    
    window_left = [4000,8000,0,2000]
    window_top = [0,1000,4000,8000]
    window_right = [4000,8000,img_gray.shape[1]-1000,img_gray.shape[1]]
    window_bottom = [img_gray.shape[0]-1000,img_gray.shape[0],4000,8000]
    windows = [window_left, window_top, window_right, window_bottom]
    
    if isinstance(side, type(None)):
        side = hsfm.core.evaluate_image_frame(img_gray)
    
    if manually_pick_fiducials:
        if image_file_name:
            principal_point, intersection_angle, fiducials = pick_fiducials_manually(image_file_name=image_file_name)
        else:
            principal_point, intersection_angle, fiducials = pick_fiducials_manually(image_array=img_gray)
    
    else:
        fiducials, principal_point = detect_fiducials_and_principal_point(windows, 
                                                                          templates, 
                                                                          img_gray,
                                                                          invisible_fiducial=invisible_fiducial)
        # QC routine
        intersection_angle = determine_intersection_angle(fiducials)
        print('Principal point intersection angle:', intersection_angle)
    
        if intersection_angle > 90.1 or intersection_angle < 89.9:
            print("Warning: intersection angle at principle point is not within orthogonality limits.")
            print('Re-attempting fiducial marker detection.')
            print("Processing left fiducial.")
            fiducials, principal_point = detect_fiducials_and_principal_point(windows, 
                                                                              templates, 
                                                                              img_gray,
                                                                              noisify='left',
                                                                              invisible_fiducial=invisible_fiducial)
            intersection_angle = determine_intersection_angle(fiducials)
            print('New intersection angle:',intersection_angle)
            if intersection_angle > 90.1 or intersection_angle < 89.9:
                print("Processing top fiducial.")
                fiducials, principal_point = detect_fiducials_and_principal_point(windows, 
                                                                                  templates, 
                                                                                  img_gray,
                                                                                  noisify='top',
                                                                                  invisible_fiducial=invisible_fiducial)
                intersection_angle = determine_intersection_angle(fiducials)
                print('New intersection angle:',intersection_angle)
                if intersection_angle > 90.1 or intersection_angle < 89.9:
                    print("Processing right fiducial.")
                    fiducials, principal_point = detect_fiducials_and_principal_point(windows, 
                                                                                      templates, 
                                                                                      img_gray,
                                                                                      noisify='right',
                                                                                      invisible_fiducial=invisible_fiducial)
                    intersection_angle = determine_intersection_angle(fiducials)
                    print('New intersection angle:',intersection_angle)
                    if intersection_angle > 90.1 or intersection_angle < 89.9:
                        print("Processing bottom fiducial.")
                        fiducials, principal_point = detect_fiducials_and_principal_point(windows, 
                                                                                          templates, 
                                                                                          img_gray,
                                                                                          noisify='bottom',
                                                                                          invisible_fiducial=invisible_fiducial)
                        intersection_angle = determine_intersection_angle(fiducials)
                        print('New intersection angle:',intersection_angle)
                    
    if intersection_angle > 90.1 or intersection_angle < 89.9:
        print("Unable to improve result for", file_name)
        print("Please select fiducial markers manually")
        if image_file_name:
            principal_point, intersection_angle, fiducials = pick_fiducials_manually(image_file_name=image_file_name)
        else:
            principal_point, intersection_angle, fiducials = pick_fiducials_manually(image_array=img_gray)
        
    if intersection_angle < 90.1 and intersection_angle > 89.9:
        cropped = crop_about_principal_point(img_gray, 
                                             principal_point,
                                             crop_from_pp_dist = crop_from_pp_dist)
        img_rot = hsfm.core.rotate_camera(cropped, side=side)
        out = os.path.join(output_directory, file_name+'.tif')
        cv2.imwrite(out,img_rot)
        final_output = hsfm.utils.optimize_geotif(out)
        os.remove(out)
        os.rename(final_output, out)
        
        
    if qc == True:
        hsfm.plot.plot_principal_point_and_fiducial_locations(img_gray,
                                                              fiducials,
                                                              principal_point,
                                                              file_name,
                                                              output_directory='qc/image_preprocessing/')

    return intersection_angle 

def pad_image_frame_slices(slices):
    
    left_slice   = slices[0]
    top_slice    = slices[1]
    right_slice  = slices[2]
    bottom_slice = slices[3]
    
    left_slice_padded   = hsfm.core.pad_image(left_slice)
    top_slice_padded    = hsfm.core.pad_image(top_slice)
    right_slice_padded  = hsfm.core.pad_image(right_slice)
    bottom_slice_padded = hsfm.core.pad_image(bottom_slice)
    
    padded_slices = [left_slice_padded, top_slice_padded, right_slice_padded, bottom_slice_padded]
    
    return padded_slices


def detect_fiducials_and_principal_point(windows, 
                                         templates, 
                                         grayscale_unit8_image_array,
                                         noisify=None,
                                         invisible_fiducial=None):
    img_gray = grayscale_unit8_image_array

    window_left   = windows[0]
    window_top    = windows[1]
    window_right  = windows[2]
    window_bottom = windows[3]

    # enhance contrast
    # img_gray_clahe = hsfm.image.clahe_equalize_image(img_gray)
    img_gray_clahe = hsfm.image.img_linear_stretch(img_gray)

    # pull out slices according to window
    slices = hsfm.core.slice_image_frame(img_gray_clahe,windows)
    
    # pad each slice so that the template can be fully moved over a given fiducial marker
    padded_slices = hsfm.core.pad_image_frame_slices(slices)
    
    if noisify == 'left':
        padded_slices[0] = noisify_template(padded_slices[0])
    elif noisify == 'top':
        padded_slices[1] = noisify_template(padded_slices[1])
    elif noisify == 'right':
        padded_slices[2] = noisify_template(padded_slices[2])
    elif noisify == 'bottom':
        padded_slices[3] = noisify_template(padded_slices[3])
          
    # detect fiducial markers
    fiducials = detect_fiducials(padded_slices, 
                                 windows, 
                                 templates, 
                                 invisible_fiducial=invisible_fiducial)
    

    # detect principal point
    principal_point = determine_principal_point(fiducials[0],
                                                fiducials[1],
                                                fiducials[2],
                                                fiducials[3])
                                                
                                                
    return fiducials, principal_point


def determine_intersection_angle(fiducials):
    left_fiducial = fiducials[0]
    top_fiducial = fiducials[1]
    right_fiducial = fiducials[2]
    bottom_fiducial = fiducials[3]
            
    # QC routine
    arc1 = np.rad2deg(np.arctan2(bottom_fiducial[1] - top_fiducial[1],
                  bottom_fiducial[0] - top_fiducial[0]))
    arc2 = np.rad2deg(np.arctan2(right_fiducial[1] - left_fiducial[1],
                  right_fiducial[0] - left_fiducial[0]))
    intersection_angle = arc1-arc2
        
    return np.round(intersection_angle,4)

def pad_image(grayscale_unit8_image_array):
    img = grayscale_unit8_image_array
    a=img.shape[0]+500
    b=img.shape[1]+500
    padded_img = np.zeros([a,b],dtype=np.uint8)
    padded_img.fill(0)
    padded_img[250:250+img.shape[0],250:250+img.shape[1]] = img
    return padded_img
    
def detect_fiducials(padded_slices, 
                     windows, 
                     templates, 
                     invisible_fiducial=None):

    left_slice_padded   = padded_slices[0]
    top_slice_padded    = padded_slices[1]
    right_slice_padded  = padded_slices[2]
    bottom_slice_padded = padded_slices[3]

    window_left   = windows[0]
    window_top    = windows[1]
    window_right  = windows[2]
    window_bottom = windows[3]

    left_template   = templates[0]
    top_template    = templates[1]
    right_template  = templates[2]
    bottom_template = templates[3]
    
    
    left_fiducial = get_fiducial(left_slice_padded, 
                                           left_template, 
                                           window_left, 
                                           position = 'left')
    top_fiducial = get_fiducial(top_slice_padded, 
                                          top_template, 
                                          window_top, 
                                          position = 'top')
    right_fiducial = get_fiducial(right_slice_padded, 
                                            right_template, 
                                            window_right, 
                                            position = 'right')
    bottom_fiducial = get_fiducial(bottom_slice_padded, 
                                             bottom_template, 
                                             window_bottom, 
                                             position = 'bottom')
    
    if invisible_fiducial == 'right':                                 
        offset = top_fiducial[0] - bottom_fiducial[0]
        x = (left_fiducial[0]+2*(top_fiducial[0]-left_fiducial[0]))
        y = left_fiducial[1] + offset
        right_fiducial = (x, y)
    
    fiducials = [left_fiducial, top_fiducial, right_fiducial, bottom_fiducial]
    
    return fiducials

def get_fiducial(grayscale_unit8_image_array,template_file, window, position = None):
    img_gray = grayscale_unit8_image_array
    loc,w,h,res = template_match(img_gray,template_file)
    
    if position == 'left':
        x = window[2] + loc[1][0] + w - 250
        y = window[0] + loc[0][0] + int(h/2) - 250
        return x,y
        
    if position == 'top':
        x = window[2] + loc[1][0] + int(w/2) - 250
        y = loc[0][0] + h - 250
        return x,y
    
    if position == 'right':
        x = window[2] + loc[1][0] - 250
        y = window[0] + loc[0][0] + int(h/2) - 250
        return x,y
    
    if position == 'bottom':
        x = window[2] + loc[1][0] + int(w/2) - 250
        y = window[0] + loc[0][0] - 250
        return x,y
        
def template_match(grayscale_unit8_image_array,template_file):
    img_gray = grayscale_unit8_image_array
    template = cv2.imread(template_file)
    template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    template = hsfm.image.img_linear_stretch(template)
    template = hsfm.core.noisify_template(template)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    loc = np.where(res==res.max())
    return loc,w,h,res
    
def noisify_template(template):
    mask = template > 50
    rand = np.random.randint(0,256,size=template.shape)
    template[mask] = rand[mask]
    return template
    
def determine_principal_point(left_fiducial, top_fiducial, right_fiducial, bottom_fiducial):
    fa = np.array([left_fiducial, top_fiducial, right_fiducial, bottom_fiducial], dtype=float)
    fx = fa[:,0]
    fy = fa[:,1]

    px = fx.reshape(-1,2).mean(axis=0).mean()
    py = fy.reshape(-1,2).mean(axis=0).mean()
    return (px,py)
    
def crop_about_principal_point(grayscale_unit8_image_array, 
                               principal_point,
                               crop_from_pp_dist = 11250):
    img_gray = grayscale_unit8_image_array

    x_L = int(principal_point[0]-crop_from_pp_dist/2)
    x_R = int(principal_point[0]+crop_from_pp_dist/2)
    y_T = int(principal_point[1]-crop_from_pp_dist/2)
    y_B = int(principal_point[1]+crop_from_pp_dist/2)
    
    cropped = img_gray[y_T:y_B, x_L:x_R]
    # TODO
    # why convert to grayscale if already being passed a grayscale image array?
    # cropped = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
    
    cropped = hsfm.image.clahe_equalize_image(cropped)
    cropped = hsfm.image.img_linear_stretch(cropped)

    
    return cropped
    
    
def calculate_distance_principal_point_to_image_edge(focal_length_mm, 
                                                     image_width_px, 
                                                     image_height_px,
                                                     altitude_above_ground_m=1500,
                                                     pixel_pitch=0.02):
                                                     
    """
    Function to calculate distance on ground from principal point to image edge.
    """
    # TODO
    # - Sample elevation of reference DEM at camera center and subtract from
    #   NAGAP altitude metadata to get altitude above ground. Assumes NAGAP
    #   flights left from sea level. May not be necessary if 3000 meters (10,000 feet)
    #   assumption is good enough for ASP bundle_adjust to correct from.
        
    # Divide image width in pixels by pixel pitch to get distance in millimeters.
    image_width_mm = image_width_px * pixel_pitch
    image_height_mm = image_height_px * pixel_pitch
    
    # Calculate angle between principal point and image edge.                      
    angle_pp_img_edge_x = np.degrees(np.arctan((image_width_mm/2)/focal_length_mm))
    angle_pp_img_edge_y = np.degrees(np.arctan((image_height_mm/2)/focal_length_mm))
    
    # In theory, the distance to the sensor should be added to get the true sensor altitude
    # above ground. Likely does not make a difference here.
    sensor_altitude_above_ground_m = focal_length_mm/1000 + altitude_above_ground_m
    
    # Calculate x and y distances seperately in case images are not square. 
    # This is needed for hsfm.trig.calculate_corner()
    distance_pp_img_edge_x_m = np.tan(np.deg2rad(angle_pp_img_edge_x)) * sensor_altitude_above_ground_m
    distance_pp_img_edge_y_m = np.tan(np.deg2rad(angle_pp_img_edge_y)) * sensor_altitude_above_ground_m
    
    # Calculate ground sample distance. (Not needed at this time.)
    # gsd_x = distance_pp_img_edge_x_m/(image_width_px/2)
    # gsd_y = distance_pp_img_edge_y_m/(image_height_px/2)
    
    return distance_pp_img_edge_x_m, distance_pp_img_edge_y_m
    
    
    
def move_match_files_in_sequence(bundle_adjust_directory,
                                 image_prefix,
                                 stereo_directory,
                                 sequence):
    i = sequence[0]
    j = i+1
    
    hsfm.io.create_dir(stereo_directory)
    
    match_files = sorted(glob.glob(os.path.join(bundle_adjust_directory,'*-clean.match')))
    
    for match_file in match_files:
        if image_prefix + str(i) in match_file and image_prefix + str(j) in match_file:
        
            path, name, extension = hsfm.io.split_file(match_file)
            out = os.path.join(stereo_directory, name+ extension)
            shutil.copyfile(match_file, out)
        
            i = i+1
            j = i+1
    
    print('Match files copied to',stereo_directory)
    new_match_files = sorted(glob.glob(os.path.join(stereo_directory,'*.match')))
    
#     return new_match_files


def move_camera_files_in_sequence(bundle_adjust_directory,
                                  image_prefix,
                                  stereo_directory,
                                  sequence,
                                  extension='.tsai'):
    i = sequence[0]
    j = i+1
    
    hsfm.io.create_dir(stereo_directory)
    
    camera_files = sorted(glob.glob(os.path.join(bundle_adjust_directory,'*'+ extension)))
    
    for camera_file in camera_files:
        path, name, extension = hsfm.io.split_file(camera_file)
        out = os.path.join(stereo_directory, name + extension)
        shutil.copyfile(camera_file, out)
    
    print('Camera files copied to', stereo_directory)
    new_camera_files = sorted(glob.glob(os.path.join(stereo_directory,'*'+ extension)))
    
#     return new_camera_files
    
    
def metashape_cameras_to_tsai(project_file_path,
                             original_metashape_metadata,
                             image_extension = '.tif'):
    
    
    
    output_directory = os.path.join(os.path.dirname(project_file_path),'metashape_cameras')
    hsfm.io.create_dir(output_directory)
    
    metashape_export = hsfm.metashape.get_estimated_camera_centers(project_file_path)
    images, lons, lats, alts, yaws, pitches, rolls, omegas, phis, kappas = metashape_export

    images = [s + image_extension for s in images]
    
    dict = {'image_file_name': images, 
        'lon': lons, 
        'lat': lats,
        'alt': alts,
        'lon_acc': 100,
        'lat_acc': 100,
        'alt_acc': 100,
        'yaw': yaws,
        'pitch': pitches,
        'roll': rolls,
        'yaw_acc': 30,
        'pitch_acc': 30,
        'roll_acc': 30,
        'omega': omegas,
        'phi': phis,
        'kappa': kappas}  
    
    df = pd.DataFrame(dict)
    
    metashape_metadata_df = pd.read_csv(original_metashape_metadata)
    unaligned_cameras = df[df.isnull().any(axis=1)]['image_file_name'].values
    # replace unaligned cameras with values from original input metadata
    for i in unaligned_cameras:
        df[df['image_file_name'].str.contains(i)] = metashape_metadata_df[metashape_metadata_df['image_file_name'].str.contains(i)]
    
    gdf = hsfm.geospatial.df_xyz_coords_to_gdf(df,
                                               lon='lon',
                                               lat='lat',
                                               z='alt',
                                               crs='4326')
    gdf = gdf.to_crs({'init' :'epsg:4978'})
    
    gdf = hsfm.geospatial.extract_gpd_geometry(gdf)
    
    gdf = gdf[['image_file_name','x','y','z']]
    
    for index, row in gdf.iterrows():
        image_base_name = row['image_file_name']
        out = os.path.join(output_directory,image_base_name+'.tsai')
        with open(out, 'w') as f:

            C0 = str(row['x'])
            C1 = str(row['y'])
            C2 = str(row['z'])

            line0 = 'VERSION_4\n'
            line1 = 'PINHOLE\n'
            line2 = 'fu = ' + str(7564.1499999999996) +'\n'
            line3 = 'fv = ' + str(7564.1499999999996) +'\n'
            line4 = 'cu = ' + str(5625) +'\n'
            line5 = 'cv = ' + str(5625) +'\n'
            line6 = 'u_direction = 1 0 0\n'
            line7 = 'v_direction = 0 1 0\n'
            line8 = 'w_direction = 0 0 1\n'
            line9 = ' '.join(['C =',C0,C1,C2,'\n'])
            line10 = 'R = 1 0 0 0 1 0 0 0 1\n'
            line11 = 'pitch = 1\n'
            line12 = 'NULL\n'

            f.writelines([line0,line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11,line12])
            
    return output_directory

def targets_df_to_metashape_metadata(df,
                                     output_directory='input_data',
                                     reference_dem    = None,
                                     flight_altitude_m = 1500):

    hsfm.io.create_dir(output_directory)
    
    df['yaw']             = df['heading'].round()
    df['pitch']           = 1.0
    df['roll']            = 1.0
    df['image_file_name'] = df['fileName']+'.tif'

    if reference_dem:
        lons = df['Longitude'].values
        lats = df['Latitude'].values
        
        df['alt']             = hsfm.geospatial.sample_dem(lons, lats, reference_dem)
        df['alt']             = df['alt'] + flight_altitude_m
        df['alt']             = df['alt'].max()

    else:
        df['alt']             = flight_altitude_m

    df['lon']             = df['Longitude'].astype(float).round(6)
    df['lat']             = df['Latitude'].astype(float).round(6)
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
    

def metadata_transform(metadata_file,
                       pc_align_transform_file,
                       output_file_name = None):
    
    '''
    Applies pc_align transform to lat, lon, alt positions.
    '''
    metadata_df = pd.read_csv(metadata_file)
    df = hsfm.geospatial.df_xyz_coords_to_gdf(metadata_df, z='alt')
    df = df.to_crs({'init':'epsg:4978'})
    hsfm.geospatial.extract_gpd_geometry(df)
    
    df_tmp = pd.DataFrame()
    for index, row in df.iterrows():
        C = [row.x,row.y,row.z]
        C_translation, R_transform = hsfm.core.extract_transform(pc_align_transform_file)
        row.x,row.y,row.z = hsfm.core.apply_position_transform(C, C_translation, R_transform)

        row = row.drop(['geometry'])
        df_tmp = df_tmp.append(row)
        
    transformed_metadata = hsfm.geospatial.df_xyz_coords_to_gdf(df_tmp, 
                                                                lon='x', 
                                                                lat= 'y', 
                                                                z='z',
                                                                crs='4978')

    transformed_metadata = transformed_metadata.to_crs({'init':'epsg:4326'})
    hsfm.geospatial.extract_gpd_geometry(transformed_metadata)

    transformed_metadata[['lon', 'lat','alt']] = transformed_metadata[['x', 'y','z']]
    transformed_metadata = transformed_metadata.drop(['x', 'y','z', 'geometry'], axis=1)
    transformed_metadata = transformed_metadata[['image_file_name', 
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
    
    transformed_metadata = transformed_metadata.sort_values(by=['image_file_name'], ascending=True)
    
    if not isinstance(output_file_name, type(None)):
        transformed_metadata.to_csv(output_file_name, index = False)
    
    return transformed_metadata

def extract_transform(pc_align_transform_file):
    transform = pd.read_csv(pc_align_transform_file, header=None, delimiter=r"\s+")
    transform = transform.drop(3)
    C_translation = list(transform[3].values)
    transform = transform.drop([3],axis=1)
    a = list(transform.iloc[0].values)
    b = list(transform.iloc[1].values)
    c = list(transform.iloc[2].values)
    R_transform = [a,b,c]
    
    return C_translation, R_transform

def apply_position_transform(C, C_translation, R_transform):
    xi = R_transform[0][0]*C[0] + R_transform[0][1]*C[1] + R_transform[0][2]*C[2] + C_translation[0]
    yi = R_transform[1][0]*C[0] + R_transform[1][1]*C[1] + R_transform[1][2]*C[2] + C_translation[1]
    zi = R_transform[2][0]*C[0] + R_transform[2][1]*C[1] + R_transform[2][2]*C[2] + C_translation[2]

    return(xi,yi,zi)

def compute_point_offsets(metadata_file_1, 
                          metadata_file_2,
                          lon       = 'lon',
                          lat       = 'lat',
                          alt       = 'alt',
                          epsg_code = '32610'):
    
    df1 = pd.read_csv(metadata_file_1)
    gdf1 = hsfm.geospatial.df_xy_coords_to_gdf(df1, lon='lon', lat='lat')
    gdf1 = gdf1.to_crs({'init' :'epsg:'+epsg_code})
    hsfm.geospatial.extract_gpd_geometry(gdf1)
    
    df2 = pd.read_csv(metadata_file_2)
    gdf2 = hsfm.geospatial.df_xy_coords_to_gdf(df2, lon='lon', lat='lat')
    gdf2 = gdf2.to_crs({'init' :'epsg:'+epsg_code})
    hsfm.geospatial.extract_gpd_geometry(gdf2)
    
    x_offset = gdf1.x - gdf2.x
    y_offset = gdf1.y - gdf2.y
    z_offset = gdf1.alt - gdf2.alt
    
    return x_offset, y_offset, z_offset
    

    
    
    
    
    
    