import shutil
import numpy as np
import pandas as pd
from urllib.request import urlopen
import cv2
from skimage import exposure
import glob
import os

import hsfm.image

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
    
    
def select_images_for_download(csv_file_name,subset=None):
    
    """
    Function to convert input csv to dataframe.
    """
    # - Add option to subset with list if range not suitable
    df = pd.read_csv(csv_file_name)
    df = df.drop_duplicates()
    if subset != None:
        df['image_index_number'] = df['fileName'].str[11:].apply(int)
        df = df[df['image_index_number'].between(subset[0],subset[1])]
    return df
    
def download_image(pid):
    base_url = 'https://arcticdata.io/metacat/d1/mn/v2/object/'
    url = base_url+pid
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image
    
def slice_image_frame(grayscale_unit8_image_array):
    
    img_gray = grayscale_unit8_image_array
    
    window_left = [5000,7000,250,1500]
    window_right = [5000,6500,12000,img_gray.shape[1]]
    window_top = [0,500,6000,7200]
    window_bottom = [11000,img_gray.shape[0],6000,7200]
    
    left_slice = img_gray[window_left[0]:window_left[1],window_left[2]:window_left[3]]
    top_slice = img_gray[window_top[0]:window_top[1],window_top[2]:window_top[3]]
    right_slice = img_gray[window_right[0]:window_right[1],window_right[2]:window_right[3]]
    bottom_slice = img_gray[window_bottom[0]:window_bottom[1],window_bottom[2]:window_bottom[3]]
    
    return left_slice, top_slice, right_slice, bottom_slice
    
def pad_image_frame_slices(left_slice, top_slice, right_slice, bottom_slice):
    
    left_slice_padded = pad_img(left_slice)
    top_slice_padded = pad_img(top_slice)
    right_slice_padded = pad_img(right_slice)
    bottom_slice_padded = pad_img(bottom_slice)
    
    return left_slice_padded, top_slice_padded, right_slice_padded, bottom_slice_padded
    
def pad_image(grayscale_unit8_image_array):
    img = grayscale_unit8_image_array
    a=img.shape[0]+500
    b=img.shape[1]+500
    padded_img = np.zeros([a,b],dtype=np.uint8)
    padded_img.fill(0)
    padded_img[250:250+img.shape[0],250:250+img.shape[1]] = img
    return padded_img
    
    
def get_fiducials(grayscale_unit8_image_array,template_file, window, position = None):
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
    template = noisify_template(template)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    loc = np.where(res==res.max())
    return loc,w,h,res
    
def noisify_template(template):
    mask = template > 50
    rand = np.random.randint(0,256,size=template.shape)
    template[mask] = rand[mask]
    return template
    
def principal_point(left_fiducial, top_fiducial, right_fiducial, bottom_fiducial):
    fa = np.array([left_fiducial, top_fiducial, right_fiducial, bottom_fiducial], dtype=float)
    fx = fa[:,0]
    fy = fa[:,1]

    px = fx.reshape(-1,2).mean(axis=0).mean()
    py = fy.reshape(-1,2).mean(axis=0).mean()
    return (px,py)
    
def crop_about_principal_point(grayscale_unit8_image_array, principal_point):
    img_gray = grayscale_unit8_image_array
    
    y_dist = 11250
    x_dist = 11250

    x_L = int(principal_point[0]-x_dist/2)
    x_R = int(principal_point[0]+x_dist/2)
    y_T = int(principal_point[1]-y_dist/2)
    y_B = int(principal_point[1]+y_dist/2)
    
    cropped = img_gray[y_T:y_B, x_L:x_R]
    cropped = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
    
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
    
    return new_match_files


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
    
    return new_camera_files
    
    


