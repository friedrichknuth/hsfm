import numpy as np
import pandas as pd
from urllib.request import urlopen
import cv2
from skimage import exposure

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
    
    
def select_images_for_download(csv_file_name,image_index_range=None):
    df = pd.read_csv(csv_file_name)
    df = df.drop_duplicates()
    if image_index_range != None:
        df['image_index_number'] = df['fileName'].str[11:].apply(int)
        df = df[df['image_index_number'].between(image_index_range[0],image_index_range[1])]
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
    window_right = [5000,6500,12000,12391]
    window_top = [0,500,6000,7200]
    window_bottom = [11000,11509,6000,7200]
    
    left_slice = img_gray[window_left[0]:window_left[1],window_left[2]:window_left[3]]
    top_slice = img_gray[window_top[0]:window_top[1],window_top[2]:window_top[3]]
    right_slice = img_gray[window_right[0]:window_right[1],window_right[2]:window_right[3]]
    bottom_slice = img_gray[window_bottom[0]:window_bottom[1],window_bottom[2]:window_bottom[3]]
    
    return left_slice, top_slice, right_slice, bottom_slice
    
def convert_image_to_grayscale(unit8_multiband_image):
    img = unit8_multiband_image
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img_gray
    
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