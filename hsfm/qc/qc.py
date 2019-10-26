import glob
import os
import numpy as np
import pandas as pd

import bare

"""
Library used to evaluate intermediate and final products.
"""


def parse_base_names_from_match_file(match_file):
    if 'clean' in match_file:
        match_img1_name = os.path.split(match_file)[-1].split('-')[-2].split('__')[0]
        match_img2_name = os.path.split(match_file)[-1].split('-')[-2].split('__')[1]
    else:
        match_img1_name = os.path.split(match_file)[-1].split('-')[-1].split('__')[0]
        match_img2_name = os.path.split(match_file)[-1].split('-')[-1].split('__')[1].split('.')[0]
    return match_img1_name, match_img2_name


def get_range(x,y):
    x_range_covered = x.where(x<np.percentile(x, 99)).max() - x.where(x>np.percentile(x, 1)).min()
    y_range_covered = y.where(y<np.percentile(y, 99)).max() - y.where(y>np.percentile(y, 1)).min()
    return x_range_covered, y_range_covered

def get_coverage(x,y, image_area):
    x_range_covered, y_range_covered = get_range(x,y)
    coverage_area = x_range_covered * y_range_covered
    total_percent_covered = np.round(coverage_area /image_area,2)
    return total_percent_covered

def get_metric(key, df, dim_x, dim_y):
    image_area = dim_x * dim_y
    left_image_percent_covered = get_coverage(df.xs(key)['x1'],df.xs(key)['y1'], image_area)
    right_image_percent_covered = get_coverage(df.xs(key)['x2'],df.xs(key)['y2'], image_area)
    return left_image_percent_covered, right_image_percent_covered

def calc_matchpoint_coverage(match_files_list):
    keys = []
    for i,v in enumerate(match_files_list):
        match_img1_name, match_img2_name = parse_base_names_from_match_file(v)
        keys.append(match_img1_name+'__'+match_img2_name)

    df_list = []
    for fn in match_files_list:
        df = pd.read_csv(fn,sep=' ')
        df_list.append(df)
    df_combined = pd.concat(df_list,keys=keys)
    
    dim_x = 1400
    dim_y = 1400

    percent_left = []
    percent_right = []
    for i,v in enumerate(keys):
        left_image_percent_covered, right_image_percent_covered = get_metric(v, 
                                                                             df_combined, 
                                                                             dim_x, 
                                                                             dim_y)
        percent_left.append(left_image_percent_covered)
        percent_right.append(right_image_percent_covered)
        
    mydict = {'keys':keys,'left_percent':percent_left, 'right_percent': percent_right}
    
    df = pd.DataFrame.from_dict(mydict)
    return df, df_combined, keys

def compare_left_right(match_files):
    df, df_combined, keys = calc_matchpoint_coverage(match_files)
    df['diff'] = abs(df['left_percent'] - df['right_percent'])
    return df

def compare_matches(df1,df2):
    df2['left_df1_diff'] = abs(df1['left_percent'] - df2['left_percent'])
    df2['right_df1_diff'] = abs(df1['right_percent'] - df2['right_percent'])
    return df2

def id_reruns(cam_solve_match_files, stereo_match_files):
    df1 = compare_left_right(cam_solve_match_files)
    df2 = compare_left_right(stereo_match_files)
    df = compare_matches(df1,df2)
    keys = df[(df['left_df1_diff']>0.1) | (df['right_df1_diff']>0.1)]['keys'].values
    return list(keys)