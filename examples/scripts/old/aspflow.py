#! /usr/bin/env python

import os
from datetime import datetime

import hsfm

now = datetime.now()

scale=8
input_path    = 'input_data'
output_path   = 'asp'
reference_dem   = '../../data/reference_dems/baker_1_m/baker_2015_utm_m_img_non_glac.tif'
# reference_dem = '../../data/reference_dems/baker_1_m/baker_2015_utm_m.vrt'


images_path  = os.path.join(input_path, 'images')
cameras_path = os.path.join(input_path, 'cameras')

print('rescale')
images_path = hsfm.batch.rescale_images(images_path,
                                        output_path,
                                        scale=scale)
cameras_path = hsfm.batch.rescale_tsai_cameras(cameras_path, 
                                               output_path,
                                               scale=scale)

print('cam solve')
camera_solve_path = hsfm.asp.generate_match_points(images_path,
                                                   cameras_path,
                                                   output_path,
                                                   qc = True,
                                                   verbose=True)
print('prep ba')
overlap_list = hsfm.batch.prepare_ba_run(input_path,
                                         output_path,
                                         scale)

print('bundle adjust')
hsfm.asp.bundle_adjust_custom(images_path,
                              cameras_path,
                              output_path,
                              overlap_list=overlap_list,
                              qc=True,
                              verbose=True)
print('prep stereo')
hsfm.batch.prepare_stereo_run(output_path)

print('run stereo')
hsfm.asp.iter_stereo_pairs(output_path,
                           images_path,
                           qc=True)

print('mosaic')
mosaic_file = hsfm.asp.dem_mosaic_custom(output_path)

print('clip')
clipped_reference_dem_file = hsfm.utils.clip_reference_dem(mosaic_file, 
                                                           reference_dem,
                                                           output_file_name = os.path.join(output_path,
                                                                                           'reference_dem_clip.tif'))
print('pc align')
aligned_mosaic_file_name = hsfm.asp.pc_align_custom(mosaic_file,
                                                    clipped_reference_dem_file,
                                                    output_path,
                                                    verbose=True)

print('dem align')
dem_difference_file_name, aligned_dem_file_name = hsfm.utils.dem_align_custom(reference_dem,
                                                                              aligned_mosaic_file_name,
                                                                              output_path,
                                                                              verbose=True)

print("Elapsed time", str(datetime.now() - now))