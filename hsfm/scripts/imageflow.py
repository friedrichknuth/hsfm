#! /usr/bin/env python

import numpy as np
import os
import hsfm


input_csv = '../../data/glacier_names_pids.csv'
template_directory = '../../data/notch_templates'
reference_dem = '../../data/reference_dems/baker/SRTM3/cache/srtm.vrt'
output_directory = 'input_data'
prefix = 'NAGAP_77V6_0'
image_range= (81,85)
focal_length_mm = 151.303


image_suffix_list = []
image_suffix_list.extend(np.arange(image_range[0],image_range[-1]+1,1))
camera_positions_file_name = hsfm.core.pre_select_target_images(input_csv,
                                                                prefix,
                                                                image_suffix_list)

hsfm.batch.download_images_to_disk(os.path.join(output_directory, 'targets.csv'), 
                                   output_directory=os.path.join(output_directory, 'tn_raw_images'),
                                   image_type='pid_tn') # pid_tiff, pid_jpeg

print('Pre-processing images')
image_directory = hsfm.batch.preprocess_images(template_directory,
                                               camera_positions_file_name=camera_positions_file_name,
                                               qc=True)

print('Generating camera models')
camera_directory = hsfm.batch.batch_generate_cameras(image_directory,
                                                     camera_positions_file_name,
                                                     reference_dem,
                                                     focal_length_mm,
                                                     output_directory)