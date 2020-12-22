#! /usr/bin/env python

import hsfm
from os.path import join

data_dir = os.environ('hsfm_data_dir')
license_dir = os.environ('agisoft_LICENSE')
reference_dem = join(data_dir,'/reference_dem_highres/baker/baker_trimmed_easton_2015_utm_m.tif')

project_name = 'easton'
roll         = '77V6'
out_dir      = join(data_dir, 'easton_77_metaflow')
bounds = (-121.846, 48.76, -121.823, 48.70)

hsfm.batch.NAGAP_pre_process_images(project_name,
                                    bounds,
                                    roll = roll,
                                    output_directory=out_dir)

output_DEM_resolution   = 1
image_matching_accuracy = 2
densecloud_quality      = 4
pixel_pitch             = 0.02
metashape_licence_file  = join(license_dir, 'uw_agisoft.lic')


hsfm.batch.batch_process(project_name,
                         reference_dem,
                         input_directory         = out_dir,
                         pixel_pitch             = pixel_pitch,
                         output_DEM_resolution   = output_DEM_resolution,
                         image_matching_accuracy = image_matching_accuracy,
                         densecloud_quality      = densecloud_quality,
                         metashape_licence_file  = metashape_licence_file,
                         attempts_to_adjust_cams = 1)