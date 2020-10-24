#! /usr/bin/env python

import hsfm


project_name = 'easton'
roll         = '77V6'
out_dir      = '../'
bounds              = (-121.846, 48.76, -121.823, 48.70) # easton

# bounds              = (-121.94, 48.84, -121.70, 48.70) # baker
# bounds              = (-121.7, 48.43, -120.97, 48.28) # south cascade


hsfm.batch.NAGAP_pre_process_images(project_name,
                                    bounds,
                                    roll = '77V6')


reference_dem           = '/mnt/Backups/knuth/hsfm_processing/nagap/data/reference_dems/baker_1_m/baker_2015_utm_m.tif'
output_DEM_resolution   = 2
image_matching_accuracy = 1
densecloud_quality      = 2
metashape_licence_file  = '/opt/metashape-pro/uw_agisoft.lic'


hsfm.batch.batch_process(project_name,
                         reference_dem,
                         input_directory         = out_dir,
                         output_DEM_resolution   = output_DEM_resolution,
                         image_matching_accuracy = image_matching_accuracy,
                         densecloud_quality      = densecloud_quality,
                         metashape_licence_file  = metashape_licence_file)