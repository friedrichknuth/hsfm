#! /usr/bin/env python

import hsfm


project_name = 'easton'
out_dir      = '../'
nagap_metadata_csv  = 'https://github.com/friedrichknuth/hipp/raw/master/examples/fiducial_proxy_detection/input_data/nagap_image_metadata.csv'
template_parent_dir = '../../../hipp/examples/fiducial_proxy_detection/input_data/fiducials/nagap'


bounds              = (-121.846, 48.76, -121.823, 48.70) # easton
# bounds              = (-121.94, 48.84, -121.70, 48.70) # baker
# bounds              = (-121.7, 48.43, -120.97, 48.28) # south cascade
# bounds              = (-121.935, 47.01, -121.435, 46.70) # rainier

hsfm.batch.NAGAP_pre_process_images(project_name,
                                    bounds,
                                    nagap_metadata_csv=nagap_metadata_csv,
                                    template_parent_dir = template_parent_dir,
                                    year = 77,
                                    day = 27,
                                    month = 9,
                                    output_directory=out_dir)

reference_dem           = '../input_data/reference_dem/lidar/baker.tif'
output_DEM_resolution   = 1
image_matching_accuracy = 1
densecloud_quality      = 2
dem_align_all           = True
metashape_licence_file  = '/mnt/working/knuth/sw/metashape-pro/uw_agisoft.lic'


hsfm.batch.batch_process(project_name,
                         reference_dem,
                         input_directory         = out_dir,
                         pixel_pitch             = 0.02,
                         output_DEM_resolution   = output_DEM_resolution,
                         dem_align_all           = dem_align_all,
                         image_matching_accuracy = image_matching_accuracy,
                         densecloud_quality      = densecloud_quality,
                         metashape_licence_file  = metashape_licence_file,
                         attempts_to_adjust_cams = 0,
                         check_subsets           = False)