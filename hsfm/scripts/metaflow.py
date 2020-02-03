#! /usr/bin/env python

import os
from datetime import datetime

import hsfm

now = datetime.now()

scale=8
project_name    = 'thunder'
input_path      = 'input_data'
output_path     = 'metashape/'
images_metadata = '/mnt/Backups/knuth/data/images_metadata.csv'
reference_dem   = '../../data/reference_dems/baker_1_m/baker_2015_utm_m_img_non_glac.tif'
# reference_dem = '../../data/reference_dems/baker_1_m/baker_2015_utm_m.vrt'


now = datetime.now()

hsfm.metashape.authentication()

images_path  = os.path.join(input_path, 'images')

point_cloud_file = hsfm.metashape.images2las(project_name,
                                            images_path,
                                            images_metadata,
                                            reference_dem,
                                            output_path,
                                            image_matching_accuracy=scale)

hsfm.metashape.las2dem(project_name,
                      output_path)

hsfm.metashape.images2ortho(project_name,
                           output_path)

dem = hsfm.asp.point2dem(point_cloud_file, 
                              '--nodata-value','-9999',
                              '--tr','0.5',
                              '--t_srs', 'EPSG:32610')

clipped_reference_dem = hsfm.utils.clip_reference_dem(dem, 
                                                      reference_dem,
                                                      output_file_name = os.path.join(output_path,
                                                                                      'reference_dem_clip.tif'))
aligned_dem =  hsfm.asp.pc_align_custom(dem, 
                                        clipped_reference_dem, 
                                        output_path)

hsfm.utils.dem_align_custom(clipped_reference_dem,
                            aligned_dem,
                            output_path)

print("Elapsed time", str(datetime.now() - now))