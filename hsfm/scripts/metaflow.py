#! /usr/bin/env python

import os
from datetime import datetime

import hsfm

now = datetime.now()

image_matching_accuracy = 0
densecloud_quality = 1

project_name    = 'thunder'
input_path      = '../thunder_1977_batch_backup/input_data/'
output_path     = 'metashape/'
images_metadata = '/mnt/Backups/knuth/data/baker_images_metadata.csv'
focal_length    = 151.303
pixel_pitch     = 0.02
verbose         = True

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
                                            focal_length            = focal_length,
                                            pixel_pitch             = pixel_pitch,
                                            image_matching_accuracy = image_matching_accuracy,
                                            densecloud_quality      = densecloud_quality)

# hsfm.metashape.las2dem(project_name,
#                       output_path)

# hsfm.metashape.images2ortho(project_name,
#                            output_path)

epsg_code = 'EPSG:'+ hsfm.geospatial.get_epsg_code(reference_dem)
dem = hsfm.asp.point2dem(point_cloud_file, 
                              '--nodata-value','-9999',
                              '--tr','0.5',
                              '--t_srs', epsg_code)

clipped_reference_dem = os.path.join(output_path,'reference_dem_clip.tif')
clipped_reference_dem = hsfm.utils.clip_reference_dem(dem, 
                                                      reference_dem,
                                                      output_file_name = clipped_reference_dem,
                                                      buff_size        = 2000)


aligned_dem_file, transform =  hsfm.asp.pc_align_p2p_sp2p(dem, 
                                                          clipped_reference_dem, 
                                                          output_path,
                                                          verbose = verbose)

hsfm.utils.dem_align_custom(clipped_reference_dem,
                            aligned_dem_file,
                            output_path,
                            verbose = verbose)

aligned_dem_file, transform = hsfm.asp.pc_align_tfhs_p2p_sp2p(aligned_dem_file, 
                                                              clipped_reference_dem, 
                                                              output_path,
                                                              verbose = verbose)

hsfm.utils.dem_align_custom(clipped_reference_dem,
                            aligned_dem_file,
                            output_path,
                            verbose = verbose)

print("Elapsed time", str(datetime.now() - now))