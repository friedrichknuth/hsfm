#! /usr/bin/env python

import hipp
import hsfm

from datetime import datetime
import os
import glob
import pandas as pd

# bounds= (-121.846, 48.76, -121.823, 48.70) # easton
bounds= (-121.94, 48.84, -121.70, 48.70) # baker
# bounds= (-121.7, 48.43, -120.97, 48.28) # south cascade

out_dir = '/mnt/1.0_TB_VOLUME/knuth/nagap/baker/'
template_parent_dir = 'input_data/fiducials/nagap'
nagap_metadata_csv = 'input_data/nagap_image_metadata.csv'

template_dirs = sorted(glob.glob(os.path.join(template_parent_dir, '*')))

template_types = []
for i in template_dirs:
    template_types.append(i.split('/')[-1])

df = pd.read_csv(nagap_metadata_csv)
df = df[df['fiducial_proxy_type'].isin(template_types)]

df = hipp.dataquery.NAGAP_pre_select_images(df,bounds = bounds)
rolls = sorted(list(set(df['Roll'].values)))



######################################################
srtm_reference_dem = hsfm.utils.download_srtm(bounds)
######################################################





for roll in rolls:
    df_roll = df[df['Roll']  == roll].copy()
    
    for i,v in enumerate(template_types):
        df_tmp = df_roll[df_roll['fiducial_proxy_type']  == v].copy()
        
        if not df_tmp.empty:
            
            image_directory = hipp.dataquery.NAGAP_download_images_to_disk(
                                             df_tmp,
                                             output_directory=os.path.join(
                                                                      out_dir, 
                                                                      'input_data',
                                                                      roll,
                                                                      v+'_raw_images'))
            
            template_directory = template_dirs[i]

            hipp.batch.preprocess_with_fiducial_proxies(
                       image_directory,
                       template_directory,
                       output_directory=os.path.join(
                                                out_dir, 
                                                'input_data',
                                                roll,
                                                v+'_cropped_images'),
                                                        
                       qc_df_output_directory=os.path.join(
                                                      out_dir,
                                                      'input_data',
                                                      roll,
                                                      'qc',
                                                      v+'_proxy_detection_data_frames'),
                       qc_plots_output_directory=os.path.join(
                                                         out_dir,
                                                         'input_data',
                                                         roll,
                                                         'qc',
                                                         v+'_proxy_detection'))
            
############################################################################################################

            hsfm.core.determine_image_clusters(df_tmp,
                                               output_directory=os.path.join(out_dir,
                                                                             'input_data',
                                                                             roll, 
                                                                             'sfm'),
                                               reference_dem=srtm_reference_dem,
                                               image_directory = os.path.join(out_dir, 
                                                                              'input_data',
                                                                              roll,
                                                                              v+'_cropped_images')

                                               
project_name               = 'baker'
reference_dem              = '/mnt/Backups/knuth/hsfm_processing/nagap/data/reference_dems/baker_1_m/baker_2015_utm_m.tif'
# focal_length               = 151.303 # if not specified, will read from metadata file.
pixel_pitch                = 0.02
output_DEM_resolution      = 2 # if not specified, will compute based on GSD.
image_matching_accuracy    = 1
densecloud_quality         = 2
metashape_licence_file     = '/opt/metashape-pro/uw_agisoft.lic'
verbose                    = True
cleanup                    = True


batches = sorted(glob.glob('/mnt/1.0_TB_VOLUME/knuth/nagap/baker/input_data/*/sfm/cl*'))

for i in batches:
    try:
        print('\n\n'+i)

        now = datetime.now()

        cluster_project_name = project_name+'_'+i.split('/')[-1]

        images_path          = os.path.join(i,'images')
        images_metadata_file = os.path.join(i,'metashape_metadata.csv')
        output_path          = os.path.join(i,'metashape')

        hsfm.batch.metaflow(cluster_project_name,
                            images_path,
                            images_metadata_file,
                            reference_dem,
                            output_path,
                            pixel_pitch,
                            output_DEM_resolution   = output_DEM_resolution,
                            image_matching_accuracy = image_matching_accuracy,
                            densecloud_quality      = densecloud_quality,
                            metashape_licence_file  = metashape_licence_file,
                            verbose                 = verbose,
                            cleanup                 = cleanup)
    except:
        print('FAIL:', i)

    print('\n\n'+i)
    print("Elapsed time", str(datetime.now() - now), '\n\n')