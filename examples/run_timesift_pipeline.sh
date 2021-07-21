#!/bin/bash

# Run timesift for Mt Baker bounds
###########################################################################################
# FIRST STEP
python hsfm/timesift/preprocessing.py \
    --output-directory "/data2/elilouis/timesift/baker" \
    --fiducial-templates-directory "/home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/fiducials/nagap/" \
    --nagap-images-csv-path "/home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/nagap_image_metadata_updated_manual.csv" \
    --bounds "-121.94 48.84 -121.70 48.70"

# SECOND STEP
# The first step has a few outputs that must be fed into the second step below
# 1. "metashape_metadata.csv" (a file)
# 2. "image_metadata.csv" (a file)
# 3. preprocessed_images/ (a directory)
# which must be fed in to the second step below.
# We use the same output directory as in the previous step.
# And we must feed additional inputs - two reference DEMs, some Metashape parameters, 
# an output resolution, a Metashape license path, and parallelization value.

python hsfm/timesift/timesift.py \
    --metashape-metadata-file   "/data2/elilouis/timesift/baker/metashape_metadata.csv" \
    --image-metadata-file       "/data2/elilouis/timesift/baker/image_metadata.csv" \
    --raw-images-directory      "/data2/elilouis/timesift/baker/preprocessed_images/" \
    --output-directory          "/data2/elilouis/timesift/baker/" \
    --reference-dem-lowres      "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/baker_2015_utm_10m.tif" \
    --reference-dem-hires       "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/baker_2015_utm_m.tif" \
    --densecloud-quality        2 \
    --image-matching-accuracy   1 \
    --output-resolution         2 \
    --license-path              "uw_agisoft.lic" \
    --parallelization           1

# Run timesift for Mt Adams bounds
###########################################################################################
# See above, replace as necessary with these inputs.
... 
    --output-path           /data2/elilouis/mt_adams_timesift_cam_calib/
    --bounds                -121.5857 46.2708 -121.4036 46.1195
    --reference-dem-lowres  /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/adams/2016_10m.tif
    --reference-dem-hires   /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/adams/2016.tif
    

# Run timesift for Mt Hood bounds
###########################################################################################
# Run first step 
...
    --output-path           /data2/elilouis/mt_hood_timesift_cam_calib/
    --bounds                -121.7467 45.4722 -121.6138 45.3015
    --reference-dem-lowres  /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/2009_10m.tif
    --reference-dem-hires   /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/2009.tif
