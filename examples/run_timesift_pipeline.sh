#!/bin/bash

# Run timesift for Mt Hood bounds
python hsfm/timesift/timesift.py \ 
    --output-path           /data2/elilouis/mt_hood_timesift/
    --templates-dir         /home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/fiducials/nagap/ \
    --bounds                -121.7467 45.4722 -121.6138 45.3015 \
    --nagap-metadata-csv        /home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/nagap_image_metadata.csv \
    --densecloud-quality        4 \
    --image-matching-accuracy   2 \
    --output-resolution         2 \
    --pixel-pitch               0.02 \
    --parallelization           2 \
    --reference-dem             /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/2009_merged_cropped_meters.tif \
    --license-path              uw_agisoft.lic

# Run timesift for Carbon Glacier Bounds
python hsfm/timesift/timesift.py \
    --output-path           /data2/elilouis/rainier_carbon_automated_timesift/ \
    --templates-dir         /home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/fiducials/nagap/ \
    --bounds                -121.7991 46.9902 -121.7399 46.8826 \
    --nagap-metadata-csv        /home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/nagap_image_metadata.csv \
    --densecloud-quality        2 \
    --image-matching-accuracy   1 \
    --output-resolution         1 \
    --pixel-pitch               0.02 \
    --parallelization           4 \
    --exclude-years             77 \
    --reference-dem             /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/rainier_lidar_dsm-adj.tif \
    --license-path              uw_agisoft.lic

# Run timesift for Emmons
python hsfm/timesift/timesift.py \
    --output-path           /data2/elilouis/rainier_emmons_automated_timesift/ \
    --templates-dir         /home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/fiducials/nagap/ \
    --bounds                -121.7090 46.8938 -121.6597 46.8383 \
    --nagap-metadata-csv        /home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/nagap_image_metadata.csv \
    --densecloud-quality        4 \
    --image-matching-accuracy   4 \
    --output-resolution         1 \
    --pixel-pitch               0.02 \
    --parallelization           2 \
    --reference-dem             /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/rainier_lidar_dsm-adj.tif \
    --license-path              uw_agisoft.lic

# For Winthrop
python hsfm/timesift/timesift.py \
    --output-path           /data2/elilouis/rainier_winthrop_automated_timesift/ \
    --templates-dir         /home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/fiducials/nagap/ \
    --bounds                -121.7449 46.9580 -121.7014 46.8743 \
    --nagap-metadata-csv        /home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/nagap_image_metadata.csv \
    --densecloud-quality        2 \
    --image-matching-accuracy   1 \
    --output-resolution         1 \
    --pixel-pitch               0.02 \
    --parallelization           4 \
    --reference-dem             /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/rainier_lidar_dsm-adj.tif \
    --license-path              uw_agisoft.lic

# (Re)Running on cluster/date that's already been generated
# Set rotation-enabled to false (timesift run enabled rotation)
nohup python hsfm/pipeline/pipeline.py \
    --reference-dem            /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/rainier_lidar_dsm-adj.tif \
    --input-images-path        /data2/elilouis/rainier_carbon_automated_timesift/preprocessed_images/ \
    --project-name             test \
    --output-path                   /data2/elilouis/rainier_carbon_automated_timesift/individual_clouds/91_9_9_test/ \
    --input-images-metadata-file    /data2/elilouis/rainier_carbon_automated_timesift/individual_clouds/91_9_9_test/metashape_metadata.csv \
    --densecloud-quality            3 \
    --image-matching-accuracy       2 \
    --output-resolution 1 \
    --pixel-pitch                   0.02 \
    --license-path uw_agisoft.lic \
    --iterations 3 \
    --rotation-enabled false &

# For baker Mazama glacier 1
python hsfm/timesift/timesift.py \
    --output-path           /data2/elilouis/baker_mazama_automated_timesift/ \
    --templates-dir         /home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/fiducials/nagap/ \
    --bounds                -121.8240 48.8357 -121.7544 48.8003 \
    --nagap-metadata-csv        /home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/nagap_image_metadata.csv \
    --densecloud-quality        2 \
    --image-matching-accuracy   1 \
    --output-resolution         1 \
    --pixel-pitch               0.02 \
    --parallelization           4 \
    --reference-dem             /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/baker_2015_utm_m.tif \
    --license-path              uw_agisoft.lic

# For baker Coleman glacier 2
python hsfm/timesift/timesift.py \
    --output-path           /data2/elilouis/baker_coleman_automated_timesift/ \
    --templates-dir         /home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/fiducials/nagap/ \
    --bounds                -121.8897 48.8259 -121.8146 48.7752 \
    --nagap-metadata-csv        /home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/nagap_image_metadata.csv \
    --densecloud-quality        2 \
    --image-matching-accuracy   1 \
    --output-resolution         1 \
    --pixel-pitch               0.02 \
    --parallelization           4 \
    --reference-dem             /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/baker_2015_utm_m.tif \
    --license-path              uw_agisoft.lic