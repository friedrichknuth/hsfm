#! /usr/bin/env python
import hsfm


project_name = "rainier_carbon"
out_dir = "/data2/elilouis/"

# bounds
carbon_glacier_bounds = (-121.7991, 46.9902, -121.7399, 46.8826)

reference_dem = (
    "/home/elilouis/hsfm-geomorph/data/reference_dem_highres/rainier_lidar_dsm-adj.tif"
)
nagap_metadata_csv = (
    "/home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/nagap_image_metadata.csv"
)
template_parent_dir = (
    "/home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/fiducials/nagap/"
)
output_DEM_resolution = 4
image_matching_accuracy = 4
densecloud_quality = 4
pixel_pitch = 0.02
metashape_licence_file = "/home/elilouis/hsfm/uw_agisoft.lic"

# comment out this line to skip the preprocessing step
# hsfm.batch.NAGAP_pre_process_images(
#     project_name,
#     carbon_glacier_bounds,
#     output_directory=out_dir,
#     nagap_metadata_csv=nagap_metadata_csv,
#     template_parent_dir=template_parent_dir
# )

hsfm.batch.batch_process(
    project_name,
    reference_dem,
    input_directory=out_dir,
    pixel_pitch=pixel_pitch,
    output_DEM_resolution=output_DEM_resolution,
    image_matching_accuracy=image_matching_accuracy,
    densecloud_quality=densecloud_quality,
    metashape_licence_file=metashape_licence_file,
    dem_align_all=True,
    generate_ortho=True
)
