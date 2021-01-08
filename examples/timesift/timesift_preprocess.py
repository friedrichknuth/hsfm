import hsfm
import hipp
import os
from pathlib import Path

# # Set up Parameters for Processing 
output_directory = '/data2/elilouis/rainier_carbon_timesift/'
templates_dir = "/home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/fiducials/nagap/"
bounds = (-121.7991, 46.9902, -121.7399, 46.8826) # Carbon Glacier bounds
nagap_metadata_csv = "/home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/nagap_image_metadata.csv"
reference_dem = "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/rainier_lidar_dsm-adj.tif"
project_name = "rainier_carbon_timesift"
input_images_metadata_file = "/data2/elilouis/rainier_carbon_timesift/metashape_metadata_no77.csv" # THIS NEEDS TO BE HANDLED BETTER - HOW TO AUTOMATE THE KNOWLEDGE THAT 77 IS BAD??? Need to look at each image to determine which have missing sides.
densecloud_quality = 2
image_matching_accuracy = 1
output_DEM_resolution = 1
pixel_pitch = 0.02
license_path = "uw_agisoft.lic"

# # 
input_images_path = os.path.join(output_directory, "preprocessed_images")
output_path = os.path.join(output_directory, project_name + '_hsfm/')
df = hipp.dataquery.NAGAP_pre_select_images(nagap_metadata_csv, bounds = bounds)

all_fiducial_types = list(df.fiducial_proxy_type.unique())
all_fiducial_types

# # Download Images (by fiducial proxy type)

for fiducial_type in all_fiducial_types:
    filtered_df = df[df.fiducial_proxy_type == fiducial_type]
    target_dir = os.path.join(output_directory, 'raw_images', fiducial_type)
    print(f"Attempting to download {len(filtered_df)} images with fiducial proxy type {fiducial_type}")
    hipp.dataquery.NAGAP_download_images_to_disk(
        filtered_df,
        output_directory=target_dir
    )
    num_images_downloaded = len(os.listdir(target_dir))
    print(f"Downloaded {len(filtered_df)} images with fiducial proxy type {fiducial_type}")

# # Preprocess Images (by fiducial proxy type)
# This might be sketch...not being specific to each year at all.

### TO DO
#### Need to flag the 1977 images (all the triangle fiducial marker images) as missing the top marker

for fiducial_type in all_fiducial_types:
    template_dir = os.path.join(templates_dir,fiducial_type)
    src_dir = os.path.join(output_directory, 'raw_images', fiducial_type)
    target_dir = os.path.join(output_directory, 'preprocessed_images')
    hipp.batch.preprocess_with_fiducial_proxies(
        src_dir,
        template_dir,
        output_directory = target_dir,
        qc_df_output_directory=os.path.join(output_directory, 'qc', fiducial_type+'_proxy_detection_data_frames'),
        qc_plots_output_directory=os.path.join(output_directory, 'qc', fiducial_type+'_proxy_detection_plots'),
        missing_proxy = None
    )

num_raw_images = len(list(Path(os.path.join(output_directory, 'raw_images')).rglob('*.tif')))
num_preprocessed_images = len(os.listdir(target_dir))
print(f'Preprocessed {num_preprocessed_images} of {num_raw_images} raw images.')

# # Create Metadata File

hsfm.core.prepare_metashape_metadata(
    df,
    output_directory=output_directory,
)

# # Metashape SfM Processing
hsfm.batch.pipeline(
    project_name                = project_name,
    output_path                 = output_path,
    input_images_metadata_file  = input_images_metadata_file,
    reference_dem = reference_dem,
    input_images_path = input_images_path,
    densecloud_quality = densecloud_quality,
    image_matching_accuracy = image_matching_accuracy,
    output_DEM_resolution = output_DEM_resolution,
    pixel_pitch = pixel_pitch,
    license_path = license_path
)



