import hsfm
import os
from joblib import Parallel, delayed
import multiprocessing

parallelization = 6
post_timesift_output_directory = '/data2/elilouis/rainier_carbon_timesift/rainier_carbon_post_timesift_hsfm/'
project_name_prefix = 'rainier_carbon'
reference_dem = "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/rainier_lidar_dsm-adj.tif"
input_images_path = '/data2/elilouis/rainier_carbon_timesift/preprocessed_images'
densecloud_quality = 3
image_matching_accuracy = 4
output_DEM_resolution = 2
pixel_pitch = 0.02
license_path = 'uw_agisoft.lic'


def process_image_batch(input_sfm_dir):
    print(f'\n\nProcessing image batch for date {input_sfm_dir}')
    output_path = os.path.join(post_timesift_output_directory, input_sfm_dir)
    input_images_metadata_file = os.path.join(output_path, 'metashape_metadata.csv')
    project_name = project_name_prefix + '_' + input_sfm_dir

    def call(project_name, output_path, input_images_metadata_file):
        return hsfm.batch.pipeline(
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

    new_camera_metadata = call(project_name, os.path.join(output_path, '1/'), input_images_metadata_file)
    new_camera_metadata_2 = call(f"{project_name}_2", os.path.join(output_path, '2/'), new_camera_metadata)
    new_camera_metadata_3 = call(f"{project_name}_3", os.path.join(output_path, '3/'), new_camera_metadata_2)
    return new_camera_metadata_3    


if __name__ == '__main__':
    input_sfm_dirs = os.listdir(post_timesift_output_directory)
    results = Parallel(n_jobs=parallelization)(
        delayed(process_image_batch)(i) for i in input_sfm_dirs
    )