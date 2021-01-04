### Script to run 3 iterations of the HSFM pipeline on a specified group of images.
### Example arguments:
### python pipeline.py \
###     --reference-dem                 /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/rainier_lidar_dsm-adj.tif \
###     --input-images-path             /data2/elilouis/rainier_carbon/input_data/73V3/00/00/block_cropped_images/ \
###     --project-name                  rainier_carbon_73 \
###     --output-path                   /data2/elilouis/rainier_carbon_73/ \
###     --input-images-metadata-file    /data2/elilouis/rainier_carbon/input_data/73V3/00/00/sfm/cluster_000/metashape_metadata.csv \
###     --densecloud-quality            3 \
###     --image-matching-accuracy       4 \
###     --pixel-pitch                   0.02
### 

import hsfm
import argparse

def parse_args():
    parser = argparse.ArgumentParser('Run the HSFM Pipeline on a batch of images.'))
    parser.add_argument('-r','--reference-dem', help='Path to reference DEM file.', required=True)
    parser.add_argument('-i','--input-images-path', help='Path to directory containing preprocessed input images listed in the input images metadata file.', required=True)
    parser.add_argument('-p', '--project-name', help="Name for Metashape project files.", required=True)
    parser.add_argument('-o', '--output-path', help="Path to directory where pipeline output will be stored.", required=True)
    parser.add_argument('-f', '--input-images-metadata-file', help="Path to csv file containing appropriate Metashape metadata with files names.", required=True)
    parser.add_argument('-q', '--densecloud-quality', help='Densecloud quality parameter for Metashape. Values include 1 - 4, from highest to lowest quality.', required=True)
    parser.add_argument('-a', '--image-matching-accuracy', help='Image matching accuracy parameter for Metashape. Values include 1 - 4, from highest to lowest quality.', required=True)
    parser.add_argument('-x', '--pixel-pitch', help='Pixel pitch/scanning resolution.', required=True)
    parser.add_argument('-l', '--licence-path', help='Path to Agisoft license file', required=False)
    return parser.parse_args()

def run_pipeline(args):
    reference_dem = args.reference_dem
    input_images_path = args.input_images_path
    project_name = args.project_name
    output_path = args.output_path
    input_images_metadata_file = args.input_images_metadata_file
    densecloud_quality = args.densecloud_quality
    image_matching_accuracy = args.image_matching_accuracy
    pixel_pitch = args.pixel_patch
    license_path = args.license_path

    def call(project_name, output_path, input_images_metadata_file):
        hsfm.batch.pipeline(
            project_name                = project_name,
            output_path                 = output_path,
            input_images_metadata_file  = input_images_metadata_file,
            reference_dem = reference_dem,
            input_images_path = input_images_path,
            densecloud_quality = densecloud_quality,
            image_matching_accuracy = image_matching_accuracy,
            pixel_pitch = pixel_pitch
            license_path = license_path
        )

    call(project_name, os.path.join(output_path, '1'), input_images_metadata_file)
    call(f"{project_name}_2", os.path.join(output_path, '2'), os.path.join(output_path, '1', "nuth_aligned_bundle_adj_metadata.csv"))
    call(f"{project_name}_3", os.path.join(output_path, '3'), os.path.join(output_path, '2', "nuth_aligned_bundle_adj_metadata.csv"))
               

def main():
    print("Parsing arguments...")
    args=parse_args()
    print(f"Arguments: \n\t {args}")
    run_pipeline(args)

if __name__ == '__main__':
    main()