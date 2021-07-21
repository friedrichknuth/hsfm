import hsfm
from hsfm.pipeline import Pipeline
import hipp
from pathlib import Path
import pandas as pd
import os
import argparse
import glob


class TimesiftPipeline:
    """Timesift Historical Structure from Motion pipeline. Generates a timeseries of DEMs 
    for a given set of images that span multiple years.

    A Pipeline is run to identify camera positions using images from all dates 
    (called 4D alignment, or timesift). These cameras/images are then sorted into
    groups by month and a Pipeline is run for each set of images. The camera models
    determined/calibrated during 4D alignment are passed into the Pipeline runs 
    for separate sets of images so that camera models are only calibrated during the 
    4d alignment step.
    """


    def __init__(
        self,
        metashape_metadata_file,
        image_metadata_file,
        raw_images_directory,
        output_directory,
        reference_dem_lowres,
        reference_dem_hires,
        densecloud_quality=2,
        image_matching_accuracy=1,
        output_DEM_resolution=2,
        license_path="uw_agisoft.lic",
        parallelization=1
    ):
        self.metashape_metadata_file = metashape_metadata_file
        self.image_metadata_file = image_metadata_file
        self.raw_images_directory = raw_images_directory
        self.output_directory = output_directory
        self.reference_dem_lowres = reference_dem_lowres
        self.reference_dem_hires = reference_dem_hires

        self.densecloud_quality = densecloud_quality
        self.image_matching_accuracy = image_matching_accuracy
        self.output_DEM_resolution = output_DEM_resolution
        self.license_path = license_path
        self.parallelization = parallelization

        self.multi_epoch_project_name = "multi_epoch_densecloud"
        # Output directory for Multi Epoch Dense cloud generation
        self.multi_epoch_cloud_output_path = os.path.join(
            output_directory, "multi_epoch_cloud/"
        )
        self.individual_clouds_output_path = os.path.join(
            output_directory, "individual_clouds/"
        )

        self.camera_calibration_directory = os.path.join(
            self.multi_epoch_cloud_output_path, 
            'camera_calibrations'
        )

    def run(self):
        """Run the full pipeline.
        1. Use hsfm.pipeline.Pipeline to generate and align a multi-epoch densecloud.
        2. Generate image clusters and organize directories for the next step. Images
            are clustered by date (day) and roll.
        3. For each date, use hsfm.pipeline.Pipeline to generate and align a single-epoch
            densecloud (images taken the same month are grouped together for "individual date" 
            processing).
        """

        metadata_timesift_aligned_file = self._generate_multi_epoch_densecloud()
        _ = self._save_image_footprints()
        _ = self._export_camera_calibration_files()
        _ = self._prepare_single_date_data(metadata_timesift_aligned_file)
        dict_of_subsets_by_date = self._find_clusters_in_individual_clouds()
        _ = self._generate_subsets_for_each_date(dict_of_subsets_by_date)
        _ = self._process_individual_clouds()

    def _generate_multi_epoch_densecloud(self):
        print("Generating and aligning multi-epoch densecloud...")
        pipeline = hsfm.pipeline.Pipeline(
            self.raw_images_directory,
            self.reference_dem_lowres,
            self.image_matching_accuracy,
            self.densecloud_quality,
            self.output_DEM_resolution,
            self.multi_epoch_project_name,
            self.multi_epoch_cloud_output_path,
            self.metashape_metadata_file,
            license_path=self.license_path,
        )
        return pipeline.run()

    def _save_image_footprints(self):
        """Creates geojson file with image footprints exported from the timesift Metashape project 
        file
        """
        gdf = hsfm.metashape.image_footprints_from_project(self._get_timesift_project_path())
        gdf.to_file(
            os.path.join(self.output_directory, "timesifted_image_footprints.geojson"), 
            driver="GeoJSON"
        )


    #TODO there must be a better alternative to this...
    def _get_timesift_project_path(self):
        return os.path.join(self.multi_epoch_cloud_output_path, self.multi_epoch_project_name + ".psx")

    def _export_camera_calibration_files(self):
        import Metashape
        metashape_project_file = os.path.join(self.multi_epoch_cloud_output_path, self.multi_epoch_project_name + ".psx")
        camera_exports_dir = self.camera_calibration_directory
        if not os.path.exists(camera_exports_dir):
            os.mkdir(camera_exports_dir)
        ##make this directory if it does not exist
        doc = Metashape.Document()
        doc.open(metashape_project_file)
        chunk = doc.chunk
        for camera in chunk.cameras:
            camera.sensor.calibration.save(f"{camera_exports_dir}/{camera.label}.xml")


    # This has a lot of annoying data manipulation that could be avoided with better handling/updating
    # of camera position.
    def _prepare_single_date_data(self, aligned_cameras_file):
        """Create a CSV file of image metadata for each date."""
        print("Preparing data for individual clouds...")
        aligned_cameras_df = pd.read_csv(aligned_cameras_file)
        image_metadata_df = pd.read_csv(self.image_metadata_file)
        image_metadata_df['image_file_name'] = image_metadata_df['fileName'] + '.tif'
        joined_df = pd.merge(
            image_metadata_df, aligned_cameras_df, on="image_file_name"
        )
        joined_df["Year"] = joined_df["Year"].astype('str')
        joined_df["Month"] = joined_df["Month"].astype('str').fillna("0")
        joined_df["Day"] = joined_df["Day"].astype('str').fillna("0")
        
        daily_dir_names = []
        for date_tuple, df in joined_df.groupby(["Year", "Month"]):
            if len(df) < 3:
                print(f"Skipping individual cloud for {date_tuple} because there are less than 3 images.")
            else:
                date_string="_".join(date_tuple)
                # Drop unncessary-for-processing columns (we only needed them to separate by year)
                df = df.drop(["fileName", "Year", "Month", "Day"], axis=1)
                # Put columns in proper order
                df = df[
                    [
                        "image_file_name",
                        "lon",
                        "lat",
                        "alt",
                        "lon_acc",
                        "lat_acc",
                        "alt_acc",
                        "yaw",
                        "pitch",
                        "roll",
                        "yaw_acc",
                        "pitch_acc",
                        "roll_acc",
                        "focal_length",
                        "pixel_pitch"
                    ]
                ]
                csv_output_path = os.path.join(
                    self.individual_clouds_output_path,
                    date_string,
                    "metashape_metadata.csv",
                )
                parent_dir = os.path.dirname(csv_output_path)
                if not os.path.isdir(parent_dir):
                    os.makedirs(parent_dir)
                df.to_csv(csv_output_path, index=False)
                daily_dir_names.append(parent_dir)
        

    def _find_clusters_in_individual_clouds(self):
        print("Searching all dates for clusters/subsets")
        individual_dir_to_subset_list_dict = {}
        for individual_sfm_dir in os.listdir(self.individual_clouds_output_path):
            try:
                print(f"Processing single date ({individual_sfm_dir}) images to check for clusters/subsets ...")
                output_path = os.path.join(self.individual_clouds_output_path, individual_sfm_dir, "cluster_metashape_run")
                input_images_metadata_file = os.path.join(self.individual_clouds_output_path, individual_sfm_dir, "metashape_metadata.csv")
                metashape_project_file, point_cloud_file = hsfm.metashape.images2las(
                    individual_sfm_dir,
                    self.raw_images_directory,
                    input_images_metadata_file,
                    output_path,
                    focal_length            = pd.read_csv(input_images_metadata_file)['focal_length'].iloc[0],
                    image_matching_accuracy = self.image_matching_accuracy,
                    densecloud_quality      = self.densecloud_quality,
                    keypoint_limit          = 40000,
                    tiepoint_limit          = 4000,
                    rotation_enabled        = True,
                    export_point_cloud      = False
                )
                ba_cameras_df, unaligned_cameras_df = hsfm.metashape.update_ba_camera_metadata(metashape_project_file, input_images_metadata_file)
                ba_cameras_df.to_csv(
                    input_images_metadata_file.replace("metashape_metadata.csv", "single_date_multi_cluster_bundle_adjusted_metashape_metadata.csv"),
                    index=False
                    )
                list_of_subsets = hsfm.metashape.determine_clusters(metashape_project_file)
                with open(
                    input_images_metadata_file.replace("metashape_metadata.csv", "subsets.txt"), 
                    'w'
                ) as f:
                    f.write(str(list_of_subsets))
                individual_dir_to_subset_list_dict[individual_sfm_dir] =  list_of_subsets
            except Exception as e:
                print(f'Failure processing/finding clusters in individual clouds for cloud {individual_sfm_dir}: \n {e}')
        return individual_dir_to_subset_list_dict

    def _generate_subsets_for_each_date(self, dict_of_subsets_by_date):
        print("Generating subsets/clusters for each date.")
        for individual_sfm_dir in os.listdir(self.individual_clouds_output_path):
            input_images_metadata = pd.read_csv(
                os.path.join(self.individual_clouds_output_path, individual_sfm_dir, "metashape_metadata.csv")
            )
            for index, subset in enumerate(dict_of_subsets_by_date[individual_sfm_dir]):
                cluster_name = f"cluster{index}"
                cluster_metashape_data = input_images_metadata[
                    input_images_metadata['image_file_name'].str.replace(".tif", "").isin(subset)
                ]
                cluster_output_dir = os.path.join(self.individual_clouds_output_path, individual_sfm_dir, cluster_name)
                if not os.path.exists(cluster_output_dir):
                    os.makedirs(cluster_output_dir)
                cluster_metashape_data.to_csv(
                    os.path.join(cluster_output_dir, "metashape_metadata.csv"),
                    index=False
                )

    def _process_individual_clouds(self):
        print('Processing individual clouds...')
        for cluster_dir in glob.glob(
            os.path.join(self.individual_clouds_output_path,"**/cluster[0-9]*"),
            recursive=True
        ):
            try:
                metadata_file = os.path.join(cluster_dir, "metashape_metadata.csv")
                print("\n\n")
                print(f"Running pipeline for single date and cluster: {cluster_dir}")
                print(f"Using metashape metadata in file: {metadata_file}")
                pipeline = hsfm.pipeline.Pipeline(
                    self.raw_images_directory,
                    self.reference_dem_hires,
                    self.image_matching_accuracy,
                    self.densecloud_quality,
                    self.output_DEM_resolution,
                    "project",
                    cluster_dir,
                    metadata_file,
                    camera_models_path = self.camera_calibration_directory,
                    license_path=self.license_path,
                )
                    
                updated_cameras = pipeline.run_multi(iterations=2)
                updated_cameras=None
                print(f"Final updated cameras for {cluster_dir}: {updated_cameras} ")
            
            except Exception as e:
                print(f'Failure processing individual clouds at {cluster_dir}: \n {e}')

def parse_args():
    parser = argparse.ArgumentParser(
    """[summary]
    Run the HSFM Timesift pipeline for any set of images.
    Provide a csv file formatted as Metashape metadata and a path to image files.
    """
    )
    parser.add_argument(
        "--metashape-metadata-file",
        help = """
        A CSV dataset containing image file name and Metashape relevant information. The first 13 columns can be named anything, but must 
        contain specific information in the following order, per the "nxyzXYZabcABC" order of columns documented in the Metashape python
        API documentation:
            n - label (file name, with extension)
            x - x coordinate (ie longitude)
            y - y coordinate (ie latitude)
            z - z coordinate (ie altitude)
            X - x coordinate accuracy (in meters)
            Y - y coordicate accuracy (in meters)
            Z - z coordinate accuracy (in meters)
            a - a (yaw) rotation angle
            b - b (pitch) rotation angle
            c - c (roll) rotation anble
            A - a (yaw) rotation angle accuracy (in degrees)
            B - b (pitch) rotation angle accuracy (in degrees)
            C - c (roll) rotation angle accuracy (in degrees)
        After these columns, there must be two columns named "focal_length" and "pixel_pitch" containing the correct 
        information for each image.
        """
    )
    parser.add_argument(
        "--image-metadata-file",
        help = """
        A CSV dataset containing image file name and date-of-capture information. 
        Must have the following columns:
            [["image_file_name", "Year", "Month", "Day"]]
        """
    )
    parser.add_argument(
        "--raw-images-directory",
        help = """
        A path to a directory containing raw images ready to be processed with Metashape. 
        Image dimensions must be square.
        """
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        help="Directory path where pipeline outputs will be stored.",
        required=True,
    )
    parser.add_argument(
        "--reference-dem-lowres",
        help="Path to reference dem used to align the timesift (4D bundle adjustment) bundle adjusted point cloud .",
        required=True,
    )
    parser.add_argument(
        "--reference-dem-hires",
        help="Path to reference dem used to align the timesift (4D bundle adjustment) bundle adjusted point cloud .",
        required=True,
    )
    parser.add_argument(
        "-q",
        "--densecloud-quality",
        help="Densecloud quality parameter for Metashape. Values include 1 - 4, from highest to lowest quality.",
        type=int,
        default=2,
    )
    parser.add_argument(
        "-a",
        "--image-matching-accuracy",
        help="Image matching accuracy parameter for Metashape. Values include 1 - 4, from highest to lowest quality.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-s",
        "--output-resolution",
        help="Output DEM target resolution",
        required=True,
        type=float,
    )
    parser.add_argument(
        "-l", "--license-path", help="Path to Agisoft license file", required=True
    )
    parser.add_argument(
        "-p",
        "--parallelization",
        help="Number of parallel processes to spawn. Parallelization only happens when individual (single epoch) dense clouds are being processed.",
        default=2,
        type=int,
    )
    
    return parser.parse_args()

def main():
    print("Parsing arguments...")
    args = parse_args()
    print(f"Arguments: \n\t {vars(args)}")
    
    print("fPerforming timesift processing...")
    
    timesift_pipeline = TimesiftPipeline(
        args.metashape_metadata_file,
        args.image_metadata_file,
        args.raw_images_directory,
        args.output_directory,
        args.reference_dem_lowres,
        args.reference_dem_hires,
        densecloud_quality=args.densecloud_quality,
        image_matching_accuracy=args.image_matching_accuracy,
        output_DEM_resolution=args.output_resolution,
        license_path=args.license_path,
        parallelization=args.parallelization
    )

    _ = timesift_pipeline.run()

if __name__ == "__main__":
    main()