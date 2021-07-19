import hsfm
from hsfm.pipeline import Pipeline
import hipp
from pathlib import Path
import pandas as pd
import os
import argparse
import glob


class NAGAPTimesiftPipeline:
    """
    Timesift Historical Structure from Motion pipeline.
    Generates a timeseries of DEMs for a given set of images that span multiple years.
    Downloading and preprocessing of NAGAP images precedes the other timesift steps 
    (see TimesiftPipeline).

    """

    def __init__(
        self,
        output_directory,
        templates_dir,
        bounds,
        nagap_metadata_csv,
        reference_dem,
        densecloud_quality=2,
        image_matching_accuracy=1,
        output_DEM_resolution=2,
        license_path="uw_agisoft.lic",
        parallelization=1,
        exclude_years=None,
    ):
        self.output_directory = output_directory
        self.templates_dir = templates_dir  # this lives in the hipp library...
        self.bounds = bounds
        self.nagap_metadata_csv = (
            nagap_metadata_csv  # this lives in the hipp library...
        )
        self.reference_dem = reference_dem
        self.densecloud_quality = densecloud_quality
        self.image_matching_accuracy = image_matching_accuracy
        self.output_DEM_resolution = output_DEM_resolution
        self.license_path = license_path
        self.parallelization = parallelization

        self.multi_epoch_project_name = "multi_epoch_densecloud"
        # # Create a few variables
        self.preprocessed_images_path = os.path.join(
            output_directory, "preprocessed_images"
        )
        # Output directory for Multi Epoch Dense cloud generation
        self.multi_epoch_cloud_output_path = os.path.join(
            output_directory, "multi_epoch_cloud/"
        )
        self.individual_clouds_output_path = os.path.join(
            output_directory, "individual_clouds/"
        )
        self.selected_images_df = hipp.dataquery.NAGAP_pre_select_images(
            nagap_metadata_csv, bounds=self.bounds
        )
        if exclude_years:
            print(f"Excluding images from years {exclude_years}")
            og_image_num = len(self.selected_images_df)
            self.selected_images_df = self.selected_images_df[
                ~self.selected_images_df.Year.isin(exclude_years)
            ]
            final_image_num = len(self.selected_images_df)
            print(f"Removed {og_image_num - final_image_num} images.")

    def run(self, skip_preprocessing=False):
        """Run the full pipeline.
        1. Download, preprocess (cropping, principal point) raw NAGAP images, and
            use hsfm.pipeline.Pipeline to generate and align a multi-epoch densecloud.
        2. Generate image clusters and organize directories for the next step. Images
            are clustered by date (day) and roll.
        3. For each date, use hsfm.pipeline.Pipeline to generate and align a multi-epoch
            densecloud.
        """
        if not skip_preprocessing:
            metadata_original_file = self.__prepare_metashape_metadata_file()
            _ = self.__download_and_preprocess_images()
        else:
            metadata_original_file = os.path.join(
                self.output_directory, "metashape_metadata.csv" # see self.__prepare_metashape_metadata_file for where this file name comes from...not ideal
            )

        metadata_timesift_aligned_file = self.__generate_multi_epoch_densecloud(
            metadata_original_file
        )
        
        _ = self.__save_image_footprints()

        _ = self.__export_camera_calibration_files()
        
        _ = self.__prepare_single_date_data(
            self.selected_images_df, metadata_timesift_aligned_file
        )
        
        dict_of_subsets_by_date = self.__find_clusters_in_individual_clouds()
        
        _ = self.__generate_subsets_for_each_date(dict_of_subsets_by_date)


    def __prepare_metashape_metadata_file(self):
        print("Preparing Metashape metadata CSV file...")
        hsfm.core.prepare_metashape_metadata(
            self.selected_images_df,
            output_directory=self.output_directory,
        )
        return os.path.join(
            self.output_directory, "metashape_metadata.csv"
        )  # this is the file created in the above method...annoying I have to recreate the path name

    def __download_and_preprocess_images(self):
        """Iterates over images grouped by fiducial marker type, roll, and date. Downloads
        and preprocess images, preparing them for SfM processing.
        """
        print("Downloading and preprocessing images...")
        for (
            fiducial,
            roll,
            year,
            month,
            day,
        ), filtered_df in self.selected_images_df.groupby(
            ["fiducial_proxy_type", "Roll", "Year", "Month", "Day"], dropna=False
        ):

            roll_and_date_string = f"{roll}_{year}_{month}_{day}"
            raw_image_dir = os.path.join(
                self.output_directory, "raw_images", fiducial, roll_and_date_string
            )
            template_dir = os.path.join(self.templates_dir, fiducial)
            qc_df_output_directory = os.path.join(
                self.output_directory,
                "qc",
                fiducial,
                roll_and_date_string,
                "proxy_detection_data_frames",
            )
            qc_plots_output_directory = os.path.join(
                self.output_directory,
                "qc",
                fiducial,
                roll_and_date_string,
                "proxy_detection_plots",
            )

            print(
                f"Attempting to download {len(filtered_df)} images with fiducial proxy type {fiducial}, roll {roll}, and date {year}_{month}_{day}."
            )
            hipp.dataquery.NAGAP_download_images_to_disk(
                filtered_df, output_directory=raw_image_dir
            )
            num_images_downloaded = len(os.listdir(raw_image_dir))
            print(f"Succesfully downloaded {len(filtered_df)} images.")

            print("Preprocessing images...")
            hipp.batch.preprocess_with_fiducial_proxies(
                raw_image_dir,
                template_dir,
                output_directory=self.preprocessed_images_path,
                qc_df_output_directory=qc_df_output_directory,
                qc_plots_output_directory=qc_plots_output_directory,
                missing_proxy=None,
            )

    def __generate_multi_epoch_densecloud(self, metadata_file):
        print("Generating and aligning multi-epoch densecloud...")
        pipeline = hsfm.pipeline.Pipeline(
            self.preprocessed_images_path,
            self.reference_dem,
            self.image_matching_accuracy,
            self.densecloud_quality,
            self.output_DEM_resolution,
            self.multi_epoch_project_name,
            self.multi_epoch_cloud_output_path,
            metadata_file,
            license_path=self.license_path,
        )
        return pipeline.run(export_orthomosaic=False, split_in_blocks=True)

    def __save_image_footprints(self):
        """Creates geojson file with image footprints exported from the timesift Metashape project 
        file
        """
        gdf = hsfm.metashape.image_footprints_from_project(self.__get_timesift_project_path())
        gdf.to_file(
            os.path.join(self.output_directory, "timesifted_image_footprints.geojson"), 
            driver="GeoJSON"
        )


    #TODO there must be a better alternative to this...
    def __get_timesift_project_path(self):
        return os.path.join(self.multi_epoch_cloud_output_path, self.multi_epoch_project_name + ".psx")

    def __export_camera_calibration_files(self):
        import Metashape
        metashape_project_file = os.path.join(self.output_directory, 'multi_epoch_cloud', self.multi_epoch_project_name + ".psx")
        camera_exports_dir = os.path.join(self.output_directory, 'multi_epoch_cloud', 'camera_calibrations')
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
    def __prepare_single_date_data(
        self, original_cameras_df, aligned_cameras_file
    ):
        """Create a CSV file of image metadata for each date."""
        print("Preparing data for individual clouds...")
        aligned_cameras_df = pd.read_csv(aligned_cameras_file)
        original_cameras_df = original_cameras_df[["fileName", "Year", "Month", "Day"]]
        original_cameras_df["image_file_name"] = (
            original_cameras_df["fileName"] + ".tif"
        )
        joined_df = pd.merge(
            original_cameras_df, aligned_cameras_df, on="image_file_name"
        )
        joined_df["Month"] = joined_df["Month"].fillna("0")
        joined_df["Day"] = joined_df["Day"].fillna("0")
        
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
        

    def __find_clusters_in_individual_clouds(self):
        print("Searching all dates for clusters/subsets")
        individual_dir_to_subset_list_dict = {}
        for individual_sfm_dir in os.listdir(self.individual_clouds_output_path):
            try:
                print(f"Processing single date ({individual_sfm_dir}) images to check for clusters/subsets ...")
                output_path = os.path.join(self.individual_clouds_output_path, individual_sfm_dir, "cluster_metashape_run")
                input_images_metadata_file = os.path.join(self.individual_clouds_output_path, individual_sfm_dir, "metashape_metadata.csv")
                metashape_project_file, point_cloud_file = hsfm.metashape.images2las(
                    individual_sfm_dir,
                    self.preprocessed_images_path,
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

    def __generate_subsets_for_each_date(self, dict_of_subsets_by_date):
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


# def process_image_batch(
#     individual_sfm_dir,
#     preprocessed_images_path,
#     reference_dem,
#     image_matching_accuracy,
#     densecloud_quality,
#     output_DEM_resolution,
#     individual_clouds_output_path,
#     license_path,
# ):
#     """Wrapper function to create a Pipeline and call run_multi for a single-date
#     batch of images. It is outside of the NAGAPTimesiftPipeline class because
#     we call it in parallel.

#     ToDo: Put this in a better place.
#     """
#     print(f"\tProcessing image batch for    date {individual_sfm_dir}")
#     output_path = os.path.join(individual_clouds_output_path, individual_sfm_dir)
#     input_images_metadata_file = os.path.join(output_path, "metashape_metadata.csv")
#     pipeline = Pipeline(
#         preprocessed_images_path,
#         reference_dem,
#         image_matching_accuracy,
#         densecloud_quality,
#         output_DEM_resolution,
#         individual_sfm_dir,
#         output_path,
#         input_images_metadata_file,
#         license_path=license_path,
#     )
#     return pipeline.run_multi()


########################################################################################
########################################################################################
# App code
# Run like this (for bounds around Carbon Glacier)
#   python hsfm/timesift/timesift.py \
#       --output-path           /where/u/wanna/put/rainier_carbon_automated_timesift \
#       --templates-dir         /path/to/src/code/hipp/examples/fiducial_proxy_detection/input_data/fiducials/nagap/ \
#       --bounds                -121.7991 46.9902 -121.7399 46.8826 \
#       --nagap-metadata-csv        /path/to/src/code/hipp/examples/fiducial_proxy_detection/input_data/nagap_image_metadata.csv \
#       --densecloud-quality        2 \
#       --image-matching-accuracy   1 \
#       --output-resolution         1 \
#       --pixel-pitch               0.02 \
#       --parallelization           2 \
#       --exclude-years             77
#
########################################################################################
########################################################################################
def __parse_args():
    parser = argparse.ArgumentParser(
        "Run the HSFM Timesift pipeline for NAGAP images within a bounded area."
    )
    parser.add_argument(
        "-o",
        "--output-path",
        help="Path to directory where pipeline outputs will be stored.",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--templates-dir",
        help="Path to directory containing NAGAP fiducial marker proxys. Contained in the hipp python package.",
        required=True,
    )
    parser.add_argument(
        "-b",
        "--bounds",
        help="Bounds for selecting images to process. Formatted as a list of floats in order ULLON ULLAT LRLON LRLAT.",
        nargs="+",
        default=[],
        type=float,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--nagap-metadata-csv",
        help="Path to NAGAP metadata csv file. Contained in the hipp python package.",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--reference-dem",
        help="Path to reference dem used to align single-date point clouds,.",
        required=True,
    )
    parser.add_argument(
        "-r4d",
        "--reference-dem-4d",
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
        "-x",
        "--pixel-pitch",
        help="Pixel pitch/scanning resolution.",
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
    parser.add_argument(
        "-e",
        "--exclude-years",
        help="""
            List of years you want to exclude from the processing. Useful if you know images from
            certain years are bad. Write 2 digit numbers i.e. for 1977, write 77.
            """,
        nargs="+",
    )
    parser.add_argument(
        "--skip-preprocessing",
        help="""
            Skip preprocessing steps (downloading, preprocessing images), go straight to 
            multi-epoch densecloud creation.
            """,
        type=bool,
        default=False
    )
    parser.add_argument(
        "--only-process-4d",
        help="""
            Flag to only process the 4d timesift point cloud, to skip processing individual clouds
            after the 4d timesift point cloud is created and aligned. Defaults to false.
            """,
        type=bool,
        default=False
    )
    parser.add_argument(
        "--only-process-individual-clouds",
        help="""
            Flag to only process individual dates. It is expected that first part of the timesift
            pipeline has already been run with similar arguments (see the --only-process-4d). 
            Defaults to false.
            """,
        type=bool,
        default=False
    )
    parser.add_argument(
        "--use-timesift-calibrated-cameras",
        help="""
            Flag to use the timesift-calibrated camera models in the processing of individual date point clouds/elevation models. Defaults to True.
            """,
        type=bool,
        default=True
    )
    return parser.parse_args()

# To Do: make this a member function of the NAGAPTimesiftPipeline class
def process_individual_clouds(
    output_path,
    reference_dem,
    image_matching_accuracy,
    densecloud_quality,
    output_DEM_resolution,
    use_timesift_calibrated_cameras,
    license_path
):
    preprocessed_images_path = os.path.join(
            output_path, "preprocessed_images"
        )
    for cluster_dir in glob.glob(
        os.path.join(output_path,"individual_clouds/**/cluster[0-9]*"),
        recursive=True
    ):
        try:
            metadata_file = os.path.join(cluster_dir, "metashape_metadata.csv")
            print("\n\n")
            print(f"Running pipeline for single date and cluster: {cluster_dir}")
            print(f"Using metashape metadata in file: {metadata_file}")
            if use_timesift_calibrated_cameras: 
                pipeline = hsfm.pipeline.Pipeline(
                    preprocessed_images_path,
                    reference_dem,
                    image_matching_accuracy,
                    densecloud_quality,
                    output_DEM_resolution,
                    "project",
                    cluster_dir,
                    metadata_file,
                    camera_models_path = os.path.join(output_path, 'multi_epoch_cloud', 'camera_calibrations'),
                    license_path=license_path,
                )
            else:
                pipeline = hsfm.pipeline.Pipeline(
                    preprocessed_images_path,
                    reference_dem,
                    image_matching_accuracy,
                    densecloud_quality,
                    output_DEM_resolution,
                    "project",
                    cluster_dir,
                    metadata_file,
                    license_path=license_path,
                )
                
            updated_cameras = pipeline.run_multi(iterations=2)
            updated_cameras=None
            print(f"Final updated cameras for {cluster_dir}: {updated_cameras} ")
        
        except Exception as e:
            print(f'Failure processing individual clouds at {cluster_dir}: \n {e}')

def main():
    print("Parsing arguments...")
    args = __parse_args()
    print(f"Arguments: \n\t {vars(args)}")

    if not args.only_process_individual_clouds:   
        print("fPerforming timesift processing...")
        bounds_tuple = tuple(args.bounds)
        assert len(bounds_tuple) == 4, "Bounds did not have 4 numbers."
        timesift_pipeline = NAGAPTimesiftPipeline(
            args.output_path,
            args.templates_dir,
            bounds_tuple,
            args.nagap_metadata_csv,
            args.reference_dem_4d,
            densecloud_quality=args.densecloud_quality,
            image_matching_accuracy=args.image_matching_accuracy,
            output_DEM_resolution=args.output_resolution,
            license_path=args.license_path,
            parallelization=args.parallelization,
            exclude_years=args.exclude_years
        )
        _ = timesift_pipeline.run(args.skip_preprocessing)

    if not args.only_process_4d:
        print("fPerforming single-date processing...")
        process_individual_clouds(
            args.output_path,
            args.reference_dem,
            args.image_matching_accuracy,
            args.densecloud_quality,
            args.output_resolution,
            args.use_timesift_calibrated_cameras,
            args.license_path
    )      

if __name__ == "__main__":
    main()
