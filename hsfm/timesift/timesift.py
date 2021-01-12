import hsfm
from hsfm.pipeline import Pipeline
import hipp
from pathlib import Path
import pandas as pd
import os
from joblib import Parallel, delayed
import multiprocessing
import argparse


class NAGAPTimesiftPipeline:
    """
    Timesift Historical Structure from Motion pipeline.
    Generates a timeseries of DEMs for a given set of images that span multiple years.

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
        pixel_pitch=0.02,
        license_path="uw_agisoft.lic",
        parallelization=1,
    ):
        """Initialize car attributes."""
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
        self.pixel_pitch = pixel_pitch
        self.license_path = license_path
        self.parallelization = parallelization

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

    def run(self):
        """Run the full pipeline.
        1. Download, preprocess (cropping, principal point) raw NAGAP images, and
            use hsfm.pipeline.Pipeline to generate and align a multi-epoch densecloud.
        2. Generate image clusters and organize directories for the next step. Images
            are clustered by date (day) and roll.
        3. For each date, use hsfm.pipeline.Pipeline to generate and align a multi-epoch
            densecloud.
        """
        _ = __download_and_preprocess_images()
        metadata_original_file = __prepare_metashape_metadata_file()
        metadata_timesift_aligned_file = __generate_multi_epoch_densecloud(
            metadata_original_file
        )
        _ = __prepare_individual_clouds_data(
            metadata_original_file, metadata_timesift_aligned_file
        )
        _ = __process_all_individual_clouds()

    def __download_and_preprocess_images(self):
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

            count_raw_images = len(list(Path(raw_image_dir).rglob("*.tif")))
            count_preprocessed_images = len(
                list(Path(self.preprocessed_images_path).rglob("*.tif"))
            )
            print(
                f"Preprocessed {count_preprocessed_images} of {count_raw_images} raw images."
            )

    def __prepare_metashape_metadata_file(self):
        print("Preparing Metashape metadata CSV file...")
        hsfm.core.prepare_metashape_metadata(
            self.selected_images_df,
            output_directory=self.output_directory,
        )
        return os.path.join(
            output_directory, "metashape_metadata.csv"
        )  # this is the file created in the above method...annoying I have to recreate it

    def __generate_multi_epoch_densecloud(self, metadata_file):
        print("Generating and aligning multi-epoch densecloud...")
        pipeline = hsfm.pipeline.Pipeline(
            self.preprocessed_images_path,
            self.reference_dem,
            self.pixel_pitch,
            self.image_matching_accuracy,
            self.densecloud_quality,
            self.output_DEM_resolution,
            "multi_epoch_densecloud",
            self.multi_epoch_cloud_output_path,
            metadata_file,
            license_path=self.license_path,
        )
        return pipeline.run()

    # This has a lot of annoying data manipulation that could be avoided with better handling/updating
    # of camera position.
    def __prepare_individual_clouds_data(
        self, original_cameras_file, aligned_cameras_file
    ):
        print("Preparing data for individual clouds...")
        aligned_cameras_df = pd.read_csv(aligned_cameras_file)
        og_data_df = pd.read_csv(original_cameras_file, index_col=0)
        og_data_df = og_data_df[["fileName", "Year", "Month", "Day"]]
        og_data_df["image_file_name"] = og_data_df["fileName"] + ".tif"
        joined_df = pd.merge(og_data_df, aligned_cameras_df, on="image_file_name")
        joined_df["Month"] = joined_df["Month"].fillna("0")
        joined_df["Day"] = joined_df["Day"].fillna("0")
        datestrings_and_dfs = [
            ("_".join(date_tuple), df)
            for date_tuple, df in grouped_by_dayjoined_df.groupby(
                ["Year", "Month", "Day"]
            )
        ]
        daily_dir_names = []
        for date_string, df in datestrings_and_dfs:
            # Drop unncessary-for-processing columns (we only needed them to separate by year)
            df = df.drop(["fileName", "Year", "Month", "Day"], axis=1)
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

    def __process_all_individual_clouds(self):
        print("Processing all individual clouds...")
        individual_sfm_dirs = os.listdir(self.individual_clouds_output_path)
        results = Parallel(n_jobs=self.parallelization)(
            delayed(__process_image_batch)(i) for i in individual_sfm_dirs
        )
        return results

    def __process_image_batch(self, individual_sfm_dir):
        print(f"\tProcessing image batch for    date {individual_sfm_dir}")
        output_path = os.path.join(
            self.individual_clouds_output_path, individual_sfm_dir
        )
        input_images_metadata_file = os.path.join(output_path, "metashape_metadata.csv")
        pipeline = Pipeline(
            self.preprocessed_images_path,
            self.reference_dem,
            self.pixel_pitch,
            self.image_matching_accuracy,
            self.densecloud_quality,
            self.output_DEM_resolution,
            input_sfm_dir,
            output_path,
            input_images_metadata_file,
            license_path=self.license_path,
        )
        return pipeline.run_multi()


########################################################################################
########################################################################################
#
# App code
# Run like this
#
#   nohup python hsfm/timesift/timesift.py \
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
    )
    parser.add_argument(
        "-t",
        "--templates-dir",
        help="Path to directory containing NAGAP fiducial marker proxys. Contained in the hipp python package.",
    )
    parser.add_argument(
        "-b",
        "--bounds",
        help="Bounds for selecting images to process. Formatted as a list (ULLON, ULLAT, LRLON, LRLAT). ",
    )
    parser.add_argument(
        "-m",
        "--nagap-metadata-csv",
        help="Path to NAGAP metadata csv file. Contained in the hipp python package.",
    )
    parser.add_argument("-r", "--reference-dem", help="Path to reference dem.")
    parser.add_argument(
        "-q",
        "--densecloud-quality",
        help="Densecloud quality parameter for Metashape. Values include 1 - 4, from highest to lowest quality.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-a",
        "--image-matching-accuracy",
        help="Image matching accuracy parameter for Metashape. Values include 1 - 4, from highest to lowest quality.",
        required=True,
        type=int,
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
        "-l", "--license-path", help="Path to Agisoft license file", required=False
    )
    parser.add_argument(
        "-p",
        "--parallelization",
        help="Number of parallel processes to spawn. Parallelization only happens when individual (single epoch) dense clouds are being processed.",
    )


def main():
    print("Parsing arguments...")
    args = __parse_args()
    print(f"Arguments: \n\t {args}")
    bounds_tuple = tuple(args.bounds)
    assert (len(bounds_tuple) == 4, "Bounds did not have 4 numbers.")
    timesift_pipeline = hsfm.timesift.NAGAPTimesiftPipeline(
        args.output_path,
        args.teplates_dir,
        bounds_tuple,
        args.nagap_metadata_csv,
        args.reference_dem,
        densecloud_quality=args.densecloud_quality,
        image_matching_accuracy=args.image_matching_accuracy,
        output_DEM_resolution=args.output_DEM_resolution,
        pixel_pitch=args.pixel_pitch,
        license_path=args.license_path,
        parallelization=args.parallelization,
    )
    final_camera_metadata_list = timesift_pipeline.run()
    print(
        f"Final updated camera metadata files at paths:\n\t{final_camera_metadata_list}"
    )


if __name__ == "__main__":
    main()
