import hsfm
from hsfm.pipeline import Pipeline
import hipp
from pathlib import Path
import pandas as pd
import os
from joblib import Parallel, delayed
import argparse
import functools

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
        exclude_years=None
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
        if exclude_years:
            print(f"Excluding images from years {exclude_years}")
            og_image_num = len(self.selected_images_df)
            self.selected_images_df = self.selected_images_df[
                ~self.selected_images_df.Year.isin(exclude_years)
            ]
            final_image_num = len(self.selected_images_df)
            print(f"Removed {og_image_num - final_image_num} images.")

    def run(self):
        """Run the full pipeline.
        1. Download, preprocess (cropping, principal point) raw NAGAP images, and
            use hsfm.pipeline.Pipeline to generate and align a multi-epoch densecloud.
        2. Generate image clusters and organize directories for the next step. Images
            are clustered by date (day) and roll.
        3. For each date, use hsfm.pipeline.Pipeline to generate and align a multi-epoch
            densecloud.
        """
        _ = self.__download_and_preprocess_images()
        metadata_original_file = self.__prepare_metashape_metadata_file()
        metadata_timesift_aligned_file = self.__generate_multi_epoch_densecloud(
            metadata_original_file
        )
        _ = self.__prepare_individual_clouds_data(
            self.selected_images_df, metadata_timesift_aligned_file
        )
        final_camera_location_files = self.__process_all_individual_clouds()
        return final_camera_location_files

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

    def __prepare_metashape_metadata_file(self):
        print("Preparing Metashape metadata CSV file...")
        hsfm.core.prepare_metashape_metadata(
            self.selected_images_df,
            output_directory=self.output_directory,
        )
        return os.path.join(
            self.output_directory, "metashape_metadata.csv"
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
        self, original_cameras_df, aligned_cameras_file
    ):
        """Create a CSV file of image metadata for each date.
        """
        print("Preparing data for individual clouds...")
        aligned_cameras_df = pd.read_csv(aligned_cameras_file)
        original_cameras_df = original_cameras_df[["fileName", "Year", "Month", "Day"]]
        original_cameras_df["image_file_name"] = original_cameras_df["fileName"] + ".tif"
        joined_df = pd.merge(original_cameras_df, aligned_cameras_df, on="image_file_name")
        joined_df["Month"] = joined_df["Month"].fillna("0")
        joined_df["Day"] = joined_df["Day"].fillna("0")
        datestrings_and_dfs = [
            ("_".join(date_tuple), df)
            for date_tuple, df in joined_df.groupby(
                ["Year", "Month", "Day"]
            )
        ]
        daily_dir_names = []
        for date_string, df in datestrings_and_dfs:
            # Drop unncessary-for-processing columns (we only needed them to separate by year)
            df = df.drop(["fileName", "Year", "Month", "Day"], axis=1)
            # Put columns in proper order
            df = df[["image_file_name","lon","lat","alt","lon_acc","lat_acc","alt_acc","yaw","pitch","roll","yaw_acc","pitch_acc","roll_acc","focal_length"]]
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
        process_image_batch_partial = functools.partial(
            process_image_batch,
            preprocessed_images_path = self.preprocessed_images_path,
            reference_dem = self.reference_dem,
            pixel_pitch = self.pixel_pitch,
            image_matching_accuracy = self.image_matching_accuracy,
            densecloud_quality = self.densecloud_quality,
            output_DEM_resolution = self.output_DEM_resolution,
            individual_clouds_output_path = self.individual_clouds_output_path,
            license_path = self.license_path
        )
        results = Parallel(n_jobs=self.parallelization)(
            delayed(process_image_batch_partial)(i) for i in individual_sfm_dirs
        )
        return results

def process_image_batch(
    individual_sfm_dir,
    preprocessed_images_path,
    reference_dem,
    pixel_pitch,
    image_matching_accuracy,
    densecloud_quality,
    output_DEM_resolution,
    individual_clouds_output_path,
    license_path
):
    """Wrapper function to create a Pipeline and call run_multi for a single-date 
    batch of images. It is outside of the NAGAPTimesiftPipeline class because 
    we call it in parallel.

    ToDo: Put this in a better place.
    """
    print(f"\tProcessing image batch for    date {individual_sfm_dir}")
    output_path = os.path.join(
        individual_clouds_output_path, individual_sfm_dir
    )
    input_images_metadata_file = os.path.join(output_path, "metashape_metadata.csv")
    pipeline = Pipeline(
        preprocessed_images_path,
        reference_dem,
        pixel_pitch,
        image_matching_accuracy,
        densecloud_quality,
        output_DEM_resolution,
        individual_sfm_dir,
        output_path,
        input_images_metadata_file,
        license_path=license_path,
    )
    return pipeline.run_multi()

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
        help="Path to reference dem.",
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
        type=int
    )
    parser.add_argument(
        "-e",
        "--exclude-years",
        help="List of years you want to exclude from the processing. Useful if you know images from certain years are bad. Write 2 digit numbers i.e. for 1977, write 77.",
        nargs="+"
    )
    return parser.parse_args()


def main():
    print("Parsing arguments...")
    args = __parse_args()
    print(f"Arguments: \n\t {vars(args)}")
    bounds_tuple = tuple(args.bounds)
    assert len(bounds_tuple) == 4, "Bounds did not have 4 numbers."
    timesift_pipeline = NAGAPTimesiftPipeline(
        args.output_path,
        args.templates_dir,
        bounds_tuple,
        args.nagap_metadata_csv,
        args.reference_dem,
        densecloud_quality=args.densecloud_quality,
        image_matching_accuracy=args.image_matching_accuracy,
        output_DEM_resolution=args.output_resolution,
        pixel_pitch=args.pixel_pitch,
        license_path=args.license_path,
        parallelization=args.parallelization,
        exclude_years=args.exclude_years,
    )
    final_camera_metadata_list = timesift_pipeline.run()
    print(
        f"Final updated camera metadata files at paths:\n\t{final_camera_metadata_list}"
    )


if __name__ == "__main__":
    main()
