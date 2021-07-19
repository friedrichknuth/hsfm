import os
import argparse
import hipp
import hsfm


class NAGAPPreprocessingPipeline:
    """Perform preprocessing steps for NAGAP image datasets.

    Given an output directory, directory of fiducial templates, a csv file listing NAGAP image info,
    and a set of bounds, NAGAP images will be downloaded and preprocessed. 

    This pipeline outputs 3 crucial outputs into the specified output directory:
        1. image_metadata.csv (contains file names and date of image capture)
        2. metashape_metadata.csv (contains file names and columns required for Metashape processing)
        3. preprocessed_images/ (directory containing image files ready for Metashape processing)

    The 4 steps included in this pipeline are:
        1. Search the NAGAP image dataset for images queried with program arguments.
        2. Prepare a metashape_metadata csv file (necessary for Metashape processing)
            using the results of the query in step 1.
        3. Download raw images.
        4. Preprocess raw images (detect fiducial marker proxies, estimate principal point,
            prepare/crop images so they are ready to be used by Metashape)
    """

    def __init__(
        self,
        output_directory,
        fiducial_templates_directory,
        nagap_images_csv_path,
        bounds
    ):
        self.output_directory = output_directory
        self.fiducial_templates_directory = fiducial_templates_directory
        self.nagap_images_csv_path = nagap_images_csv_path
        self.bounds = bounds

        self.image_metadata_file = os.path.join(self.output_directory, "image_metadata.csv")
        # This file is created by the call in `run` to hsfm.core.prepare_metashape_metadata
        # which is annoying but unavoidable for now
        self.metashape_metadata_file = os.path.join(self.output_directory, "metashape_metadata.csv")

    def run(self):
        print('Filtering NAGAP image dataset with provided bounds...')
        selected_images_df = hipp.dataquery.NAGAP_pre_select_images(
            self.nagap_images_csv_path, bounds=self.bounds
        )
        selected_images_df.to_csv(self.image_metadata_file, index=False)
        print(f'Query results include {len(selected_images_df)} images.')

        print('Preparing Metashape metadata CSV file...')
        hsfm.core.prepare_metashape_metadata(selected_images_df, output_directory=self.output_directory)    
        print(f'Generated Metashape metadata file saved to path: {self.metashape_metadata_file}')

        print('Downloading and preprocessing images')
        _ = self.__download_and_preprocess_images(selected_images_df)

    def __download_and_preprocess_images(self, selected_images_df):
        preprocessed_images_directory = os.path.join(self.output_directory, "preprocessed_images")
        for (
            fiducial,
            roll,
            year,
            month,
            day,
        ), filtered_df in selected_images_df.groupby(
            ["fiducial_proxy_type", "Roll", "Year", "Month", "Day"], dropna=False
        ):

            roll_and_date_string = f"{roll}_{year}_{month}_{day}"
            raw_image_dir = os.path.join(
                self.output_directory, "raw_images", fiducial, roll_and_date_string
            )
            template_dir = os.path.join(self.fiducial_templates_directory, fiducial)
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
                output_directory=preprocessed_images_directory,
                qc_df_output_directory=qc_df_output_directory,
                qc_plots_output_directory=qc_plots_output_directory,
                missing_proxy=None,
            )

def __parse_args():
    parser = argparse.ArgumentParser("Run the NAGAP preprocessing pipeline.")
    parser.add_argument(
        "-o",
        "--output-directory",
        help="Path to directory where outputs will be stored.",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--fiducial-templates-directory",
        help="Path to directory containing NAGAP fiducial marker proxys. Many are contained in the HIPP python package.",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--nagap-images-csv-path",
        help="Path to NAGAP image database csv file. Contained in the hipp python package.",
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

def main():
    print("Parsing arguments...")
    args = __parse_args()
    print(f"Arguments: \n\t {vars(args)}")

    pipeline = NAGAPPreprocessingPipeline(
        args.output_directory,
        args.fiducial_templates_directory,
        args.nagap_images_csv_path,
        args.bounds
    )
    _ = pipeline.run()
    
if __name__ == "__main__":
    main()