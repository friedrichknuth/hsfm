from pathlib import Path
import pandas as pd
import os
import argparse
import glob

from hsfm.pipeline.pipeline import Pipeline
import hsfm.metashape
import hsfm.asp
import hipp.dataquery
import hipp.batch


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
                
        metadata_timesift_aligned_file, unaligned_cameras_df = self._generate_multi_epoch_densecloud()
        # Do something with the unaligned cameras!!!
        _ = self._save_image_footprints()
        _ = self._export_camera_calibration_files()
        _ = self._prepare_single_date_data(metadata_timesift_aligned_file)
        dict_of_subsets_by_date = self._find_clusters_in_individual_clouds()
        _ = self._generate_subsets_for_each_date(dict_of_subsets_by_date)
        _ = self._process_individual_clouds()

    # ToDo must check iteratively - determine clusters in this step too
    def _generate_multi_epoch_densecloud(self):
        print("Generating and aligning multi-epoch densecloud...")
        # It makes sense not to make these parameters configurable form the immediate interface,
        # but hard coding them is bad too. Maybe a configuration file would be useful at this point... 
        # so many parameters,
        pipeline = Pipeline(
            self.raw_images_directory,
            self.reference_dem_lowres,
            2,
            4,
            10,
            self.multi_epoch_project_name,
            self.multi_epoch_cloud_output_path,
            self.metashape_metadata_file,
            license_path=self.license_path,
        )
        nuthed_aligned_bundle_adjusted_metadata_file, unaligned_cameras_df = pipeline.run(export_orthomosaic=False)

        if len(unaligned_cameras_df) > 2:
            unaligned_cams_file = nuthed_aligned_bundle_adjusted_metadata_file.replace('nuth_aligned_bundle_adj_metadata.csv', 'metaflow_bundle_adj_unaligned_metadata.csv')
            print(f"Unaligned cameras count more than 2, running a pipeline with unaligned images with metadata in file {unaligned_cams_file}")
            pipeline = Pipeline(
                self.raw_images_directory,
                self.reference_dem_lowres,
                2,
                4,
                10,
                self.multi_epoch_project_name,
                os.path.join(self.multi_epoch_cloud_output_path, 'unaligned_retry'),
                unaligned_cams_file,
                license_path=self.license_path,
            )
            nuthed_aligned_bundle_adjusted_metadata_file_2, unaligned_cameras_df_2 = pipeline.run(export_orthomosaic=False)
            if len(unaligned_cameras_df_2) > 0:
                print(f"Some unaligned cameras remain ({len(unaligned_cameras_df_2)})")
        
        combined_nuthed_aligned_bundle_adjusted_metadata_df = pd.concat([
            pd.read_csv(nuthed_aligned_bundle_adjusted_metadata_file), 
            pd.read_csv(nuthed_aligned_bundle_adjusted_metadata_file_2)
        ])
        combined_nuthed_aligned_bundle_adjusted_metadata_file = os.path.join(self.multi_epoch_cloud_output_path, 'unaligned_retry', 'all_aligned_cameras_metadata.csv')
        combined_nuthed_aligned_bundle_adjusted_metadata_df.to_csv(
            combined_nuthed_aligned_bundle_adjusted_metadata_file, 
            index=False
        )
        return combined_nuthed_aligned_bundle_adjusted_metadata_file, unaligned_cameras_df_2

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
        image_metadata_df = image_metadata_df[['fileName', 'Year', 'Month', 'Day']]
        image_metadata_df['image_file_name'] = image_metadata_df['fileName'] + '.tif'
        joined_df = pd.merge(
            image_metadata_df, aligned_cameras_df, on="image_file_name"
        )
        joined_df["Year"] = joined_df["Year"].astype('str')
        joined_df["Month"] = joined_df["Month"].astype('str').fillna("0")
        joined_df["Day"] = joined_df["Day"].astype('str').fillna("0")
        
        daily_dir_names = []
        # ToDo may fail here if day is Nan
        for date_tuple, df in joined_df.groupby(["Year", "Month", "Day"]):
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
        
    #ToDo make the clustering recursive until no more than 2 unaligned cameras - should be a separate function
    def _find_clusters_in_individual_clouds(self):
        print("Searching all dates for clusters/subsets")
        individual_dir_to_subset_list_dict = {}
        all_individual_sfm_dirs = os.listdir(self.individual_clouds_output_path)
        for individual_sfm_dir in all_individual_sfm_dirs:
            list_of_subsets = []
            try:
                print(f"Processing single date ({individual_sfm_dir}) images to check for clusters/subsets ...")
                output_path = os.path.join(self.individual_clouds_output_path, individual_sfm_dir, "cluster_metashape_run")
                input_images_metadata_file = os.path.join(self.individual_clouds_output_path, individual_sfm_dir, "metashape_metadata.csv")
                #TODO DON't HARDCODE THESE, but also don't take the parameters from the arguments...generally too high
                metashape_project_file, point_cloud_file = hsfm.metashape.images2las(
                    individual_sfm_dir,
                    self.raw_images_directory,
                    input_images_metadata_file,
                    output_path,
                    focal_length            = pd.read_csv(input_images_metadata_file)['focal_length'].iloc[0],
                    image_matching_accuracy = 2,
                    densecloud_quality      = 4,
                    keypoint_limit          = 80000,
                    tiepoint_limit          = 8000,
                    rotation_enabled        = True,
                    export_point_cloud      = False
                )
                print("SfM processing completed. ")
                print("Updating Cameras.")
                ba_cameras_df, unaligned_cameras_df = hsfm.metashape.update_ba_camera_metadata(metashape_project_file, input_images_metadata_file)
                # ToDo: Come up with a cleaner way of rerunning images2las with the unaligned cameras
                # This should really act recursively, but after the second try, still-unaligned cameras are left to rot
                ba_cameras_metadata_file_path = input_images_metadata_file.replace("metashape_metadata.csv", "single_date_multi_cluster_bundle_adjusted_metashape_metadata.csv")
                unaligned_cameras_metadata_file_path = input_images_metadata_file.replace("metashape_metadata.csv", "single_date_multi_cluster_bundle_adjusted_unaligned_metashape_metadata.csv")
                ba_cameras_df.to_csv(ba_cameras_metadata_file_path, index=False)
                unaligned_cameras_df.to_csv(unaligned_cameras_metadata_file_path, index=False)

                print("Determining clusters.")
                clusters_intial_attempt = hsfm.metashape.determine_clusters(metashape_project_file)
                list_of_subsets = list_of_subsets + clusters_intial_attempt
                
                if len(unaligned_cameras_df) > 2:
                    print("More than 2 unaligned cameras left.")
                    print("Processing unaligned cameras.")
                    output_path_2 = output_path.replace('cluster_metashape_run', 'cluster_metashape_run2')
                    metashape_project_file_2, point_cloud_file_2 = hsfm.metashape.images2las(
                        individual_sfm_dir,
                        self.raw_images_directory,
                        unaligned_cameras_metadata_file_path,
                        output_path_2,
                        image_matching_accuracy = 2,
                        densecloud_quality      = 4,
                        keypoint_limit          = 80000,
                        tiepoint_limit          = 8000,
                        rotation_enabled        = True,
                        export_point_cloud      = False
                    )
                    print("SfM processing completed for unaligned cameras. ")
                    print("Updating cameras for unaligned cameras.")
                    ba_cameras_df_2, unaligned_cameras_df_2 = hsfm.metashape.update_ba_camera_metadata(metashape_project_file_2, unaligned_cameras_metadata_file_path)
                    
                    ba_cameras_metadata_file_path_2 = input_images_metadata_file.replace("metashape_metadata.csv", "single_date_multi_cluster_bundle_adjusted_metashape_metadata2.csv")
                    unaligned_cameras_metadata_file_path_2 = input_images_metadata_file.replace("metashape_metadata.csv", "single_date_multi_cluster_bundle_adjusted_unaligned_metashape_metadata2.csv")
                    ba_cameras_df_2.to_csv(ba_cameras_metadata_file_path_2, index=False)
                    unaligned_cameras_df_2.to_csv(unaligned_cameras_metadata_file_path_2, index=False)  
                    
                    print("Determining clusters for unaligned cameras.")   
                    clusters_second_attempt = hsfm.metashape.determine_clusters(metashape_project_file_2)
                    list_of_subsets = list_of_subsets + clusters_second_attempt
            except Exception as e:
                print(f'Failure processing/finding clusters in individual clouds for cloud {individual_sfm_dir}: \n {e}')
            print("Writing subsets to file/")
            with open(
                input_images_metadata_file.replace("metashape_metadata.csv", "subsets.txt"), 
                'w'
            ) as f:
                f.write(str(list_of_subsets))
            individual_dir_to_subset_list_dict[individual_sfm_dir] =  list_of_subsets
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

    def _process_individual_clouds(self, date_keys=[], output_DEM_resolution_override=None):
        print('Processing individual clouds...')
        all_dates_folders = glob.glob(
            os.path.join(self.individual_clouds_output_path,"**/cluster[0-9]*"),
            recursive=True
        )
        if date_keys:
            def flatten_list_of_lists(ls):
                return [item for sublist in ls for item in sublist]
            valid_cluster_dates = flatten_list_of_lists(
                [[f for f in all_dates_folders if date_key in f] for date_key in date_keys]
            )
        else:
            valid_cluster_dates = all_dates_folders

        for cluster_dir in valid_cluster_dates:
            try:
                metadata_file = os.path.join(cluster_dir, "metashape_metadata.csv")
                assert len(pd.read_csv(metadata_file)) > 2, "Skipping cluster because not enough images."
                print("\n\n")
                print(f"Running pipeline for single date and cluster: {cluster_dir}")
                print(f"Using metashape metadata in file: {metadata_file}")
                final_output_DEM_resolution = output_DEM_resolution_override if output_DEM_resolution_override else self.output_DEM_resolution
                pipeline = Pipeline(
                    self.raw_images_directory,
                    self.reference_dem_hires,
                    self.image_matching_accuracy,
                    self.densecloud_quality,
                    final_output_DEM_resolution,
                    "project",
                    cluster_dir,
                    metadata_file,
                    camera_models_path = self.camera_calibration_directory,
                    license_path=self.license_path,
                )
                    
                updated_cameras_file, unaligned_cameras_df = pipeline.run_multi()
                # ToDo what to do with unaligned cameras df? Run another pipeline if more than two cameras!?
                print(f"Final updated cameras for {cluster_dir}: {updated_cameras_file} ")
                print(f"There are {len(unaligned_cameras_df)} unaligned cameras remaining.")
                updated_cameras=None
            
            except Exception as e:
                print(f'Failure processing individual clouds at {cluster_dir}: \n {e}')
  
    def create_results_report(self, iteration=0):
        """Read the nuth-aligned results of each date-cluster. Gather NMAD data and report.
        """
        # ToDo: don't use subprocess here...no guarantee that imagemagick was installed!
        import subprocess
        results_report_file = os.path.join(self.output_directory, 'individual_clouds', 'results.pdf')
        qc_files = glob.glob(os.path.join(self.output_directory, f'individual_clouds/**/{iteration}/**/*align.png'), recursive=True)
        print(f'Found {len(qc_files)} align.png files')
        subprocess.call(
            ["convert"] + qc_files + [results_report_file]
        )
        return results_report_file

    def process_dem_align_all_intermediate_steps(self):
        """
        Run dem_align.py on all intermediate point cloud alignment products (point2plane, spoint2point, spoint2point-bareground).
        

        Returns:
            _type_: _description_
        """
        return None
    
    def process_final_orthomosaics(self, date_cluster_bests, iteration = 0):
        """
        Create "final" aligned mosaics according to a manually approved list of good DEMs. Has the side effect of creating
        2 files, both of which are saved to the dem_align.py output directory identified by the input date_cluster_bests 
        dictionary. The generated files are:
            "final_metashape_metadata.csv" - Camera metadata transformed by the identified (best) PC alignment and by the nuth and kaab alignment
            "**align_orthomosaic.tif" - Orthomosaic output that is aligned by PC alignment transform and the nuth and kaab alignment. This
                                        orthomosaic is in alignment with the final DEM output by nuth and kaab. The filename is similar to the name
                                        of the aligned dem output file of dem_align.py, but "*align.tif" is replaced with "*align_orthomosaic.tif"
                                            
        For each year-cluster (if included in manually approved list):
        1. Identify the transform file of the selected best pc_align step
        2. Apply this transform to the bundle adjusted cameras.
        3. Apply the nuth and kaab transform to the now-aligned cameras.
        4. Export orthomosaic using the now-pc-aligned-and-nuth-aligned cameras


        date_cluster_bests (dict): Dictionary of strings pointing to strings, indicating for each date-cluster, which 
        intermediate PC aligned product is best and should be used to generate a final orthomosaic. The keys
        should be paths to the cluster-date directory, ie ending in .../cluster[0-9] and the values should one of the 
        following: [
            "point2plane",
            "spoint2point",
            "spoint2point_bareground"
        ]
        Args:
            date_cluster_bests (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        for date_cluster, pc_align_prefix in date_cluster_bests.items():
            print(f"Processing {date_cluster}:")
            # 1. identify transform file and selected nuth_aligned output
                #path to the transform file from the transform that produced the best/manually identified DEM
            transform_file_full_path = os.path.join(
                date_cluster,
                f"{iteration}/pc_align",
                pc_align_prefix + "-transform.txt"
            )
            print(f"\t1/4: Identified transform file: {transform_file_full_path}")
                #path to the dem_align.py output file that was run for the transform that produced the best/manually identified DEM
            nuth_aligned_output_directory_full_path = os.path.join(
                date_cluster,
                f"{iteration}/pc_align",
                pc_align_prefix + "-trans_source-DEM_dem_align"
            )
                #path to the bundle adjusted metadata from the SfM run. These cameras are pre any transform.
            bundle_adj_cameras_csv_file = os.path.join(
                date_cluster,
                f"{iteration}/metaflow_bundle_adj_metadata.csv"
            )
                #new path to where the final metashape metadata will be saved
            final_aligned_cameras_csv_file = os.path.join(
                nuth_aligned_output_directory_full_path, 'final_metashape_metadata.csv'
            )
                #path to the metashape project file.
            metashape_project_file = os.path.join(date_cluster, str(iteration), 'project.psx')
            
                #list of files grabbed by globbing, should contain only one file path, the DEM output of dem_align.py 
            final_dem_file_list = glob.glob(
                os.path.join(nuth_aligned_output_directory_full_path, "*align.tif")
            )
            assert len(final_dem_file_list) == 1, f"More or less than one final DEM found (found {len(final_dem_file_list)}). {final_dem_file_list}"
            final_dem_file = final_dem_file_list[0]

                #new path to where we will save the aligned orthomosaic
            final_orthomosaic_file = final_dem_file.replace("align.tif", "align_orthomosaic.tif")

            print(f"\t2/4: Applying PC align transform to bundle-adjusted cameras.")
            # 2. apply transform to bundle adjusted cameras
            print("Applying PC alignment transform to bundle-adjusted camera positions...")
            pc_aligned_cameras_df = hsfm.core.metadata_transform(
                bundle_adj_cameras_csv_file,
                transform_file_full_path,
                output_file_name=None #no files will be written
            )

            print(f"\t3/4: Applying nuth and kaab transform to PC-aligned-and-bundle-adjusted-cameras.")
            # 3. apply nuth and kaab transform
            hsfm.utils.apply_nuth_transform_to_camera_metadata(
                pc_aligned_cameras_df,
                nuth_aligned_output_directory_full_path,
                final_aligned_cameras_csv_file
            )
            
            print(f"\t4/4: Exporting final orthomosaic to {final_orthomosaic_file}.")
            # 4. export final orthomosaic          
            hsfm.metashape.export_updated_orthomosaic(
                metashape_project_file,
                final_aligned_cameras_csv_file,
                final_orthomosaic_file
            )
        return None
    
    def mosaic_orthos(self, new_file_path, file_list):
        import subprocess
        subprocess.call(['gdalbuildvrt', 'MergedImage.vrt'] + file_list)
        subprocess.call(['gdal_translate', '-of', 'GTiff', '-co', 'TILED=YES', 'MergedImage.vrt', new_file_path])
        subprocess.call(['gdal_translate', '-of', 'GTiff', '-tr', '0.000011', '0.000011', '-co', 'TILED=YES', 'MergedImage.vrt', new_file_path.replace('.tif', '_lowres.tif')])
        os.remove('MergedImage.vrt')
        return None
    
    def mosaic_dems(self, new_file_path, file_list, make_lowres = False, threads = 32):
        hsfm.asp.dem_mosaic(
            new_file_path,
            file_list,
            threads=32
        )
        if make_lowres:
            import subprocess
            subprocess.call(['gdalwarp', '-r', 'cubic', '-overwrite', '-tr', '5', '5', '-co', 'TILED=YES', new_file_path, new_file_path.replace('.tif', '_lowres.tif')])
        return None

    def process_selected_dems_into_mosaics(self, date_cluster_bests):
        """Identifies the good DEMs by those that have an accompanied orthomoasic named as in the function above.
        """
        date_to_files_dict = {
            file.split('individual_clouds/')[1].split('/')[0]: {
                'dem' : [],
                'dod': [],
                'ortho': []
            } for file in date_cluster_bests.keys()
        }
        for date_cluster, pc_align_prefix in date_cluster_bests.items():
            date_key = date_cluster.split('individual_clouds/')[1].split('/')[0]
            
            ortho_file_path = glob.glob(os.path.join(date_cluster, "**", "*align_orthomosaic.tif"), recursive=True)
            assert len(ortho_file_path) == 1, f"length was {len(ortho_file_path)}"
            ortho_file_path = ortho_file_path[0]
            
            dem_file_path = ortho_file_path.replace("align_orthomosaic.tif", "align.tif")
            dod_file_path = ortho_file_path.replace("align_orthomosaic.tif", "align_diff.tif")

            date_to_files_dict[date_key]['dem'].append(dem_file_path)
            date_to_files_dict[date_key]['dod'].append(dod_file_path)
            date_to_files_dict[date_key]['ortho'].append(ortho_file_path)

        #For each date, create mosaics for each data type
        for k, file_list in date_to_files_dict.items():
            print(f"Mosaicing products for {k}")
            mosaiced_dem_fn = file_list['dem'][0].split('cluster')[0] + 'dem.tif'
            mosaiced_dod_fn = file_list['dem'][0].split('cluster')[0] + 'dod.tif'
            mosaiced_ortho_fn = file_list['dem'][0].split('cluster')[0] + 'orthomosaic.tif'
            self.mosaic_dems(
                mosaiced_dem_fn,
                file_list['dem'],
                make_lowres=True
            )
            self.mosaic_dems(
                mosaiced_dod_fn, 
                file_list['dod']
            )
            self.mosaic_orthos(
                mosaiced_ortho_fn,
                file_list['ortho']    
            )
            



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