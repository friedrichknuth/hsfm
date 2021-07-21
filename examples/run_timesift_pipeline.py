from hsfm.pipeline import TimesiftPipeline
# from hsfm.pipeline import NAGAPPreprocessingPipeline

# preprocess_pipeline = NAGAPPreprocessingPipeline(
#     output_directory,
#     fiducial_templates_directory = "/data2/elilouis/generate_ee_dems_baker/fiducials",
#     nagap_images_csv_path,
#     bounds
# )

# preprocess_pipeline.run()

# timesift_pipeline = TimesiftPipeline(
#     metashape_metadata_file = "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/combined_metashape_metadata.csv",
#     image_metadata_file = "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/combined_image_metadata.csv",
#     raw_images_directory = "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/preprocessed_images/",
#     output_directory = "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/",
#     reference_dem_lowres = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/baker_2015_utm_10m.tif',
#     reference_dem_hires = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/baker_2015_utm_m.tif',
#     image_matching_accuracy = 2,
#     densecloud_quality = 4,
#     output_DEM_resolution = 5,
#     license_path="/home/elilouis/hsfm/uw_agisoft.lic",
#     parallelization=1
# )
# timesift_pipeline.run()

# These two csv files just have a small subset of the images
timesift_pipeline = TimesiftPipeline(
    metashape_metadata_file = "/data2/elilouis/generate_ee_dems_baker/mixed_timesift_test/combined_metashape_metadata.csv",
    image_metadata_file = "/data2/elilouis/generate_ee_dems_baker/mixed_timesift_test/combined_image_metadata.csv",
    raw_images_directory = "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/preprocessed_images/",
    output_directory = "/data2/elilouis/generate_ee_dems_baker/mixed_timesift_test/",
    reference_dem_lowres = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/baker_2015_utm_10m.tif',
    reference_dem_hires = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/baker_2015_utm_m.tif',
    image_matching_accuracy = 5,
    densecloud_quality = 5,
    output_DEM_resolution = 5,
    license_path="/home/elilouis/hsfm/uw_agisoft.lic",
    parallelization=1
)
timesift_pipeline.run()