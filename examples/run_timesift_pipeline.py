from hsfm.pipeline import TimesiftPipeline
from hsfm.pipeline import NAGAPPreprocessingPipeline

# Example script running preprocessing and and timesift processing for all NAGAP images within bounds near Mt Baker.

preprocess_pipeline = NAGAPPreprocessingPipeline(
    "/data2/elilouis/timesift/baker",
    fiducial_templates_directory = "/home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/fiducials/nagap/",
    nagap_images_csv_path = "/home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/nagap_image_metadata_updated_manual.csv",
    bounds = "-121.94 48.84 -121.70 48.70"
)

preprocess_pipeline.run()

timesift_pipeline = TimesiftPipeline(
    metashape_metadata_file =   "/data2/elilouis/timesift/baker/metashape_metadata.csv",
    image_metadata_file =       "/data2/elilouis/timesift/baker/image_metadata.csv",
    raw_images_directory =      "/data2/elilouis/timesift/baker/preprocessed_images/",
    output_directory =          "/data2/elilouis/timesift/baker/",
    reference_dem_lowres =      "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/baker_2015_utm_10m.tif",
    reference_dem_hires =       "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/baker_2015_utm_m.tif",
    image_matching_accuracy =   1,
    densecloud_quality =        2,
    output_DEM_resolution =     1,
    license_path =              "/home/elilouis/hsfm/uw_agisoft.lic",
    parallelization =           1
)
timesift_pipeline.run()