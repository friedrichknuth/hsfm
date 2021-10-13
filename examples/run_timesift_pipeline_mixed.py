from hsfm.pipeline import TimesiftPipeline

# Example script running timesift processing for a mix of NAGAP and EE images within bounds surrounding Mt Baker.
# Currently the preprocessing to download and prep images is performed manually in a notebook.

# See jupyter notebook for preparation steps

# Instantiate the pipeline
timesift_pipeline = TimesiftPipeline(
    metashape_metadata_file = "/data2/elilouis/timesift/baker-ee/mixed_timesift/combined_metashape_metadata.csv",
    image_metadata_file = "/data2/elilouis/timesift/baker-ee/mixed_timesift/combined_image_metadata.csv",
    raw_images_directory = "/data2/elilouis/timesift/baker-ee/mixed_timesift/preprocessed_images/",
    output_directory = "/data2/elilouis/timesift/baker-ee/mixed_timesift/",
    reference_dem_lowres = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015_10m.tif',
    reference_dem_hires = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015.tif',
    image_matching_accuracy = 1,
    densecloud_quality = 2,
    output_DEM_resolution = 1,
    license_path="/home/elilouis/hsfm/uw_agisoft.lic",
    parallelization=1
)

# Run the pipeline
timesift_pipeline.run()

# Create a DEM quality report
timesift_pipeline.create_results_report()

# Generate mosaic files for each date
timesift_pipeline.create_mosaics(3.5)