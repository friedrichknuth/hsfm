from hsfm.pipeline import TimesiftPipeline

# Example script running timesift processing for a mix of NAGAP and EE images within bounds near Mt Baker.
# Currently the preprocessing to download and prep images is performed manually in a notebook.

# See jupyter notebook for preparation steps

timesift_pipeline = TimesiftPipeline(
    metashape_metadata_file = "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/combined_metashape_metadata.csv",
    image_metadata_file = "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/combined_image_metadata.csv",
    raw_images_directory = "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/preprocessed_images/",
    output_directory = "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/",
    reference_dem_lowres = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/baker_2015_utm_10m.tif',
    reference_dem_hires = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/baker_2015_utm_m.tif',
    image_matching_accuracy = 2,
    densecloud_quality = 4,
    output_DEM_resolution = 5,
    license_path="/home/elilouis/hsfm/uw_agisoft.lic",
    parallelization=1
)
timesift_pipeline.run()