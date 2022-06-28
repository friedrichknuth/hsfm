import hsfm

bounds=(-121.85443639373807, 48.7260076474602, -121.81825977316481, 48.7485052926928)

hsfm.dataquery.process_3DEP_laz_to_DEM(
    bounds,
    output_path="./easton",
    DEM_file_name='easton_DSM.tif',
    verbose=True,
    cleanup=False,
    dem_resolution = 1,
    cache_directory="cache",
    dry_run=False
)


