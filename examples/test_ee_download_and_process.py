from hsfm import batch

batch.EE_create_fiducial_marker_proxies_for_project_date(
    bounds = (-122, 49, -121.5, 48.5),
    ee_project_name = 'LK000',
    year = 1950,
    month = 9,
    day = 2,
    # output_directory    = '/data2/elilouis/hsfm_testing',
    output_directory    = '/Users/elischwat/Downloads/hsfm_testing',
    ee_query_max_results   = 10,
    ee_query_label = 'test_download'
)

batch.EE_pre_process_images(
        api_key,
        project_name = 'test',
        bounds = (-122, 49, -121.5, 48.5),
        ee_project_name = 'LK000',
        year = 1950,
        month = 9,
        day = 2,
        pixel_pitch = None,
        focal_length        = None,
        buffer_m            = 2000,
        threshold_px        = 50,
        missing_proxy       = None,
        keep_raw            = True,
        download_images     = True,
        image_square_dim    = None,
        template_parent_dir = '/Users/elischwat/Downloads/fiducials',
        output_directory    = /Users/elischwat/Downloads/hsfm_testing',
        ee_query_max_results   = 50000,
        ee_query_label = 'test_download'
    )