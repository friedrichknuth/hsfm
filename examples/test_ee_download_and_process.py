from hsfm import batch
from hipp.dataquery import EE_login
from getpass import getpass

print('Enter username:')
username = input()

api_key = EE_login(username, password = getpass())

batch.EE_pre_process_images(
        api_key,
        project_name = 'test',
        bounds = (-122.60, 48.62, -121.58, 48.60),
        ee_project_name = 'LK000',
        year = 1950,
        month = 9,
        day = 2,
        pixel_pitch = None,
        buffer_m            = 2000,
        threshold_px        = 50,
        missing_proxy       = None, 
        download_images     = True,
        template_parent_dir = 'examples/input_data/fiducials/ee/LK000',
        output_directory    = 'examples/input_data/test_ee_download_and_process',
        ee_query_max_results   = 2,
        ee_query_label = 'test_download'
    )