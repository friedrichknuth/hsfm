import hsfm
import pandas as pd
import os

original_input_images_metadata_file = "/home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/nagap_image_metadata.csv" #contains original metadata for images, camera extrinsics and image dates included
post_timesift_input_images_metadata_file = "/data2/elilouis/rainier_carbon_timesift/rainier_carbon_timesift_hsfm/nuth_aligned_bundle_adj_metadata.csv" #contains timesift-updated camera locations
post_timesift_output_directory = '/data2/elilouis/rainier_carbon_timesift/rainier_carbon_post_timesift_hsfm/'

## Join the aligned cameras/images dataset with the dataset that has the date information
aligned_cameras_df = pd.read_csv(post_timesift_input_images_metadata_file)
og_data_df = pd.read_csv(original_input_images_metadata_file, index_col=0)
og_data_df = og_data_df[['fileName', 'Year', 'Month', 'Day']]
og_data_df['image_file_name'] = og_data_df['fileName'] + ".tif"
joined_df = pd.merge(og_data_df, aligned_cameras_df, on='image_file_name')

joined_df['Month'] = joined_df['Month'].fillna('0')
joined_df['Day'] = joined_df['Day'].fillna('0')
grouped_by_day = joined_df.groupby(['Year', 'Month', 'Day'])
datestrings_and_dfs = [('_'.join(date_tuple), df) for date_tuple, df in grouped_by_day]

daily_dir_names = []
for date_string, df in datestrings_and_dfs:
    # Drop unncessary-for-processing columns (we only needed them to separate by year)
    df = df.drop(
        ['fileName','Year', 'Month','Day'],
        axis = 1
    )
    csv_output_path = os.path.join(post_timesift_output_directory, date_string, 'metashape_metadata.csv')
    parent_dir = os.path.dirname(csv_output_path)
    
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir)
    df.to_csv(csv_output_path, index=False)
    daily_dir_names.append(parent_dir)


