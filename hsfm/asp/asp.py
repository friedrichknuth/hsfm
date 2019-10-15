from osgeo import gdal
import os
import glob
import utm
import shutil

import hsfm.io
import hsfm.core
import hsfm.utils


"""
This library is intended to contain wrappers around ASP functions.
"""

# TODO 
# - implement stereo pair matching based on image footprints

def generate_ba_cameras(image_directory,
                        gcp_directory,
                        intial_cameras_directory):
                        
    # TODO
    # - the core of this function should live here, but the iteration over multiple files
    #   should be moved to batch
    
    output_directory = 'output_data/cameras/'
    
    images = sorted(glob.glob(os.path.join(image_directory,'*.tif')))
    gcp = sorted(glob.glob(os.path.join(gcp_directory,'*.gcp')))
    cameras = sorted(glob.glob(os.path.join(intial_cameras_directory,'*.tsai')))

    for i, v in enumerate(images):
        file_path, file_name, file_extension = hsfm.io.split_file(images[i])
    
        call =['bundle_adjust',
               '-t', 'nadirpinhole',
               images[i],
               cameras[i],
               gcp[i],
               '--datum', 'wgs84',
               '--inline-adjustments',
               '--camera-weight', '10',
               '--max-iterations' ,'0',
               '--robust-threshold', '10',
               '--num-passes', '1',
               '-o', 'output_data/cameras/tmp/run']
        hsfm.utils.run_command(call)

    camera_files = glob.glob(os.path.join(output_directory, 'tmp',"*.tsai"))
    for camera_file in camera_files:
        file_path, file_name, file_extension = hsfm.io.split_file(camera_file)
        new_camera_name = os.path.join(output_directory, file_name[4:] + file_extension)
        shutil.copy2(camera_file,new_camera_name)

    shutil.rmtree('output_data/cameras/tmp/')
    return output_directory


def bundle_adjust_custom(image_files_directory, 
                         camera_files_directory, 
                         output_directory_prefix,
                         print_asp_call=False):
    
    input_image_files  = sorted(glob.glob(os.path.join(image_files_directory,'*.tif')))
    input_camera_files  = sorted(glob.glob(os.path.join(camera_files_directory,'*.tsai')))
    
    ba_dir = os.path.split(output_directory_prefix)[0]
    
    log_directory = os.path.join(ba_dir,'log')
    hsfm.io.create_dir(log_directory)
    
    call =['bundle_adjust',
           '--threads', '1',
           '--disable-tri-ip-filter',
           '--force-reuse-match-files',
           '--skip-rough-homography',
           '-t', 'nadirpinhole',
           '--ip-inlier-factor', '1',
           '--ip-uniqueness-threshold', '0.9',
           '--ip-per-tile','4000',
           '--datum', 'wgs84',
           '--inline-adjustments',
           '--camera-weight', '0.0',
           '--num-iterations', '500',
           '--num-passes', '3']
           
    call.extend(input_image_files)
    call.extend(input_camera_files)
    call.extend(['-o', output_directory_prefix])
    
    if print_asp_call==True:
        print(*call)
    
    hsfm.utils.run_command(call, 
                           verbose=False, 
                           log_directory=log_directory)
                           
    print('Bundle adjust results saved in', ba_dir)
    return ba_dir


def parallel_stereo_custom(first_image, 
                           second_image,
                           first_camera,
                           second_camera, 
                           stereo_output_directory_prefix,
                           print_asp_call=False):
    

    stereo_output_directory = os.path.split(stereo_output_directory_prefix)[0]
    
    log_directory = os.path.join(stereo_output_directory,'log')
    hsfm.io.create_dir(log_directory)
    
    call =['parallel_stereo',
           '--force-reuse-match-files',
           '--stereo-algorithm', '2',
           '-t', 'nadirpinhole',
           '--skip-rough-homography',
           '--ip-inlier-factor', '1',
           '--ip-per-tile','2000',
           '--ip-uniqueness-threshold', '0.9',
           '--ip-debug-images']
           
    call.extend([first_image,second_image])
    call.extend([first_camera,second_camera])
    call.extend([stereo_output_directory_prefix])
    
    if print_asp_call==True:
        print(*call)
        
    hsfm.utils.run_command(call, 
                           verbose=False, 
                           log_directory=log_directory)
                           
    print('Parallel stereo results saved in', stereo_output_directory)
    return stereo_output_directory
    
def dem_mosaic_custom(stereo_output_directories_parent, 
                      output_file_name,
                      verbose=False,
                      print_asp_call=False):
    """
    Function to run ASP dem_mosaic.
    """
    dems = glob.glob(os.path.join(stereo_output_directories_parent,'*','run-DEM.tif'))
    
    call = ['dem_mosaic']
    call.extend(dems)
    call.extend(['-o', output_file_name])

    if print_asp_call==True:
        print(*call)
        
    hsfm.utils.run_command(call, verbose=verbose)

def generate_match_points(image_directory,
                          camera_directory,
                          output_directory='output_data/match_files',
                          verbose=False,
                          print_asp_call=False):
    image_file_list = sorted(glob.glob(os.path.join(image_directory,'*.tif')))
    camera_file_list = sorted(glob.glob(os.path.join(camera_directory,'*.tsai')))
    template_camera = camera_file_list[0]
    
    call =['camera_solve',
           output_directory]
    call.extend(image_file_list)
    call.extend(['--calib-file', 
                 template_camera,
                 '--bundle-adjust-params', 
                 '"--no-datum --ip-per-tile 1000 --ip-uniqueness-threshold 0.9"'])
    if print_asp_call==True:
        print(*call)
    hsfm.utils.run_command(call, verbose=verbose)

def point2dem_custom(point_cloud_file_name, 
                     proj_string='"+proj=utm +zone=10 +datum=WGS84 +units=m +no_defs"',
                     verbose=False,
                     print_asp_call=False):
    # TODO
    # - build proj string upstream
    
    """
    Function to run ASP point2dem.
    """
    
    call =['point2dem',
           '--t_srs',
           proj_string,
           '--errorimage']
       
    call.extend([point_cloud_file_name])

    if print_asp_call==True:
        print(*call)
        
    call = ' '.join(call)

    
    hsfm.utils.run_command(call, verbose=verbose, shell=True)
    
    file_path, file_name, file_extension = hsfm.io.split_file(point_cloud_file_name)
    dem_file_name = os.path.join(file_path,file_name+'-DEM'+file_extension)
    return dem_file_name
    
    
def pc_align_custom(input_dem_file_name,
                    reference_dem_file_name,
                    output_directory_prefix,
                    verbose=False,
                    print_asp_call=False):
    """
    Function to run ASP pc_align.                
    """
    
#     log_directory = os.path.join(output_directory,'log')
    log_directory = None
    
    call = ['pc_align',
            '--save-transformed-source-points',
            '--max-displacement', '-1',
            reference_dem_file_name,
            input_dem_file_name,
            '--alignment-method', 'similarity-point-to-point',
            '-o', output_directory_prefix
    ]
    
    if print_asp_call==True:
        print(*call)

    hsfm.utils.run_command(call, 
                           log_directory=log_directory, 
                           verbose=verbose)
    
    output_directory = os.path.split(output_directory_prefix)[0]
    point_cloud_file_name = os.path.join(output_directory,'run-trans_source.tif')
    dem_file_name = point2dem_custom(point_cloud_file_name)
    return dem_file_name








'''
####
FUNCTIONS BELOW HERE SHOULD BE MOVED ELSEWHERE.
####
'''

def iter_stereo_pairs(stereo_input_directory,
                      image_files_directory,
                      camera_files_directory,
                      stereo_output_directory_prefix,
                      image_extension = '.tif',
                      camera_extension = '.tsai'):
    """
    Function to run pairwise bundle_adjust based on match files.
    """
                       
    match_files = sorted(glob.glob(os.path.join(stereo_input_directory,'*.match')))
    input_camera_files  = sorted(glob.glob(os.path.join(camera_files_directory,'*'+camera_extension)))

    for match_file in match_files:
    
        
        match_file_a = os.path.split(match_file)[-1].split('-')[-2].split('__')[0]
        match_file_b = os.path.split(match_file)[-1].split('-')[-2].split('__')[1]
    
        image_a = os.path.join(image_files_directory, match_file_a + image_extension)
        image_b = os.path.join(image_files_directory, match_file_b + image_extension)
    
        for camera_file in input_camera_files:
            if match_file_a in camera_file:
                camera_a = camera_file
        
            if match_file_b in camera_file:
                camera_b = camera_file
            
        output_folder = match_file_a + '__' + match_file_b
            
        output_directory = os.path.join(stereo_output_directory_prefix,output_folder+'/run')
        
        print('Running parallel stereo on', image_a, 'and', image_b)
    
        stereo_output_directory = parallel_stereo_custom(image_a, 
                                                         image_b,
                                                         camera_a,
                                                         camera_b,
                                                         output_directory)
                               
        try:
            point_cloud_file_name = glob.glob(os.path.join(stereo_output_directory,'*PC.tif'))[0]
            point2dem_custom(point_cloud_file_name)
        except:
            print('Unable to generate point cloud from', match_file_a,'and', match_file_b)
                               
                                        
                                        