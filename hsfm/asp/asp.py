import glob
from osgeo import gdal
import rasterio
import os
import psutil
import shutil
import utm

import bare
import hsfm.io
import hsfm.core
import hsfm.utils
import hsfm.geospatial


"""
Wrappers around ASP functions.
"""

def generate_ba_cameras(image_directory,
                        gcp_directory,
                        intial_cameras_directory,
                        output_directory,
                        subset=None):
                        
    # TODO
    # - the core of this function should live here, but the iteration over multiple files
    #   should be moved to batch
    
    output_directory = os.path.join(output_directory, 'cameras')
    
    images = sorted(glob.glob(os.path.join(image_directory,'*.tif')))
    gcp = sorted(glob.glob(os.path.join(gcp_directory,'*.gcp')))
    cameras = sorted(glob.glob(os.path.join(intial_cameras_directory,'*.tsai')))
    
    images = hsfm.core.subset_input_image_list(images, subset=subset)
    cameras = hsfm.core.subset_input_image_list(cameras, subset=subset)
    gcp = hsfm.core.subset_input_image_list(gcp, subset=subset)
    
    tmp = os.path.join(output_directory, 'tmp')

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
               '-o', tmp+'/run']
        hsfm.utils.run_command(call)

    camera_files = glob.glob(os.path.join(output_directory, 'tmp',"*.tsai"))
    for camera_file in camera_files:
        file_path, file_name, file_extension = hsfm.io.split_file(camera_file)
        new_camera_name = os.path.join(output_directory, file_name[4:] + file_extension)
        shutil.copy2(camera_file,new_camera_name)

    shutil.rmtree(tmp)
    
    return output_directory


def bundle_adjust_custom(image_files_directory, 
                         camera_files_directory, 
                         output_directory,
                         print_asp_call=False,
                         verbose=False,
                         overlap_list=False,
                         qc=False):
    
    # TODO
    # - make ba arguments optional
    input_image_files  = sorted(glob.glob(os.path.join(image_files_directory,'*.tif')))
    input_camera_files  = sorted(glob.glob(os.path.join(camera_files_directory,'*.tsai')))
    
    ba_output_directory = os.path.join(output_directory, 'ba')
    output_directory_prefix = os.path.join(ba_output_directory+ '/asp_ba_out')
    
    log_directory = os.path.join(ba_output_directory,'log')
    hsfm.io.create_dir(log_directory)
    
    call =['bundle_adjust']
    call.extend(input_image_files)
    call.extend(input_camera_files)   
    call.extend(['--threads', '1',
                 '--ip-detect-method','1',
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
                 '--num-passes', '3'])
    if overlap_list:
                call.extend(['--overlap-list', overlap_list])
                
    call.extend(['-o', output_directory_prefix ])

    if print_asp_call==True:
        print(*call)
    
    else:
        hsfm.utils.run_command(call, 
                           verbose=verbose, 
                           log_directory=log_directory)
        
        if qc == True:
            destination_file_path=os.path.join(output_directory, 'qc/match_files/ba/')
            hsfm.io.batch_rename_files(
                output_directory,
                file_extension='clean.match',
                destination_file_path=destination_file_path)
            bare.core.iter_mp_to_csv(destination_file_path)
            try:
                hsfm.batch.plot_match_overlap(destination_file_path, 
                                          image_files_directory, 
                                          output_directory=os.path.join(output_directory, 'qc/ba_matches'))
                
            except:
                bare.batch.plot_mp_over_images(destination_file_path,
                                               image_files_directory,
                                               output_directory=os.path.join(output_directory, 'qc/ba_matches'))
                
            print('camera_solve match point qc plots saved in', os.path.join(output_directory, 'qc/ba_matches'))
            
        return ba_output_directory


def parallel_stereo_custom(first_image, 
                           second_image,
                           first_camera,
                           second_camera, 
                           stereo_output_directory_prefix,
                           print_asp_call=False,
                           qc = False):
    

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
           '--num-matches-from-disp-triplets','10000']
           
    call.extend([first_image,second_image])
    call.extend([first_camera,second_camera])
    call.extend([stereo_output_directory_prefix])
    
    if print_asp_call==True:
        print(*call)
        
    else:
        hsfm.utils.run_command(call, 
                           verbose=False, 
                           log_directory=log_directory)          
        print('Parallel stereo results saved in', stereo_output_directory)
        return stereo_output_directory
    
def dem_mosaic_custom(output_directory,
                      verbose=False,
                      print_asp_call=False):
    """
    Function to run ASP dem_mosaic.
    """
    output_file = os.path.join(output_directory,'mosaic.tif')
    
    stereo_output_directory = os.path.join(output_directory, 'stereo/stereo_run')
    
    dems = glob.glob(os.path.join(stereo_output_directory,'*','*-DEM.tif'))
    
    call = ['dem_mosaic']
    call.extend(dems)
    call.extend(['-o', output_file])

    if print_asp_call==True:
        print(*call)
        
    else:
        hsfm.utils.run_command(call, verbose=verbose)
        
        return output_file

def generate_match_points(image_directory,
                          camera_directory,
                          output_directory,
                          log=False,
                          verbose=False,
                          print_asp_call=False,
                          qc=False):
    # TODO
    # - make ba arguments optional 
    
    qc_output_directory = os.path.join(output_directory,'qc')
    
    output_directory = os.path.join(output_directory,'cam_solve')
    
    
    image_file_list = sorted(glob.glob(os.path.join(image_directory,'*.tif')))
    camera_file_list = sorted(glob.glob(os.path.join(camera_directory,'*.tsai')))
    template_camera = camera_file_list[0]
    
    call =['camera_solve',
           output_directory]
    call.extend(image_file_list)
    call.extend(['--calib-file', 
                 template_camera,
                 '--reuse-theia-matches',
                 '--bundle-adjust-params', 
                 '"--skip-matching --robust-threshold 15"'])
#                  '"--force-reuse-match-files --no-datum --ip-per-tile 16000 --ip-detect-method 1 --ip-uniqueness-threshold 0.9 --disable-tri-ip-filter --skip-rough-homography --ip-inlier-factor 1"'])
    if print_asp_call==True:
        print(*call)
    else:
        call = ' '.join(call)
        hsfm.utils.run_command2(call, verbose=verbose, log=log)
        
        if qc == True:
            try:
                hsfm.io.batch_rename_files(
                    output_directory,
                    file_extension='.match',
                    destination_file_path=os.path.join(qc_output_directory,
                                                       'match_files/cam_solve/'))
                clean_match_file_list = sorted(glob.glob(os.path.join(qc_output_directory,
                                                                      'match_files/cam_solve/',
                                                                      '*clean.match')))
                for match_file in clean_match_file_list:
                    os.remove(match_file)
                bare.core.iter_mp_to_csv(qc_output_directory,
                                         'match_files/cam_solve/')
                try:
                    hsfm.batch.plot_match_overlap(os.path.join(qc_output_directory,'match_files/cam_solve/'), 
                                                  image_directory, 
                                                  output_directory=os.path.join(qc_output_directory,
                                                                                'cam_solve_matches'))
                except:
                    bare.batch.plot_mp_over_images(os.path.join(qc_output_directory,'match_files/cam_solve/'),
                                                                image_directory,
                                                                output_directory=os.path.join(qc_output_directory,
                                                                                               'cam_solve_matches/'))
                print('camera_solve match point qc plots saved in qc/cam_solve_matches/')
                
            except:
                print('unable to generate match points with camera_solve')
                pass
            
    
        return output_directory


    
def point2dem(point_cloud_file, 
              *args, 
              print_call=False, 
              verbose=False):
    
    args = list(args)
    call =['point2dem', '--threads', str(psutil.cpu_count(logical=False))]
    call.extend(args)
    
    call.append(point_cloud_file)
    
    if print_call==True:
        print(*call)
        
    else:
        call = ' '.join(call)
        hsfm.utils.run_command(call, verbose=verbose, shell=True)

        file_path, file_name, file_extension = hsfm.io.split_file(point_cloud_file)
        dem_file_name = os.path.join(file_path,file_name+'-DEM.tif')
        return dem_file_name
    

def pc_align(input_dem_file,
             reference_dem_file,
             output_directory,
             *args,
             res        = False,
             prefix     = 'run',
             create_dem = True,
             print_call = False,
             verbose    = False,
             shell=False):
    
    
    output_directory_prefix = os.path.join(output_directory,'pc_align/' + prefix)
    transform = output_directory_prefix+'-transform.txt'
    
    args = list(args)
    call = ['pc_align', '--threads', str(psutil.cpu_count(logical=False))]
    if create_dem:
        call.extend(['--save-transformed-source-points',])
    call.extend(args)
    call.extend([reference_dem_file,
                 input_dem_file,
                 '-o',
                 output_directory_prefix])

    if print_call:
        print(*call)
        return '', ''
        
    else:
        hsfm.utils.run_command(call, 
                               verbose=verbose)
    
        if create_dem:
            if not res:
              ds = rasterio.open(input_dem_file)
              res = ds.res[0]
            print('Gridding pc_align outputs to',res)
            output_directory = os.path.split(output_directory_prefix)[0]
            point_cloud_file = output_directory_prefix+'-trans_source.tif'
            epsg_code = 'EPSG:'+ hsfm.geospatial.get_epsg_code(input_dem_file)
            aligned_dem_file = point2dem(point_cloud_file, 
                                         '--t_srs', epsg_code,
                                         '--tr', res)
        
            return aligned_dem_file, transform
        
        else:
            return None, transform
        
        
    
    
def pc_align_p2p_sp2p(input_dem_file,
                      reference_dem_file,
                      output_directory,
                      prefix     = 'run',
                      p2p_max_displacement = 2000,
                      sp2p_max_displacement = 1000,
                      m_sp2p_max_displacement = 100,
                      print_call = False,
                      verbose    = False):
        
    if print_call:
        output_directory_prefix =  os.path.join(output_directory,'pc_align/' + prefix)
        transform = output_directory_prefix+'-transform.txt'
    
    """
    Point 2 Plane ICP
    """
    aligned_dem_file, transform = hsfm.asp.pc_align(input_dem_file,
                                  reference_dem_file,
                                  output_directory,
                                '--save-transformed-source-points',
                                  '--max-displacement',
                                  str(p2p_max_displacement),
#                                 '--outlier-ratio', '90',
                                  '--alignment-method', 
                                  'point-to-plane',
                                  print_call=print_call,
                                  verbose=verbose,
                                  prefix=prefix,
                                  create_dem=True)
    
    hsfm.utils.dem_align_custom(reference_dem_file,
                                aligned_dem_file,
                                verbose = verbose)
    
    """
    Point 2 Point ICP
    """
    prefix0 = '-'.join([prefix,prefix])
    
    aligned_dem_file, transform = hsfm.asp.pc_align(input_dem_file,
                                  reference_dem_file,
                                  output_directory,
                                '--save-transformed-source-points',
                                  '--max-displacement',
                                  str(sp2p_max_displacement),
#                                 '--outlier-ratio', '90',
                                  '--initial-transform', transform,
                                  '--alignment-method',
                                  'similarity-point-to-point',
                                  print_call=print_call,
                                  verbose=verbose,
                                  prefix=prefix0,
                                  create_dem=True)
    
    hsfm.utils.dem_align_custom(reference_dem_file,
                                aligned_dem_file,
                                verbose = verbose)
    
    """
    Point 2 Point ICP with masked reference DEM
    """
    
    masked_reference_dem_file = hsfm.utils.mask_dem(reference_dem_file)

    prefix1 = '-'.join([prefix,prefix,prefix])

    aligned_dem_file, transform = hsfm.asp.pc_align(input_dem_file,
                                                    masked_reference_dem_file,
                                                    output_directory,
                                                    '--save-transformed-source-points',
                                                    '--max-displacement',
                                                    str(m_sp2p_max_displacement),
                                                    '--initial-transform', transform,
                                                    '--alignment-method', 
                                                    'similarity-point-to-point',
                                                    print_call=print_call,
                                                    verbose=verbose,
                                                    prefix=prefix1,
                                                    create_dem=True)

    return aligned_dem_file, transform
    

    
# def pc_align_tfhs_p2p_sp2p(input_dem_file,
#                            reference_dem_file,
#                            output_directory,
#                            verbose=False):
    
#     output_directory = os.path.join(output_directory,'pc_align/tmp')
    
#     point_cloud_file, transform = hsfm.asp.pc_align(input_dem_file,
#                                                     input_dem_file,
#                                                     output_directory,
#                                                     '--max-displacement','-1',
#                                                     '--num-iterations', '0',
#                                                     '--max-num-source-points', '1000',
#                                                     '--max-num-reference-points', '1000',
#                                                     '--save-transformed-source-points',
#                                                     create_dem=False,
#                                                     prefix='dem',
#                                                     verbose=verbose)
    
#     epsg_code = 'EPSG:'+ hsfm.geospatial.get_epsg_code(input_dem_file)
#     tmp_input_dem_file = point2dem(point_cloud_file, 
#                                        '--t_srs', epsg_code,
#                                        '--tr', '5')
    
#     point_cloud_file, transform = hsfm.asp.pc_align(reference_dem_file,
#                                                     reference_dem_file,
#                                                     output_directory,
#                                                     '--max-displacement','-1',
#                                                     '--num-iterations', '0',
#                                                     '--max-num-source-points', '1000',
#                                                     '--max-num-reference-points', '1000',
#                                                     '--save-transformed-source-points',
#                                                     create_dem=False,
#                                                     prefix='ref_dem',
#                                                     verbose=verbose)

    
#     tmp_reference_dem_file = point2dem(point_cloud_file, 
#                                            '--t_srs', epsg_code,
#                                            '--tr', '5')

#     _ , transform = hsfm.asp.pc_align(tmp_input_dem_file,
#                                          tmp_reference_dem_file,
#                                          output_directory,
#                                          '--save-transformed-source-points',
#                                          '--max-displacement','100',
#                                          '--initial-transform-from-hillshading',
#                                          'similarity',
#                                          create_dem=False,
#                                          prefix='run',
#                                          verbose=verbose)
    
#     _ , transform = hsfm.asp.pc_align(tmp_input_dem_file,
#                                      tmp_reference_dem_file,
#                                      output_directory,
#                                      '--save-transformed-source-points',
#                                      '--max-displacement','100',
#                                      '--initial-transform', transform,
#                                      '--alignment-method', 'point-to-plane',
#                                      create_dem=False,
#                                      prefix='run-run',
#                                      verbose=verbose)
    
#     _ , transform = hsfm.asp.pc_align(tmp_input_dem_file,
#                                      tmp_reference_dem_file,
#                                      output_directory,
#                                      '--save-transformed-source-points',
#                                      '--max-displacement','-1',
#                                      '--initial-transform', transform,
#                                      '--alignment-method', 'similarity-point-to-point',
#                                      create_dem=False,
#                                      prefix='run-run-run',
#                                      verbose=verbose)
    

#     aligned_dem_file, transform = hsfm.asp.pc_align(input_dem_file,
#                                                     reference_dem_file,
#                                                     output_directory,
#                                                     '--save-transformed-source-points',
#                                                     '--max-displacement','-1',
#                                                     '--initial-transform', transform,
#                                                     '--alignment-method', 'similarity-point-to-point',
#                                                     prefix='run-run-run-run')
    
#     return aligned_dem_file, transform

def iter_stereo_pairs(output_directory,
                      image_files_directory,
                      projection = 'EPSG:32610',
                      image_extension = '.tif',
                      camera_extension = '.tsai',
                      print_asp_call=False,
                      qc=False):
    """
    Function to run pairwise bundle_adjust based on match files.
    """
    # TODO load projection on the fly
    
    stereo_input_directory = os.path.join(output_directory, 'stereo/stereo_inputs')
    camera_files_directory = os.path.join(output_directory, 'stereo/stereo_inputs')
    stereo_output_directory = os.path.join(output_directory, 'stereo/stereo_run')
    
    
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
        
        output_dir = os.path.join(stereo_output_directory,output_folder)
        hsfm.io.create_dir(output_dir)
        
        hsfm.io.rename_file(match_file, 
                            pattern='-clean',
                            destination_file_path=output_dir,
                            write=True)
            
        output_directory = os.path.join(stereo_output_directory,output_folder+'/asp_ba_out')
        
        print('Running parallel stereo on', image_a, 'and', image_b)
    
        stereo_run_output_directory = parallel_stereo_custom(image_a, 
                                                             image_b,
                                                             camera_a,
                                                             camera_b,
                                                             output_directory,
                                                             print_asp_call=print_asp_call,
                                                             qc = qc)
                               
        try:
            point_cloud_file_name = glob.glob(os.path.join(stereo_run_output_directory,'*PC.tif'))[0]
            point2dem(point_cloud_file_name, 
                      '--t_srs', projection,
                      '--errorimage')
        except:
            print('Unable to generate point cloud from', match_file_a,'and', match_file_b)
                               
    if qc == True:
        destination_file_path=os.path.join(output_directory, 'qc/match_files/stereo/')
        hsfm.io.batch_rename_files(
            stereo_output_directory,
            file_extension='.match',
            unique_id_pattern='asp_ba_out-disp',
            destination_file_path=destination_file_path)
        bare.core.iter_mp_to_csv(destination_file_path)
        try:
            hsfm.batch.plot_match_overlap(destination_file_path, 
                                          image_files_directory, 
                                          output_directory=os.path.join(output_directory, 'qc/stereo_matches/'))
        except:
            bare.batch.plot_mp_over_images(destination_file_path,
                                           image_directory,
                                           output_directory=os.path.join(output_directory, 'qc/stereo_matches/'))
            
        hsfm.qc.eval_stereo_matches(stereo_output_directory,
                                    os.path.join(output_directory, 'qc/stereo_matches/'))
                
        print('camera_solve match point qc plots saved in', os.path.join(output_directory, 'qc/stereo_matches/'))  
        
        
def bundle_adjust(image_files,
                           camera_files,
                           output_directory_prefix,
                           *args,
                           print_call = False,
                           verbose    = False):
    
    output_directory = os.path.dirname(output_directory_prefix)
    
    call = ['bundle_adjust']
    call.extend(args)
    call.extend(image_files)
    call.extend(camera_files)
    call.extend(['-o', output_directory_prefix])
                
    if print_call:
        print(*call)
        return output_directory
    
    else:
        hsfm.utils.run_command(call, 
                               verbose=verbose)
                
        return output_directory
    
def transform_cameras(image_files,
                      camera_files,
                      transform,
                      output_directory_prefix,
                      print_call = False,
                      verbose    = False):
    
    output_directory = os.path.dirname(output_directory_prefix)
    hsfm.io.create_dir(output_directory)
    
#     output_directory = parallel_bundle_adjust(image_files,
#                                                camera_files,
#                                                output_directory_prefix,
#                                                '-t', 'nadirpinhole',
#                                                '--datum', 'wgs84',
#                                                '--force-reuse-match-files',
#                                                '--inline-adjustments',
#                                                '--num-passes', '1',
#                                                '--num-iterations', '0',
#                                                '--initial-transform',transform,
#                                                print_call = print_call,
#                                                verbose    = verbose)

    output_directory = bundle_adjust(image_files,
                                   camera_files,
                                   output_directory_prefix,
                                   '-t', 'pinhole',
                                   '--min-matches', '4',
                                   '--disable-tri-ip-filter', 
                                   '--skip-rough-homography',
                                   '--min-triangulation-angle','0.0001',
                                   '--max-iterations','0',
                                   '--num-passes', '1',
                                   '--inline-adjustments',
                                   '--initial-transform',transform,
                                   print_call = print_call,
                                   verbose    = verbose)
    return output_directory
                                        
