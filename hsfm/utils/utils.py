from osgeo import gdal
import glob
import os
import subprocess
from subprocess import Popen, PIPE, STDOUT

import hsfm.io


def run_command(command, verbose=False):
    
    p = Popen(command,
              stdout=PIPE,
              stderr=STDOUT,
              shell=False)
    
    while p.poll() is None:
        line = (p.stdout.readline()).decode('ASCII').rstrip('\n')
        if verbose == True:
            print(line)


def download_srtm(LLLON,LLLAT,URLON,URLAT,
                  output_dir='./data/reference_dems/',
                  verbose=True):
    # TODO
    # - Add reverse adjustment
    # - Add docstring, comments and useful exceptions.
    # - Add function to determin extent automatically from input cameras
    import elevation
    
    run_command(['eio', 'selfcheck'], verbose=verbose)
    print('Downloading SRTM DEM data...')

    hsfm.io.create_dir(output_dir)

    cache_dir=output_dir
    product='SRTM3'
    dem_bounds = (LLLON, LLLAT, URLON, URLAT)

    elevation.seed(bounds=dem_bounds,
                   cache_dir=cache_dir,
                   product=product,
                   max_download_tiles=999)

    tifs = glob.glob(os.path.join(output_dir,'SRTM3/cache/','*tif'))
    
    vrt_file_name = os.path.join(output_dir,'SRTM3/cache/srtm.vrt')
    
    call = ['gdalbuildvrt', vrt_file_name]
    call.extend(tifs)
    run_command(call, verbose=verbose)

    
    ds = gdal.Open(vrt_file_name)
    vrt_subset_file_name = os.path.join(output_dir,'SRTM3/cache/srtm_subset.vrt')
    ds = gdal.Translate(vrt_subset_file_name,
                        ds, 
                        projWin = [LLLON, URLAT, URLON, LLLAT])
                        
                        
    # Apply DEM geoid
    call = ['dem_geoid','--reverse-adjustment',vrt_subset_file_name]
    run_command(call, verbose=verbose)
    
    adjusted_vrt_subset_file_name = os.path.join(output_dir,'SRTM3/cache/srtm_subset-adj.vrt')

    return adjusted_vrt_subset_file_name
    