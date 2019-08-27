from osgeo import gdal
import glob
import os
import shutil
import subprocess
from subprocess import Popen, PIPE, STDOUT

import hsfm.io
import hsfm.geospatial


def run_command(command, verbose=False, log_directory=None):
    
    p = Popen(command,
              stdout=PIPE,
              stderr=STDOUT,
              shell=False)
    
    if log_directory != None:
        log_file_name = os.path.join(log_directory,command[0]+'_log.txt')
    
        with open(log_file_name, "w") as log_file:
            
            while p.poll() is None:
                line = (p.stdout.readline()).decode('ASCII').rstrip('\n')
                if verbose == True:
                    print(line)
                log_file.write(line)
    else:
        while p.poll() is None:
            line = (p.stdout.readline()).decode('ASCII').rstrip('\n')
            if verbose == True:
                print(line)
        
        


def download_srtm(LLLON,LLLAT,URLON,URLAT,
                  output_dir='./data/reference_dem/',
                  verbose=True,
                  cleanup=False):
    # TODO
    # - Add function to determine extent automatically from input cameras
    # - Make geoid adjustment and converstion to UTM optional
    # - Preserve wgs84 dem
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
                        
    
    # Adjust from EGM96 geoid to WGS84 ellipsoid
    adjusted_vrt_subset_file_name_prefix = os.path.join(output_dir,'SRTM3/cache/srtm_subset')
    call = ['dem_geoid','--reverse-adjustment', vrt_subset_file_name, '-o', adjusted_vrt_subset_file_name_prefix]
    run_command(call, verbose=verbose)
    
    adjusted_vrt_subset_file_name = adjusted_vrt_subset_file_name_prefix+'-adj.tif'

    # Get UTM EPSG code
    epsg_code = hsfm.geospatial.wgs_lon_lat_to_epsg_code(LLLON, LLLAT)
    
    # Convert to UTM
    utm_vrt_subset_file_name = os.path.join(output_dir,'SRTM3/cache/srtm_subset_utm_geoid_adj.tif')
    call = 'gdalwarp -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER -dstnodata -9999 -r cubic -t_srs EPSG:' + epsg_code
    call = call.split()
    call.extend([adjusted_vrt_subset_file_name,utm_vrt_subset_file_name])
    run_command(call, verbose=verbose)
    
    # Cleanup
    if cleanup == True:
        print('Cleaning up...','Reference DEM available at', out)
        out = os.path.join(output_dir,os.path.split(utm_vrt_subset_file_name)[-1])
        os.rename(utm_vrt_subset_file_name, out)
        shutil.rmtree(os.path.join(output_dir,'SRTM3/'))
    
        return out
        
    else:
        return utm_vrt_subset_file_name
    