import hvplot
import hvplot.xarray
import hvplot.pandas
import rasterio
import xarray as xr
import cartopy.crs as ccrs
import geoviews as gv
from geoviews import opts
import glob
import holoviews as hv
from osgeo import gdal
import os
import pandas as pd
import panel as pn
import numpy as np
import PIL
import shutil
import subprocess
from subprocess import Popen, PIPE, STDOUT
import time
import utm
import cv2

hv.extension('bokeh')

import hsfm.io
import hsfm.geospatial


"""
Utilities that call other software as subprocesses.
"""


def dem_align_custom(reference_dem,
                     dem_to_be_aligned,
                     output_directory,
                     mode='nuth',
                     max_offset = 1000,
                     verbose=False,
                     print_call=False):
    
    call = ['dem_align.py',reference_dem,
            dem_to_be_aligned,
            '-max_offset',str(max_offset),
            '-mode', mode,
            '-mask_list', 'glaciers', 'nlcd']
    
    if print_call==True:
        print(*call)
    else:         
        log_file = run_command(call, verbose=verbose, log_directory=os.path.join(output_directory,'qc/dem_align'))

        with open(log_file, 'r') as file:
            output_plot_file = file.read().split()[-3]
        dem_difference_file = glob.glob(os.path.split(output_plot_file)[0]+'/*_align_diff.tif')[0]
        aligned_dem_file = glob.glob(os.path.split(output_plot_file)[0]+'/*align.tif')[0]

        return dem_difference_file , aligned_dem_file
    

def rescale_geotif(geotif_file_name,
                   output_file_name=None,
                   scale=1,
                   verbose=False):
                   
    percent = str(100/scale) +'%'
    
    if output_file_name is None:
        file_path, file_name, file_extension = hsfm.io.split_file(geotif_file_name)
        output_file_name = os.path.join(file_path, 
                                        file_name+'_sub'+str(scale)+file_extension)
    
    call = ['gdal_translate',
            '-of','GTiff',
            '-co','TILED=YES',
            '-co','COMPRESS=LZW',
            '-co','BIGTIFF=IF_SAFER',
            '-outsize',percent,percent,
            geotif_file_name,
            output_file_name]
            
    run_command(call, verbose=verbose)
    
    return output_file_name

def optimize_geotif(geotif_file_name,
                    output_file_name=None,
                    verbose=False):
                   

    if output_file_name is None:
        file_path, file_name, file_extension = hsfm.io.split_file(geotif_file_name)
        output_file_name = os.path.join(file_path, 
                                        file_name+'_optimized'+file_extension)
    
    call = ['gdal_translate',
            '-of','GTiff',
            '-co','TILED=YES',
            '-co','COMPRESS=LZW',
            '-co','BIGTIFF=IF_SAFER',
            geotif_file_name,
            output_file_name]
    run_command(call, verbose=verbose)
    
    return output_file_name


def download_srtm(LLLON,LLLAT,URLON,URLAT,
                  output_directory='./data/reference_dem/',
                  verbose=True,
                  cleanup=False):
    # TODO
    # - Add function to determine extent automatically from input cameras
    # - Make geoid adjustment and converstion to UTM optional
    # - Preserve wgs84 dem
    import elevation
    
    run_command(['eio', 'selfcheck'], verbose=verbose)
    print('Downloading SRTM DEM data...')

    hsfm.io.create_dir(output_directory)

    cache_dir=output_directory
    product='SRTM3'
    dem_bounds = (LLLON, LLLAT, URLON, URLAT)

    elevation.seed(bounds=dem_bounds,
                   cache_dir=cache_dir,
                   product=product,
                   max_download_tiles=999)

    tifs = glob.glob(os.path.join(output_directory,'SRTM3/cache/','*tif'))
    
    vrt_file_name = os.path.join(output_directory,'SRTM3/cache/srtm.vrt')
    
    call = ['gdalbuildvrt', vrt_file_name]
    call.extend(tifs)
    run_command(call, verbose=verbose)

    
    ds = gdal.Open(vrt_file_name)
    vrt_subset_file_name = os.path.join(output_directory,'SRTM3/cache/srtm_subset.vrt')
    ds = gdal.Translate(vrt_subset_file_name,
                        ds, 
                        projWin = [LLLON, URLAT, URLON, LLLAT])
                        
    
    # Adjust from EGM96 geoid to WGS84 ellipsoid
    adjusted_vrt_subset_file_name_prefix = os.path.join(output_directory,'SRTM3/cache/srtm_subset')
    call = ['dem_geoid',
            '--reverse-adjustment',
            vrt_subset_file_name, 
            '-o', 
            adjusted_vrt_subset_file_name_prefix]
    run_command(call, verbose=verbose)
    
    adjusted_vrt_subset_file_name = adjusted_vrt_subset_file_name_prefix+'-adj.tif'

    # Get UTM EPSG code
    epsg_code = hsfm.geospatial.wgs_lon_lat_to_epsg_code(LLLON, LLLAT)
    
    # Convert to UTM
    utm_vrt_subset_file_name = os.path.join(output_directory,'SRTM3/cache/srtm_subset_utm_geoid_adj.tif')
    call = 'gdalwarp -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER -dstnodata -9999 -r cubic -t_srs EPSG:' + epsg_code
    call = call.split()
    call.extend([adjusted_vrt_subset_file_name,utm_vrt_subset_file_name])
    run_command(call, verbose=verbose)
    
    # Cleanup
    if cleanup == True:
        print('Cleaning up...','Reference DEM available at', out)
        out = os.path.join(output_directory,os.path.split(utm_vrt_subset_file_name)[-1])
        os.rename(utm_vrt_subset_file_name, out)
        shutil.rmtree(os.path.join(output_directory,'SRTM3/'))
    
        return out
        
    else:
        return utm_vrt_subset_file_name


def clip_reference_dem(dem_file, 
                       reference_dem_file,
                       output_file_name = 'reference_dem_clip.tif',
                       print_call       =False,
                       verbose          =False,
                       buff_size        = 1000):
    
    # TODO check that input DEMs are both in utm
    
    rasterio_dataset = rasterio.open(dem_file)
    bounds = rasterio_dataset.bounds
    left   = str(bounds[0] - buff_size)
    top    = str(bounds[3] + buff_size)
    right  = str(bounds[2] + buff_size)
    bottom = str(bounds[1] - buff_size)
    center = str(rasterio_dataset.xy(rasterio_dataset.height // 2, 
                                     rasterio_dataset.width // 2))
    
    call =['gdal_translate',
          '-projwin',
           left,
           top,
           right,
           bottom,
           reference_dem_file,
           output_file_name]
    
    if print_call==True:
        print(*call)
        
    else:
        call = ' '.join(call)
        hsfm.utils.run_command(call, verbose=verbose, shell=True)
        return output_file_name


'''
####
FUNCTIONS BELOW HERE SHOULD BE MOVED ELSEWHERE.
####
'''
    
## TODO move to hsfm.asp as is asp command
def difference_dems(dem_file_name_a,
                    dem_file_name_b,
                    verbose=False):
    
    file_path, file_name, file_extension = hsfm.io.split_file(dem_file_name_a)
    
    output_directory_and_prefix = os.path.join(file_path,file_name)
    
    call = ['geodiff',
            dem_file_name_a,
            dem_file_name_b,
            '-o', output_directory_and_prefix]
            
    run_command(call, verbose=verbose)
    
    output_file_name = output_directory_and_prefix+'-diff.tif'
    
    return output_file_name



## TODO move to hsfm.core as best fit (for now)
def pick_headings(image_directory, camera_positions_file_name, subset, delta=0.015):
    df = hsfm.core.select_images_for_download(camera_positions_file_name, subset)
    
    image_file_paths = sorted(glob.glob(os.path.join(image_directory, '*.tif')))
    lons = df['Longitude'].values
    lats = df['Latitude'].values

    headings = []
    for i, v in enumerate(lats):
        image_file_name = image_file_paths[i]
        camera_center_lon = lons[i]
        camera_center_lat = lats[i]

        heading = pick_heading_from_map(image_file_name,
                                        camera_center_lon,
                                        camera_center_lat,
                                        dx= delta,
                                        dy= delta)
        headings.append(heading)

    df['heading'] = headings

    return df

## TODO move to hsfm.trig (might need to rename library as hsfm.math)
def scale_down_number(number, threshold=1000):
    while number > threshold:
        number = number / 2
    number = int(number)
    return number

## TODO move to hsfm.tools (needs to be created) as this launches a self contained app
def pick_heading_from_map(image_file_name,
                          camera_center_lon,
                          camera_center_lat,
                          dx = 0.015,
                          dy = 0.015):
                          

    # Google Satellite tiled basemap imagery url
    url = 'https://mt1.google.com/vt/lyrs=s&x={X}&y={Y}&z={Z}'
    
    # TODO
    # # allow large images to be plotted or force resampling to thumbnail
    # # load the image with xarray and plot with hvplot to handle larger images
    img, subplot_width, subplot_height = hv_plot_raster(image_file_name)

    # create the extent of the bounding box
    extents = (camera_center_lon-dx, 
               camera_center_lat-dy, 
               camera_center_lon+dx, 
               camera_center_lat+dy)


    # run the tile server
    tiles = gv.WMTS(url, extents=extents)

    points = gv.Points([(camera_center_lon,
                         camera_center_lat,
                         'camera_center')], 
                         vdims='location')

    point_stream = hv.streams.PointDraw(source=points)

    base_map = (tiles * points).opts(opts.Points(width=subplot_width, 
                                                 height=subplot_height, 
                                                 size=10, 
                                                 color='black', 
                                                 tools=["hover"]))

    row = pn.Row(img, base_map)

    server = row.show(threaded=True)

    condition = True
    while condition == True:
        try:
            if len(point_stream.data['x']) == 2:
                server.stop()
                condition = False
        except:
            pass

    projected = gv.operation.project_points(point_stream.element,
                                            projection=ccrs.PlateCarree())
    df = projected.dframe()
    df['location'] = ['camera_center', 'flight_direction']
    
    heading_lon = df.x[1]
    heading_lat = df.y[1]
    
    heading = hsfm.geospatial.calculate_heading(camera_center_lon,
                                                camera_center_lat,
                                                heading_lon,
                                                heading_lat)
    
    return heading

## TODO move to hsfm.tools (needs to be created) as this launches a self contained app
def create_fiducials(pid, 
                     output_directory = 'fiducials'):
                     
    hsfm.io.create_dir(output_directory)
                     
    image_array = hsfm.core.download_image(pid)
    
    hsfm.io.create_dir('tmp/')
    temp_out = os.path.join('tmp/', 'temporary_image.tif')
    cv2.imwrite(temp_out,image_array)
    temp_out_optimized = hsfm.utils.optimize_geotif(temp_out)
    os.remove(temp_out)
    os.rename(temp_out_optimized, temp_out)
    
    hv_image, subplot_width, subplot_height = hsfm.utils.hv_plot_raster(temp_out)
    
    points = hv.Points([])
    point_stream = hv.streams.PointDraw(source=points)

    app = (hv_image * points).opts(hv.opts.Points(width=subplot_width,
                                                  height=subplot_height,
                                                  size=5,
                                                  color='blue',
                                                  tools=["hover"]))

    panel = pn.panel(app)

    server = panel.show(threaded=True)

    condition = True
    while condition == True: 
        try:
            if len(point_stream.data['x']) == 4:
                server.stop()
                condition = False
        except:
            pass

    df = point_stream.element.dframe()

    left_fiducial   = (df.x[0],df.y[0])
    top_fiducial    = (df.x[1],df.y[1])
    right_fiducial  = (df.x[2],df.y[2])
    bottom_fiducial = (df.x[3],df.y[3])

    fiducials = [left_fiducial, top_fiducial, right_fiducial, bottom_fiducial]
    
    dist = np.min(fiducials)
    
    x_L = int(left_fiducial[0]-dist)
    x_R = int(left_fiducial[0])
    y_T = int(left_fiducial[1]-dist)
    y_B = int(left_fiducial[1]+dist)
    cropped = image_array[y_T:y_B, x_L:x_R]
    cv2.imwrite(os.path.join(output_directory,'L.jpg'),cropped)
    
    x_L = int(top_fiducial[0]-dist)
    x_R = int(top_fiducial[0]+dist)
    y_T = int(top_fiducial[1]-dist)
    y_B = int(top_fiducial[1])
    cropped = image_array[y_T:y_B, x_L:x_R]
    cv2.imwrite(os.path.join(output_directory,'T.jpg'),cropped)
    
    x_L = int(right_fiducial[0])
    x_R = int(right_fiducial[0]+dist)
    y_T = int(right_fiducial[1]-dist)
    y_B = int(right_fiducial[1]+dist)
    cropped = image_array[y_T:y_B, x_L:x_R]
    cv2.imwrite(os.path.join(output_directory,'R.jpg'),cropped)
    
    x_L = int(bottom_fiducial[0]-dist)
    x_R = int(bottom_fiducial[0]+dist)
    y_T = int(bottom_fiducial[1])
    y_B = int(bottom_fiducial[1]+dist)
    cropped = image_array[y_T:y_B, x_L:x_R]
    cv2.imwrite(os.path.join(output_directory,'B.jpg'),cropped)


## TODO move to hsfm.tools (needs to be created) as this launches a self contained app
def launch_fiducial_picker(hv_image, subplot_width, subplot_height):
    points = hv.Points([])
    point_stream = hv.streams.PointDraw(source=points)

    app = (hv_image * points).opts(hv.opts.Points(width=subplot_width,
                                                  height=subplot_height,
                                                  size=5,
                                                  color='blue',
                                                  tools=["hover"]))

    panel = pn.panel(app)

    server = panel.show(threaded=True)

    condition = True
    while condition == True: 
        try:
            if len(point_stream.data['x']) == 4:
                server.stop()
                condition = False
        except:
            pass
    
    df = point_stream.element.dframe()
    
    left_fiducial   = (df.x[0],df.y[0])
    top_fiducial    = (df.x[1],df.y[1])
    right_fiducial  = (df.x[2],df.y[2])
    bottom_fiducial = (df.x[3],df.y[3])
    
    fiducials = [left_fiducial, top_fiducial, right_fiducial, bottom_fiducial]
    
    principal_point = hsfm.core.determine_principal_point(fiducials[0],
                                                          fiducials[1],
                                                          fiducials[2],
                                                          fiducials[3])
    
    return fiducials, principal_point

## TODO move to hsfm.core as best fit (for now)
def hv_plot_raster(image_file_name):
    src = rasterio.open(image_file_name)

    subplot_width  = scale_down_number(src.shape[0])
    subplot_height = scale_down_number(src.shape[1])

    da = xr.open_rasterio(src)

    hv_image = da.sel(band=1).hvplot.image(rasterize=True,
                                      width=subplot_width,
                                      height=subplot_height,
                                      flip_yaxis=True,
                                      colorbar=False,
                                      cmap='gray')
                                      
    return hv_image, subplot_width, subplot_height

## TODO move to hsfm.core as best fit (for now)
def pick_fiducials(image_file_name):
    
    hv_image, subplot_width, subplot_height = hsfm.utils.hv_plot_raster(image_file_name)
    fiducials, principal_point = hsfm.utils.launch_fiducial_picker(hv_image,
                                                                   subplot_width,
                                                                   subplot_height)
    
    intersection_angle = hsfm.core.determine_intersection_angle(fiducials)
    
    return principal_point, intersection_angle, fiducials
    
## TODO move to hsfm.io and add docs
def run_command(command, verbose=False, log_directory=None, shell=False):
    
    p = Popen(command,
              stdout=PIPE,
              stderr=STDOUT,
              shell=shell)
    
    if log_directory != None:
        log_file_name = os.path.join(log_directory,command[0]+'_log.txt')
        hsfm.io.create_dir(log_directory)
    
        with open(log_file_name, "w") as log_file:
            
            while p.poll() is None:
                line = (p.stdout.readline()).decode('ASCII').rstrip('\n')
                if verbose == True:
                    print(line)
                log_file.write(line)
        return log_file_name
    
    else:
        while p.poll() is None:
            line = (p.stdout.readline()).decode('ASCII').rstrip('\n')
            if verbose == True:
                print(line)
                
def run_command2(command, verbose=False, log=False):
    
    log_directory='logs'
    
    p = Popen(command,
              universal_newlines=True,
              stdout=PIPE,
              stderr=STDOUT,
              shell=True)
    
    if log != False:
        log_file_name = os.path.join(log_directory,command.split(' ')[0]+'_log.txt')
        hsfm.io.create_dir(log_directory)
    
        with open(log_file_name, "w") as log_file:
            
            while p.poll() is None:
                line = (p.stdout.readline())
                if verbose == True:
                    print(line)
                log_file.write(line)
        return log_file_name
    
    else:
        while p.poll() is None:
            if verbose == True:
                line = (p.stdout.readline())
                print(line)