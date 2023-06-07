import cartopy.crs as ccrs
import contextily as ctx
import geopandas as gpd
import geoviews as gv
from geoviews import opts
import haversine
import holoviews as hv
from holoviews.streams import PointDraw
import math
import numpy as np
import os
from osgeo import gdal
import pyproj
from pyproj import CRS, Transformer
import pandas as pd
import panel as pn
import requests
import urllib
import rasterio
from shapely.geometry import Point, Polygon, LineString, mapping
import utm
import time

import hsfm

"""
Geospatial data processing functions.
"""

def compare_footprints(gdf1, gdf2):
    intersection = gpd.overlay(gdf1, gdf2, how='intersection')
    if len(intersection) > 0:
        return 1
    else:
        return 0
    
def df_points_to_polygon_gdf(df, 
                             lon='lon',
                             lat='lat',
                             epsg_code='4326'):
    vertices = []

    for i in range(len(df)):
        vertex = (df[lon][i], df[lat][i])
        vertices.append(vertex)
        
    polygon = Polygon(vertices)
    polygon_gdf = gpd.GeoDataFrame(gpd.GeoSeries(polygon), 
                                          columns=['geometry'],
                                          crs='epsg:'+epsg_code) 
    return polygon_gdf

def df_xyz_coords_to_gdf(df, 
                         lon='lon',
                         lat='lat',
                         z='elevation',
                         epsg_code='4326'):
    """
    Function to convert pandas dataframe containing lat, lon, elevation coordinates to geopandas dataframe.
    Use df_xy_coords_to_gdf() if elevation data not available.
    """
    geometry = [Point(xyz) for xyz in zip(df[lon], df[lat], df[z])]        
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='epsg:'+epsg_code)
    
    return gdf
    
def df_xy_coords_to_gdf(df, 
                         lon='lon',
                         lat='lat',
                         epsg_code='4326'):
    """
    Function to convert pandas dataframe containing lat, lon coordinates to geopandas dataframe.
    """
    geometry = [Point(xy) for xy in zip(df[lon], df[lat])]
    gdf = gpd.GeoDataFrame(df,geometry=geometry, crs='epsg:'+epsg_code)
    
    return gdf
    
    
def downsample_geotif_to_array(img_file_name, scale):
    """
    Function to downsample image and return as numpy array.
    """
    img_ds = gdal.Open(img_file_name)
    
    buf_xsize = int(round(img_ds.RasterXSize/scale))
    buf_ysize = int(round(img_ds.RasterYSize/scale))
    
    img = img_ds.ReadAsArray(buf_xsize=buf_xsize, buf_ysize=buf_ysize)
    
    return img

def extract_gpd_geometry(point_gdf):
    """
    Function to extract x, y, z coordinates and add as columns to input geopandas data frame.
    """
    x = []
    y = []
    z = []
    for i in range(len(point_gdf)):
        x.append(point_gdf['geometry'].iloc[i].coords[:][0][0])
        y.append(point_gdf['geometry'].iloc[i].coords[:][0][1])
        if len(point_gdf['geometry'].iloc[i].coords[:][0]) == 3:
            z.append(point_gdf['geometry'].iloc[i].coords[:][0][2])

    point_gdf['x'] = x
    point_gdf['y'] = y
    if len(point_gdf['geometry'].iloc[0].coords[:][0]) == 3:
        point_gdf['z'] = z
    return point_gdf
        
def lon_lat_to_utm_epsg_code(lon, lat):
    """
    Function to retrieve local UTM EPSG code from WGS84 geographic coordinates.
    """
    utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return epsg_code
    
def lon_lat_to_utm_navd88_epsg_code(lon, lat):
    """
    Function to retrieve local UTM EPSG code from WGS84 geographic coordinates using NAVD88 as vertical datum.
    """
    utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+ utm_band
    if lat >= 0:
        epsg_code = '269' + utm_band
        return epsg_code
    else:
        print('Not sure what the right NAD83 / UTM zone EPSG code is for southern latitudes.')

def distance_two_point_on_earth(point1_lon, point1_lat, point2_lon, point2_lat):
    """
    Function to calculate distance between two WGS84 geographic points on earth.
    """
    p1 = (point1_lat,point1_lon)
    p2 = (point2_lat,point2_lon)
    
    distance = haversine.haversine(p1, p2)
    return distance
    

def calculate_heading(point1_lon, point1_lat, point2_lon, point2_lat):
    """
    Calculates the bearing between two points.
    """
    
    p1 = (point1_lat,point1_lon)
    p2 = (point2_lat,point2_lon)
    
    lat1 = math.radians(p1[0])
    lat2 = math.radians(p2[0])

    delta_x = math.radians(p2[1] - p1[1])

    x = math.sin(delta_x) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(delta_x))

    initial_heading = math.atan2(x, y)

    initial_heading = math.degrees(initial_heading)
    final_heading = (initial_heading + 360) % 360

    return final_heading

def rescale_geotif_to_file(geotif_file_name, scale_factor):
    
    # TODO
    # - Get no data value instead of hard code -9999
    
    # create output file name
    file_path, file_name, file_extension = hsfm.io.split_file(geotif_file_name)
    output_file = os.path.join(file_path, file_name +'_sub'+str(scale_factor)+'.tif')
    
    # downsample array
    img = downsample_geotif_to_array(geotif_file_name, scale_factor)

    # get new shape
    [cols,rows] = img.shape

    # preserve information about original file
    transform = img_ds.GetGeoTransform()
    projection = img_ds.GetProjection()
    nodatavalue = -9999
    # nodatavalue = img_ds.GetNoDataValue() # get attributes property broken in gdal 2.4

    # set gdal GTiff driver
    outdriver = gdal.GetDriverByName("GTiff")

    # create output file, write data and add metadata
    outdata = outdriver.Create(output_file, rows, cols, 1, gdal.GDT_Byte)
    outdata.GetRasterBand(1).WriteArray(img)
    outdata.GetRasterBand(1).SetNoDataValue(nodatavalue)
    outdata.SetGeoTransform(transform)
    outdata.SetProjection(projection)
    
    
def request_basemap_tiles(lon, lat, dx, dy, 
                          url='https://mt1.google.com/vt/lyrs=s&x={X}&y={Y}&z={Z}',
                          utm=False):
                            
    if utm == False:
        extents = (lon-dx, lat-dy, lon+dx, lat+dy)
        tiles = gv.WMTS(url, extents=extents)
        location = gv.Points([], vdims="vertices")                   
        point_stream = PointDraw(source=location)
        tiles = gv.WMTS(url, extents=extents)
        
        return tiles
    
    else:
        u = utm.from_latlon(lat,lon)
        utm_lon           = u[0]
        utm_lat           = u[1]
        utm_zone          = u[2]
        utm_zone_code     = u[3]
        
        extents = utm_lon-dx, utm_lat-dy, utm_lon+dx, utm_lat+dy
        tiles = gv.WMTS(url, extents=extents, crs=ccrs.UTM(utm_zone))
        
        return tiles, utm_zone
    
    
    
def pick_points_from_basemap_tiles(tiles, utm_zone=None):
    
    if utm_zone == None:
        location = gv.Points([], vdims="vertices")
    
    else:
        location = gv.Points([], vdims="vertices", crs=ccrs.UTM(utm_zone))
    
    point_stream = PointDraw(source=location)
    base_map = (tiles * location).opts(opts.Points(width=500, 
                                                   height=500, 
                                                   size=12, 
                                                   color='black', 
                                                   tools=["hover"]))
    app = pn.panel(base_map)
    return app, point_stream
    
    
def basemap_points_to_dataframe(point_stream):
    
    df = gv.operation.project_points(point_stream.element).dframe()
    return df
    
def download_basemap_tiles_as_geotif(lon, lat, dx, dy,
                                     output_file_name='output.tif',
                                     url="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"):
                                     
    west, south, east, north = (lon-dx, lat-dy, lon+dx, lat+dy)
    
    img = ctx.tile.bounds2raster(west,
                                 south,
                                 east,
                                 north,
                                 output_file_name,
                                 zoom="auto",
                                 url=url,
                                 ll=True,
                                 wait=0,
                                 max_retries=100)
                                 
    return output_file_name
    
def get_raster_statistics(rasterio_dataset):
    array = rasterio_dataset.read()
    
    # mask out no data values
    mask = (array == rasterio_dataset.nodata)
    masked_array = np.ma.masked_array(array, mask=mask)
    
    stats = []
    for band in masked_array:
        stats.append({
            'min'    : band.min(),
            'max'    : band.max(),
            'median' : np.median(band),
            'mean'   : band.mean(),
            'no_data':rasterio_dataset.nodata,
        })
    return stats

def mask_array_with_nan(array,nodata_value):
    """
    Function to return a masked array in which the fill value has also 
    been replaced with np.nan. This can be used for calculating and 
    plotting clim=np.nanpercentile(masked_array,[lower_bound,upper_bound])
    """
    mask = (array == nodata_value)
    masked_array = np.ma.masked_array(array, mask=mask)
    masked_array = np.ma.filled(masked_array, fill_value=np.nan)
    
    return masked_array

def calculate_hillshade(array,
                        azimuth=315,
                        angle_altitude=45):
    
    azimuth = 360.0 - azimuth 

    x, y = np.gradient(array)
    slope = np.pi/2.-np.arctan(np.sqrt(x*x+y*y))
    aspect = np.arctan2(-x,y)
    azimuthrad = azimuth*np.pi/180.
    altituderad = angle_altitude*np.pi/180.

    shaded = np.sin(altituderad) * \
             np.sin(slope) + \
             np.cos(altituderad) * \
             np.cos(slope) * \
             np.cos((azimuthrad-np.pi/2.) - \
             aspect)

    return 255*(shaded + 1)/2


def reproject_geotif(geotif_file_name, 
                     epsg_code,
                     output_file_name=None,
                     verbose=False):
    """
    Function to reproject a geotif.
    """
    # TODO
    # - consider moving this to hsfm.utils for consistency.
    
    
    if output_file_name == None:
        file_path, file_name, file_extension = hsfm.io.split_file(geotif_file_name)
        
        output_file_name = os.path.join(file_path,file_name+'_EPSG_'+str(epsg_code)+file_extension)
        
    call = ['gdalwarp',
            '-co','COMPRESS=LZW',
            '-co','TILED=YES',
            '-co','BIGTIFF=IF_SAFER',
            '-dstnodata', '-9999',
            '-r','cubic',
            '-t_srs', 'EPSG:'+str(epsg_code),
            geotif_file_name,
            output_file_name]
    
    hsfm.utils.run_command(call, verbose=verbose)
    
    return output_file_name
    
def get_epsg_code(dem_file_name):
    
    src = rasterio.open(dem_file_name)
    epsg_code = str(src.crs.to_epsg())
    
    return epsg_code

def sample_dem(lons, lats, dem_file_name):
    # TODO
    # - check fill value from DEM
    # - interpolate value if fill value or nan
    
    src = rasterio.open(dem_file_name)
    epsg_code = get_epsg_code(dem_file_name)
    
    data = {'lon': lons, 'lat': lats}
    df = pd.DataFrame(data)
    gdf = hsfm.geospatial.df_xy_coords_to_gdf(df)
    gdf = hsfm.geospatial.extract_gpd_geometry(gdf)
    gdf = gdf.to_crs('epsg:'+epsg_code)
    
    lons = gdf.x.to_list()
    lats = gdf.y.to_list()
    lon_lats = list(zip(lons, lats))
    
    elevations = []
    for elevation in src.sample(lon_lats):
        elevations.append(elevation[0])
    return elevations

# From https://github.com/dshean/pygeotools/blob/master/pygeotools/lib/geolib.py
# Formulas for CE90/LE90 here:
# http://www.fgdc.gov/standards/projects/FGDC-standards-projects/accuracy/part3/chapter3

def CE90(x_offset,y_offset):
    RMSE_x = np.sqrt(np.sum(x_offset**2)/x_offset.size) 
    RMSE_y = np.sqrt(np.sum(y_offset**2)/y_offset.size) 
    c95 = 2.4477
    c90 = 2.146
    RMSE_min = min(RMSE_x, RMSE_y)
    RMSE_max = max(RMSE_x, RMSE_y)
    ratio = RMSE_min/RMSE_max
    if ratio > 0.6 and ratio < 1.0:
        out = c90 * 0.5 * (RMSE_x + RMSE_y)
    else:
        out = c90 * np.sqrt(RMSE_x**2 + RMSE_y**2)
    return out

def LE90(z_offset):
    RMSE_z = np.sqrt(np.sum(z_offset**2)/z_offset.size)
    c95 = 1.9600
    c90 = 1.6449
    return c90 * RMSE_z

def compare_dem_extent(dem1_file,
                       dem2_file):
    
    '''
    Returns larger_dem_extent_file, smaller_dem_extent_file
    '''
    
    ds = rasterio.open(dem1_file)
    bounds = ds.bounds
    a = bounds.right - bounds.left + bounds.top - bounds.bottom

    ds = rasterio.open(dem2_file)
    bounds = ds.bounds
    b = bounds.right - bounds.left + bounds.top - bounds.bottom
    
    if a > b:
        return dem1_file, dem2_file
    else:
        return dem2_file, dem1_file
    
def USGS_elevation_function(lats_list, lons_list):
    """
    Query USGS Elevation Point Service using lat, lon lists. Return elevations as list.
    """
    transformer_3d = Transformer.from_crs(CRS("EPSG:5498").to_3d(), 
                                          CRS("EPSG:4326").to_3d(),
                                          always_xy=True,)
    print('Requesting elevations from USGS Elevation Point Service...')
    url = 'https://epqs.nationalmap.gov/v1/json?'
    elevations = []
    for lat, lon in zip(lats_list, lons_list):
        params = {
            'output': 'json',
            'x': lon,
            'y': lat,
            'units': 'Meters',
            'wkid' : '4326'
        }
        c = 0
        while(c < 5):
            try:
                result = requests.get((url + urllib.parse.urlencode(params)))
                elev = float(result.json()['value'])
                elevations.append(transformer_3d.transform(lon, lat, elev)[2])
                break
            except:
                print('Waiting on 200 response from USGS Elevation Point Service...')
                time.sleep(3)
                c = c+1
    return elevations