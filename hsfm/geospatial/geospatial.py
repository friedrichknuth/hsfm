import rasterio
from shapely.geometry import Point, Polygon, LineString, mapping
import geopandas as gpd
from osgeo import gdal
import math
import pyproj
import utm
import haversine

def df_xyz_coords_to_gdf(df, 
                         lon='lon',
                         lat='lat',
                         z='elevation',
                         crs='4326'):
    """
    Function to convert pandas dataframe containing lat, lon, elevation coordinates to geopandas dataframe.
    Use df_xy_coords_to_gdf() if elevation data not available.
    """
    geometry = [Point(xyz) for xyz in zip(df[lon], df[lat], df[z])]        
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs={'init':'epsg:'+crs})
    
    return gdf
    
def df_xy_coords_to_gdf(df, 
                         lon='lon',
                         lat='lat',
                         crs='4326'):
    """
    Function to convert pandas dataframe containing lat, lon coordinates to geopandas dataframe.
    """
    geometry = [Point(xy) for xy in zip(df[lon], df[lat])]
    gdf = gpd.GeoDataFrame(df,geometry=geometry, crs={'init':'epsg:'+crs})
    
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
    if len(point_gdf['geometry'].iloc[1].coords[:][0]) == 3:
        point_gdf['z'] = z
        
def wgs_lon_lat_to_epsg_code(lon, lat):
    """
    Function to retreive local UTM EPSG code from WGS84 geographic coordinates.
    """
    utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return epsg_code
    
# def wgs_lon_lat_to_utm_coordinates(lon,lat):
#     """
#     Function to convert WGS84 geographic coordinates to UTM.
#     """
#     # TODO
#     # - Figure out why exactly results differ from using bare.geospatial.utm.from_latlon(lat, lon)
#     epsg_code = wgs_lon_lat_to_epsg_code(lon, lat)
#     utm_code = wgs_lon_lat_to_epsg_code(lon, lat)
#     crs_wgs = pyproj.Proj(init='epsg:4326')
#     crs_utm = pyproj.Proj(init='epsg:{0}'.format(utm_code))
#     x, y = pyproj.transform(crs_wgs, crs_utm, lon, lat)
#     return x, y
    
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