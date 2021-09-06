import contextily as ctx
import fsspec
import geopandas as gpd
import os
import glob
import shutil
import sys
import pathlib
import subprocess
from subprocess import Popen, PIPE, STDOUT
import pathlib
import json
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import psutil
import numpy as np
import hsfm


##### 3DEP AWS lidar #####
# TODO 
# - make this a class

def process_3DEP_laz_to_DEM(
    bounds,
    aws_3DEP_directory=None,
    epsg_code=None,
    output_path="./",
    DEM_file_name="dem.tif",
    verbose=True,
    cleanup=False,
    cache_directory='cache',
):
    """
    Grids bounds into 0.01 deg tiles and processes to laz to DSM. 
    Some tiles may contain no data on AWS and are expexted to fail.
    bounds = [east, south, west, north]
    """
    print('Requested bounds:',bounds)
    print('Should be in order of [east, south, west, north]')
    
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    result_gdf, bounds_gdf = hsfm.dataquery.get_3DEP_lidar_data_dirs(bounds, cache_directory=cache_directory)

    if not epsg_code:
        epsg_code = hsfm.dataquery.get_UTM_EPSG_code_from_bounds(bounds)
    epsg_code = str(epsg_code)

    if aws_3DEP_directory:
        if aws_3DEP_directory in result_gdf["directory"].to_list():
            result_gdf = result_gdf.loc[result_gdf["directory"] == aws_3DEP_directory]
            result_gdf = result_gdf.reset_index(drop=True)
        else:
            message = " ".join(
                [
                    aws_3DEP_directory,
                    "not in",
                    " ".join(result_gdf["directory"].to_list()),
                ]
            )
            sys.exit(message)
    
    hsfm.dataquery.plot_3DEP_bounds(result_gdf, bounds_gdf, qc_plot_output_directory=output_path)
    
    if len(result_gdf.index) != 1:
        print(
            "Multiple directories with laz data found on AWS.",
            "Rerun and specify a valid aws_3DEP_directory",
            "you would like to download data from. Options include",
            " ".join(result_gdf["directory"].to_list()),"Check bounds_qc_plot.png in",
            output_path,"directory for coverage."
        )

    else:
        aws_3DEP_directory = result_gdf["directory"].loc[0]
        
        # reduce bounds to extent of available data
        # may need to increase default 0.01 deg tiling else hangs if no data
        # within requested bounds.
        r_minx, r_miny, r_maxx, r_maxy = result_gdf.bounds.values[0]
        b_minx, b_miny, b_maxx, b_maxy = bounds_gdf.bounds.values[0]
        if r_minx > b_minx:
            minx = r_minx
        else:
            minx = b_minx
        if r_miny > b_miny:
            miny = r_miny
        else:
            miny = b_miny
        if r_maxx < b_maxx:
            maxx = r_maxx
        else:
            maxx = b_maxx
        if r_maxy < b_maxy:
            maxy = r_maxy
        else:
            maxy = b_maxy
        bounds = [maxx, miny, minx, maxy]
        print('Bounds with available data:',bounds)

        tiles, tile_polygons = hsfm.dataquery.divide_bounds_to_tiles(bounds, result_gdf)
        tile_polygons_gdf = gpd.GeoDataFrame({'geometry':tile_polygons})
        tile_polygons_gdf.crs = result_gdf.crs
        hsfm.dataquery.plot_3DEP_bounds(result_gdf, 
                                        bounds_gdf, 
                                        tile_polygons_gdf = tile_polygons_gdf,
                                        qc_plot_output_directory=output_path)
        
        print("Processing", len(tiles), "tiles.")
        
        c = 0
        for tile in tiles:
            result_gdf, bounds_gdf = hsfm.dataquery.get_3DEP_lidar_data_dirs(tile, cache_directory=cache_directory)
            try:
                output_path_tmp = os.path.join(output_path, str(c).zfill(5))
                pathlib.Path(output_path_tmp).mkdir(parents=True, exist_ok=True)
                
                pipeline_json_file, output_laz_file = hsfm.dataquery.create_3DEP_pipeline(
                    bounds_gdf,
                    aws_3DEP_directory,
                    epsg_code,
                    output_path=output_path_tmp,
                )

                hsfm.dataquery.run_3DEP_pdal_pipeline(pipeline_json_file, verbose=verbose)
                print(output_laz_file)

            #         output_dem_file = hsfm.dataquery.grid_3DEP_laz(output_laz_file, epsg_code, verbose=verbose)
                output_dem_file = grid_3DEP_multi_laz(output_path_tmp, epsg_code, verbose=verbose)

                out = os.path.join(output_path_tmp, DEM_file_name)
                os.rename(output_dem_file, out)

                if cleanup == True:
                    files = glob.glob(os.path.join(output_path_tmp, "*.laz"))
                    for i in files:
                        os.remove(i)
                    os.remove(pipeline_json_file)
                    files = glob.glob(os.path.join(output_path_tmp, "*log*.txt"))
                    for i in files:
                        os.remove(i)
                    files = glob.glob(os.path.join(output_path_tmp, "output*-DEM.tif"))
                    for i in files:
                        os.remove(i)
                c+=1
            except:
                shutil.rmtree(output_path_tmp)
                pass
            
        
        tmp = os.path.join(output_path, '*/*dem.tif')
        output_dem_file = os.path.join(output_path, 'output-DEM.tif')
        call = ['dem_mosaic',
                tmp,
                "--threads", str(psutil.cpu_count(logical=True)),
               '-o', output_dem_file]
        call = ' '.join(call)
        hsfm.utils.run_command2(call)
        if cleanup == True:
            files = glob.glob(tmp)
            for i in files:
                dir_path = str(pathlib.Path(i).parent.resolve())
                shutil.rmtree(dir_path)
            files = glob.glob(os.path.join(output_path, "*log*.txt"))
            for i in files:
                os.remove(i)
        out = os.path.join(output_path, DEM_file_name)
        os.rename(output_dem_file, out)
        print(out)
        print('DONE')
        return out
        
def divide_bounds_to_tiles(bounds,
                           result_gdf,
                           pad = 0.0001,
                           width = 0.01,
                           height = 0.01):
    xmin,ymin,xmax,ymax =  [bounds[2],bounds[1],bounds[0],bounds[3]]
    xmin,ymin,xmax,ymax = xmin-pad ,ymin-pad ,xmax+pad , ymax+pad
    rows = int(np.ceil((ymax-ymin) /  height))
    cols = int(np.ceil((xmax-xmin) / width))
    XleftOrigin = xmin
    XrightOrigin = xmin+width
    YtopOrigin = ymax
    YbottomOrigin = ymax-height
    tiles = []
    tile_polygons = []
    for i in range(cols):
        XleftOrigin_tmp = XleftOrigin -pad
        XrightOrigin_tmp = XrightOrigin +pad
        Ytop = YtopOrigin+pad
        Ybottom = YbottomOrigin-pad
        for j in range(rows):
            polygon = Polygon([(XleftOrigin_tmp, Ytop), 
                               (XrightOrigin_tmp, Ytop), 
                               (XrightOrigin_tmp, Ybottom), 
                               (XleftOrigin_tmp, Ybottom)])
            grid = gpd.GeoDataFrame({'geometry':[polygon]})
            grid.crs = result_gdf.crs
            tmp_gdf = gpd.overlay(grid,result_gdf)
            if not tmp_gdf.empty:
                tiles.append([XrightOrigin,Ybottom,XleftOrigin,Ytop])
                tile_polygons.append(polygon)
            Ytop = Ytop - height
            Ybottom = Ybottom - height
        XleftOrigin = XleftOrigin + width
        XrightOrigin = XrightOrigin + width
        
    return tiles, tile_polygons

def grid_3DEP_multi_laz(input_directory, 
                        epsg_code, 
                        target_resolution=1,
                        verbose=False):
    out_srs = "EPSG:" + str(epsg_code)
    call = ['parallel']
    sub_call = '"point2dem --nodata-value -9999 --t_srs ' + out_srs + \
    ' --threads '+ str(psutil.cpu_count(logical=True)) + ' --tr '+\
    str(target_resolution)+' {}"'
    call.append(sub_call)
    tmp = os.path.join(input_directory, '*.laz')
    call.extend([':::',tmp])
    call = ' '.join(call)
    hsfm.utils.run_command2(call,verbose=verbose)
    
    tmp = os.path.join(input_directory, '*DEM.tif')
    out = os.path.join(input_directory, 'output-DEM.tif')
    call = ['dem_mosaic',
            tmp,
            "--threads", str(psutil.cpu_count(logical=True)),
           '-o', out]
    call = ' '.join(call)
    hsfm.utils.run_command2(call,verbose=verbose)
    
    return out

def grid_3DEP_laz(laz_file, epsg_code, target_resolution=1, verbose=False):
    out_srs = "EPSG:" + str(epsg_code)
    call = [
        "point2dem",
        "--nodata-value",
        "-9999",
        "--threads", str(psutil.cpu_count(logical=True)),
        "--t_srs",
        out_srs,
        "--tr",
        str(target_resolution),
        laz_file,
    ]
    hsfm.utils.run_command(call, verbose=verbose)

    file_path = str(pathlib.Path(laz_file).parent.resolve())
    file_name = str(pathlib.Path(laz_file).stem)
    output_dem_file = os.path.join(file_path, file_name + "-DEM.tif")
    return output_dem_file


def run_3DEP_pdal_pipeline(pipeline_json_file, verbose=True):

    call = ["pdal", "pipeline", pipeline_json_file]
    if verbose:
        call.extend(["--verbose", "7"])
    hsfm.utils.run_command(call, verbose=verbose)


def create_3DEP_pipeline(
    bounds_gdf,
    aws_3DEP_directory,
    epsg_code,
    output_path="./",
    pipeline_json_file="pipeline.json",
    output_laz_file="output#.laz",
):
    pipeline_json_file = os.path.join(output_path, pipeline_json_file)
    output_laz_file = os.path.join(output_path, output_laz_file)

    base_url = "http://usgs-lidar-public.s3.amazonaws.com/"
    filename = os.path.join(base_url, aws_3DEP_directory, "ept.json")
    print('Downloading from',filename)

    lons, lats = bounds_gdf.to_crs("EPSG:3857").geometry.boundary.loc[0].xy
    lats = list(set(lats))
    lons = list(set(lons))
    bounds_str = "(" + str(lons) + "," + str(lats) + ")"

    out_srs = "EPSG:" + str(epsg_code)

    pipeline = {
        "pipeline": [
            {
                "type": "readers.ept",
                "filename": filename,
                "bounds": bounds_str,
                "threads": str(psutil.cpu_count(logical=True)),
            },
            {"type": "filters.returns", "groups": "first,only"},
            {"type": "filters.reprojection", "out_srs": out_srs},
            # using the splitter causes noisy data points and requires
            # filtering which takes more time. 
            # using the splitter doesn't seem to have a speed advantage.
#                         {
#                             "type": "filters.splitter",
#                             "length": "1000",
#                             "buffer": "10",
#                         },
#                         {
#                             "type":"filters.outlier",
#                             "method":"statistical",
#                             "mean_k":12,
#                             "multiplier":2.2
#                         },
#                         {
#                             "type":"filters.range",
#                             "limits":"Classification![7:7]"
#                         },

            output_laz_file,
        ]
    }

    with open(pipeline_json_file, "w") as f:
        json.dump(pipeline, f)

    return pipeline_json_file, output_laz_file


def get_3DEP_lidar_data_dirs(bounds, cache_directory="cache"):
    """
    bounds = [east, south, west, north]
    """
    fs = fsspec.filesystem("s3", anon=True)

    base_url = "s3://usgs-lidar-public/"
    aws_3DEP_directories = fs.ls(base_url)

    vertices = [
        (bounds[0], bounds[1]),
        (bounds[0], bounds[3]),
        (bounds[2], bounds[3]),
        (bounds[2], bounds[1]),
    ]

    bounds_polygon = Polygon(vertices)
    bounds_gdf = gpd.GeoDataFrame(
        gpd.GeoSeries(bounds_polygon), columns=["geometry"], crs="epsg:4326"
    )
    data_dirs_without_boundary_file = []

    pathlib.Path(cache_directory).mkdir(parents=True, exist_ok=True)
    out = os.path.join(cache_directory, "boundary.geojson")

    if os.path.isfile(out):
        df = gpd.read_file(out)
        result_gdf = gpd.overlay(df, bounds_gdf)

    else:
        df = gpd.GeoDataFrame(columns=["directory", "geometry"])
        for directory in aws_3DEP_directories:
            if os.path.isfile(out):
                gdf = gpd.read_file(out)
                gdf["directory"] = directory.split("/")[-1]
                df = df.append(gdf)
            else:
                try:
                    dir_url = "s3://" + directory
                    url = os.path.join(dir_url, "boundary.json")
                    gdf = gpd.read_file(fs.open(url, "rb"))
                    gdf["directory"] = directory.split("/")[-1]
                    df = df.append(gdf)
                except FileNotFoundError:
                    # not doing anything with this but could log
                    data_dirs_without_boundary_file.append(directory)
                    pass

        df.crs = bounds_gdf.crs
        df.to_file(out, driver="GeoJSON")
        result_gdf = gpd.overlay(df, bounds_gdf)

    return result_gdf, bounds_gdf


def get_UTM_EPSG_code_from_bounds(bounds):
    """
    bounds = [east, south, west, north]
    """
    east_south_epsg_code = hsfm.geospatial.lon_lat_to_utm_epsg_code(
        bounds[0], bounds[1]
    )
    west_north_epsg_code = hsfm.geospatial.lon_lat_to_utm_epsg_code(
        bounds[2], bounds[3]
    )

    if east_south_epsg_code == west_north_epsg_code:
        epsg_code = west_north_epsg_code
        return epsg_code
    else:
        print("Bounds span two UTM zones.")
        print(
            "EPSG:" + west_north_epsg_code,
            "and",
            "EPSG:" + east_south_epsg_code,
        )
        print("Using", "EPSG:" + west_north_epsg_code)
        epsg_code = west_north_epsg_code
        return epsg_code


def plot_3DEP_bounds(result_gdf, 
                     bounds_gdf, 
                     tile_polygons_gdf = None,
                     qc_plot_output_directory=None):
    """
    takes outputs from tools.get_3DEP_lidar_data_dirs()

    results_gdf: geopandas.geodataframe.GeoDataFrame
    bounds_gdf: geopandas.geodataframe.GeoDataFrame

    """

    bounds_gdf["coords"] = bounds_gdf["geometry"].apply(
        lambda x: x.representative_point().coords[:]
    )
    result_gdf["coords"] = result_gdf["geometry"].apply(
        lambda x: x.representative_point().coords[:]
    )

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    fig, ax = plt.subplots(figsize=(10, 10))

    for i, v in enumerate(result_gdf.index):
        result_gdf.loc[result_gdf.index == i].plot(
            ax=ax, edgecolor=colors[i], facecolor="none", linewidth=3
        )
        
    if not isinstance(tile_polygons_gdf,type(None)):
        tile_polygons_gdf["coords"] = tile_polygons_gdf["geometry"].apply(
            lambda x: x.representative_point().coords[:]
        )
        tile_polygons_gdf.plot(ax=ax, edgecolor="black", facecolor="none")
        for idx, row in tile_polygons_gdf.iterrows():
            plt.annotate(
                s=str(idx), xy=row["coords"][0], horizontalalignment="center"
            )
    
    bounds_gdf.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=1)

    try:
        ctx.add_basemap(
            ax,
            source="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            crs=bounds_gdf.crs.to_string(),
            alpha=0.5,
        )
    except:
        # if fails the bounds are likely too small to pull a tile
        pass

    for idx, row in result_gdf.iterrows():
        plt.annotate(
            s=row["directory"], xy=row["coords"][0], horizontalalignment="center"
        )

    if qc_plot_output_directory:
        out = os.path.join(qc_plot_output_directory, "bounds_qc_plot.png")
        plt.tight_layout()
        plt.savefig(out, bbox_inches="tight", pad_inches=0.1)