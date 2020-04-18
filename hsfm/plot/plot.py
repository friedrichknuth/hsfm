import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from osgeo import gdal
import rasterio

import hsfm.io
import hsfm.geospatial

"""
Functions to plot various products.
"""

def plot_image_histogram(image_array, 
                         image_base_name,
                         output_directory='qc/image_histograms/',
                         suffix=None):
                   
    hsfm.io.create_dir(output_directory)
                   
    fig, ax = plt.subplots(1, figsize=(10, 10))
    n, bins, patches = ax.hist(img.ravel()[::40],
                                bins=256, 
                                range=(0,256),
                                color='steelblue',
                                edgecolor='none')
                                
    if output_directory == None:
        plt.imshow(img)
    
    else:
        output_file_name = os.path.join(output_directory,image_base_name+suffix+'.png')
        fig.savefig(output_file_name)
    
    plt.close()
    
    
def plot_principal_point_and_fiducial_locations(image_array,
                                                fiducials,
                                                principal_point,
                                                image_base_name,
                                                output_directory='qc/image_preprocessing/'):
                                                
    left_fiducial = fiducials[0]
    top_fiducial = fiducials[1]
    right_fiducial = fiducials[2]
    bottom_fiducial = fiducials[3]
    
    hsfm.io.create_dir(output_directory)
    
    fig,ax = plt.subplots(1, figsize=(8,8))
    ax.set_aspect('equal')
    # ax.grid()
    ax.invert_yaxis()
    ax.scatter(left_fiducial[0], left_fiducial[1],
               s=0.2, 
               label='Fiducials', 
               color='midnightblue')
    ax.scatter(top_fiducial[0], top_fiducial[1],
               s=0.2, 
               color='midnightblue')
    ax.scatter(right_fiducial[0], right_fiducial[1],
               s=0.2, 
               color='midnightblue')
    ax.scatter(bottom_fiducial[0], bottom_fiducial[1],
               s=0.2,
               color='midnightblue')
    ax.scatter(int(principal_point[0]), int(principal_point[1]),
               s=0.2,
               label='Principal Point',
               color='red')
    ax.plot([left_fiducial[0],right_fiducial[0]], 
            [left_fiducial[1],right_fiducial[1]],
            color='k', lw=0.1)
    ax.plot([top_fiducial[0],bottom_fiducial[0]], 
            [top_fiducial[1],bottom_fiducial[1]],
            color='k', lw=0.1)
    ax.legend()
    
    plt.imshow(image_array, alpha=0.9, cmap='gray')
    
    if output_directory == None:
        plt.show()
    
    else:
        output_file_name = os.path.join(output_directory,image_base_name+'_pp_and_fiducial_location.png')
        fig.savefig(output_file_name,dpi=300)
    
    plt.close()
    
def plot_dem_difference_map(masked_array,
                            output_file_name=None,
                            cmap='RdBu',
                            percentile_min=1,
                            percentile_max=99,
                            spread=None,
                            extent=None):
                      
    """
    Function to plot difference map between two DEMs from masked array. 
    Replaces fill values with nan. 
    Use hsfm.geospatial.mask_array_with_nan(array,nodata_value) to create an appropriate
    masked array as input.
    """
                                          
    if spread == None:
        lowerbound, upperbound = np.nanpercentile(masked_array,[percentile_min,percentile_max])
        spread = max([abs(lowerbound), abs(upperbound)])
    
    fig, ax = plt.subplots(1,figsize=(10,10))
    
    im = ax.imshow(masked_array,
                   cmap=cmap,
                   clim=(-spread, spread),
                   extent=extent)
    
    fig.colorbar(im,extend='both')
    
    if output_file_name == None:
        plt.show()
    
    else:
        fig.savefig(output_file_name, dpi=300)
        
def plot_dem_difference_from_file_name(dem_difference_file_name,
                                       output_file_name=None,
                                       cmap='RdBu',
                                       percentile_min=1,
                                       percentile_max=99,
                                       spread=None,
                                       extent=None,
                                       mask_glacier=False):
                      
    """
    Function to plot difference map between two DEMs from file.
    """
                                       
    from demcoreg import dem_mask
    
    rasterio_dataset = rasterio.open(dem_difference_file_name)
    array = rasterio_dataset.read(1)
    nodata_value = rasterio_dataset.nodata
    masked_array = hsfm.geospatial.mask_array_with_nan(array,
                                                       nodata_value)
    
    if mask_glacier == True:
        ds = gdal.Open(dem_difference_file_name)
        mask = dem_mask.get_icemask(ds)
        masked_array = np.ma.array(masked_array,mask=~mask)
        
    plot_dem_difference_map(masked_array,
                            output_file_name=output_file_name,
                            cmap=cmap,
                            percentile_min=percentile_min,
                            percentile_max=percentile_max,
                            spread=spread,
                            extent=extent)
        
def plot_dem_with_hillshade(masked_array,
                            output_file_name=None,
                            cmap='inferno'):
    """
    Function to plot DEM with hillshade. Uses a masked array with nans as fill value.
    Use hsfm.geospatial.mask_array_with_nan(array,nodata_value) to create an appropriate
    masked array as input.
    """
    hillshade = hsfm.geospatial.calculate_hillshade(masked_array)
    
    fig, ax = plt.subplots(1,figsize=(10,10))
    
    im = ax.imshow(masked_array, 
                   cmap=cmap)
    
    ax.imshow(hillshade, 
              cmap='gray',
              alpha=0.5)
    
    fig.colorbar(im,extend='both')

    if output_file_name == None:
        plt.show()
    
    else:
        fig.savefig(output_file_name, dpi=300)

def plot_dem_from_file(dem_file_name,
                       output_file_name=None,
                       cmap='inferno'):
    
    rasterio_dataset = rasterio.open(dem_file_name)
    array = rasterio_dataset.read(1)
    nodata_value = rasterio_dataset.nodata
    masked_array = hsfm.geospatial.mask_array_with_nan(array,
                                                       nodata_value)
    
    plot_dem_with_hillshade(masked_array,
                            output_file_name=output_file_name,
                            cmap=cmap)

def plot_intersection_angles_qc(intersections, file_names, show=False):
    df = pd.DataFrame({"Angle off mean":intersections,"filename":file_names}).set_index("filename")
    df_mean = df - df.mean()
    fig, ax = plt.subplots(1, figsize=(10, 10))
    df_mean.plot.bar(grid=True,ax=ax)
    if show:
        plt.show()
    fig.savefig('qc/image_preprocessing/principal_point_intersection_angle_off_mean.png')
    plt.close()
    angle = np.round((df.mean() - 90).values[0],4)
    print("Mean rotation off 90 degree intersection at principal point:",angle)
    print("Further QC plots for principal point and fiducial marker detection available under qc/image_preprocessing/")

    