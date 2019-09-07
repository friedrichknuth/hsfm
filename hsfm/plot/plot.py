from demcoreg import dem_mask
import matplotlib.pyplot as plt
import numpy as np
import os
from osgeo import gdal
import rasterio

import hsfm.io
import hsfm.geospatial

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
                                                left_fiducial,
                                                top_fiducial,
                                                right_fiducial,
                                                bottom_fiducial,
                                                principal_point,
                                                image_base_name,
                                                output_directory='qc/image_preprocessing/'):
                                                
    hsfm.io.create_dir(output_directory)
    
    fig,ax = plt.subplots(1, figsize=(8,8))
    ax.set_aspect('equal')
    ax.grid()
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
               label='Principle Point',
               color='red')
    ax.plot([left_fiducial[0],right_fiducial[0]], 
            [left_fiducial[1],right_fiducial[1]],
            color='k', lw=0.1)
    ax.plot([top_fiducial[0],bottom_fiducial[0]], 
            [top_fiducial[1],bottom_fiducial[1]],
            color='k', lw=0.1)
    ax.legend()
    
    plt.imshow(image_array, alpha=0.3)
    
    if output_directory == None:
        plt.show()
    
    else:
        output_file_name = os.path.join(output_directory,image_base_name+'_pp_and_fiducial_location.jpg')
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
    
    