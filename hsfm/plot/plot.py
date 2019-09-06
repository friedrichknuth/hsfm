import matplotlib.pyplot as plt
import os

import hsfm.io

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
    
def plot_masked_array(masked_array,
                      output_file_name=None,
                      cmap='gray',
                      percentile_min=1,
                      percentile_max=99,
                      extent=None):
                      
    """
    Function to plot masked array with nans as fill value.
    """
    
    lowerbound, upperbound = np.nanpercentile(masked_array,[percentile_min,percentile_max])
    
    fig, ax = plt.subplots(1,figsize=(10,10))
    im = ax.imshow(masked_array[0],
                   cmap='gray',
                   clim=(lowerbound, upperbound),
                   extent=extent)
    fig.colorbar(im,extend='both')
    
    if output_file_name == None:
        plt.show()
    
    else:
        fig.savefig(output_file_name, dpi=300)