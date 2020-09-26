import os
import glob
import pandas as pd

import hsfm

"""
Agisoft Metashape processing pipeline.
"""

# TODO add logging output

### SETUP
# TODO handle import Metashape with class structure

def authentication(METASHAPE_LICENCE_FILE):
    metashape_licence_file_symlink = os.path.join(os.getcwd(),
                                                  os.path.basename(METASHAPE_LICENCE_FILE))
    if not os.path.exists(metashape_licence_file_symlink):
        os.symlink(METASHAPE_LICENCE_FILE,
                   metashape_licence_file_symlink)


def images2las(project_name,
               images_path,
               images_metadata_file,
               output_path,
               focal_length            = None,
               pixel_pitch             = None,
               crs                     = 'EPSG::4326',
               image_matching_accuracy = 4,
               densecloud_quality      = 4,
               keypoint_limit          = 40000,
               tiepoint_limit          = 4000,
               rotation_enabled        = True):

    # Levels from https://www.agisoft.com/forum/index.php?topic=11697.msg52455#msg52455
    """
    image_matching_accuracy = Highest/High/Medium/Low/Lowest -> 0/1/2/4/8
    densecloud_quality      = Ultra/High/Medium/Low/Lowest   -> 1/2/4/8/16
    """
    

    # TODO
    # check if project already exists and prompt for overwrite or pickup where left off
    # doc.open(output_path + project_name + ".psx")
    
    import Metashape

    os.makedirs(output_path)
    metashape_project_file = os.path.join(output_path, project_name  + ".psx")
    report_file            = os.path.join(output_path, project_name  + "_report.pdf")
    point_cloud_file       = os.path.join(output_path, project_name  + ".las")

    doc = Metashape.Document()
    doc.save(metashape_project_file)

    crs = Metashape.CoordinateSystem(crs)

    chunk = doc.chunk
    if len(doc.chunks):
        chunk = doc.chunk
    else:
        chunk = doc.addChunk()

    images = glob.glob(os.path.join(images_path,'*'))
    chunk.addPhotos(images)
    
    if focal_length:
        chunk.cameras[0].sensor.focal_length = focal_length
    
    if pixel_pitch:
        chunk.cameras[0].sensor.pixel_height = pixel_pitch
        chunk.cameras[0].sensor.pixel_width  = pixel_pitch

    chunk.importReference(images_metadata_file,
                          columns="nxyzXYZabcABC", # from metashape py api docs
                          delimiter=',',
                          format=Metashape.ReferenceFormatCSV)

    # optionally orient first camera
#     chunk.cameras[1].reference.rotation_enabled = rotation_enabled
    
    # need to iterate to orient all cameras if desired
    for i,v in enumerate(chunk.cameras):
        v.reference.rotation_enabled = rotation_enabled

    chunk.crs = crs
    chunk.updateTransform()

    doc.save()

    chunk.matchPhotos(downscale=image_matching_accuracy,
                      generic_preselection=True,
                      reference_preselection=False,
                      keypoint_limit=keypoint_limit,
                      tiepoint_limit=tiepoint_limit)

    chunk.alignCameras()


    doc.save()

    chunk.buildDepthMaps(downscale=densecloud_quality,
                         filter_mode=Metashape.MildFiltering)
    chunk.buildDenseCloud()
    doc.save()
    
    chunk.exportReport(report_file)
    chunk.exportPoints(path=point_cloud_file,
                       format=Metashape.PointsFormatLAS, crs = chunk.crs)

    return metashape_project_file, point_cloud_file


def las2dem(project_name,
            output_path,
            split_in_blocks = False,
            resolution = 2):
            
    import Metashape

    doc = Metashape.Document()
    doc.open(output_path + project_name + ".psx")
    doc.read_only = False

    chunk = doc.chunk

    chunk.buildDem(source_data=Metashape.DenseCloudData, 
                   interpolation=Metashape.DisabledInterpolation,
                   resolution=resolution)

    doc.save()

#     chunk.exportRaster(output_path + project_name + "_DEM.tif", 
#                        source_data= Metashape.ElevationData,
#                        image_format=Metashape.ImageFormatTIFF, 
#                        format=Metashape.RasterFormatTiles, 
#                        nodata_value=-32767, 
#                        save_kml=False, 
#                        save_world=False,
#                        split_in_blocks = split_in_blocks)

def images2ortho(project_name,
                 output_path,
                 split_in_blocks = False,
                 iteration       = 0):
                 
    import Metashape
    
    ortho_file = os.path.join(output_path, project_name  +"_orthomosaic.tif")

    doc = Metashape.Document()
    doc.open(output_path + project_name + ".psx")
    doc.read_only = False

    chunk = doc.chunk

    chunk.buildOrthomosaic(surface_data=Metashape.ElevationData)

    doc.save()

    chunk.exportRaster(ortho_file,
                       source_data= Metashape.OrthomosaicData,
                       split_in_blocks = split_in_blocks)

def get_estimated_camera_centers(metashape_project_file):
    
    import Metashape
    
    doc = Metashape.Document()
    doc.open(metashape_project_file)
    chunk = doc.chunk
    
    images  = []
    lons    = []
    lats    = []
    alts    = []
    yaws    = []
    pitches = []
    rolls   = []
    omegas  = []
    phis    = []
    kappas  = []
    
    T = chunk.transform.matrix
    
    for camera in chunk.cameras:
        image = camera.label
        
        try:
            lon, lat, alt = chunk.crs.project(T.mulp(camera.center)) #estimated camera positions
            m = chunk.crs.localframe(T.mulp(camera.center)) #transformation matrix to the LSE coordinates in the given point
            R = m * T * camera.transform * Metashape.Matrix().Diag([1, -1, -1, 1])
            row = list()
            for j in range (0, 3): #creating normalized rotation matrix 3x3
                row.append(R.row(j))
                row[j].size = 3
                row[j].normalize()
            R = Metashape.Matrix([row[0], row[1], row[2]])
            yaw, pitch, roll = Metashape.utils.mat2ypr(R) #estimated orientation angles
            omega, phi, kappa = Metashape.utils.mat2opk(R)
            
            
        except:
            lon, lat, alt, yaw, pitch, roll, omega, phi, kappa = [None] * 9
            
        images.append(image)
        lons.append(lon)
        lats.append(lat)
        alts.append(alt)
        yaws.append(yaw)
        pitches.append(pitch)
        rolls.append(roll)
        omegas.append(omega)
        phis.append(phi)
        kappas.append(kappa)
        
    return images, lons, lats, alts, yaws, pitches, rolls, omegas, phis, kappas

def update_ba_camera_metadata(metashape_project_file, 
                              metashape_metadata_csv,
                              image_file_extension = '.tif',
                              output_file_name = None,
                              xyz_acc = 180,
                              ypr_acc = 10):
    '''
    Returns dataframe with bundle adjusted camera positions and camera positions for cameras
    that were not bundle adjusted in case a match can be made in subsequent runs.
    '''
    
    metashape_metadata_df = pd.read_csv(metashape_metadata_csv)
    
    metashape_export = hsfm.metashape.get_estimated_camera_centers(metashape_project_file)
    images, lons, lats, alts, yaws, pitches, rolls, omegas, phis, kappas = metashape_export
    images = [s + image_file_extension for s in images]
    
    dict = {'image_file_name': images,
            'lon': lons,
            'lat': lats,
            'alt': alts,
            'lon_acc': xyz_acc,
            'lat_acc': xyz_acc,
            'alt_acc': xyz_acc,
            'yaw': yaws,
            'pitch': pitches,
            'roll': rolls,
            'yaw_acc': ypr_acc,
            'pitch_acc': ypr_acc,
            'roll_acc': ypr_acc,
           }  

    ba_camera_metadata = pd.DataFrame(dict)

    unaligned_cameras = ba_camera_metadata[ba_camera_metadata.isnull().any(axis=1)]\
    ['image_file_name'].values
    
    # replace unaligned cameras with values from original input metadata
    for i in unaligned_cameras:
        ba_camera_metadata[ba_camera_metadata['image_file_name'].str.contains(i)] = \
        metashape_metadata_df[metashape_metadata_df['image_file_name'].str.contains(i)].values
    
    if not isinstance(output_file_name, type(None)):
        ba_camera_metadata.to_csv(output_file_name, index = False)
    
    return ba_camera_metadata





    
    
    
