import os
import glob


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
               reference_dem_file,
               output_path,
               focal_length            = None,
               pixel_pitch             = None,
               crs                     = 'EPSG::4326',
               image_matching_accuracy = 4,
               densecloud_quality      = 4,
               keypoint_limit          = 40000,
               tiepoint_limit          = 4000,
               rotation_enabled        = True):

    """
    image_matching_accuracy = Highest/High/Medium/Low/Lowest -> 0/1/2/4/8
    densecloud_quality      = Ultra/High/Medium/Low/Lowest   -> 1/2/4/8/16
    """
    
    # Levels from https://www.agisoft.com/forum/index.php?topic=11697.msg52455#msg52455

    # TODO
    # check if project already exists and prompt for overwrite or pickup where left off
    # doc.open(output_path + project_name + ".psx")
    
    import Metashape

    os.makedirs(output_path)
    doc = Metashape.Document()
    doc.save(output_path + project_name + ".psx")

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
    
#     for i,v in enumerate(chunk.cameras):
#         v.reference.rotation_enabled = rotation_enabled

    chunk.cameras[1].reference.rotation_enabled = rotation_enabled

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
    
    chunk.exportReport(output_path + project_name + "_report.pdf")

    output_file = output_path + project_name + ".las"

    chunk.exportPoints(path=output_file,
                       format=Metashape.PointsFormatLAS, crs = chunk.crs)

    return output_file


def las2dem(project_name,
            output_path,,
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
                 split_in_blocks = False):
                 
    import Metashape

    doc = Metashape.Document()
    doc.open(output_path + project_name + ".psx")
    doc.read_only = False

    chunk = doc.chunk

    chunk.buildOrthomosaic(surface_data=Metashape.ElevationData)

    doc.save()

    chunk.exportRaster(output_path + project_name + "_orthomosaic.tif",
                       source_data= Metashape.OrthomosaicData,
                       split_in_blocks = split_in_blocks)

def get_estimated_camera_centers(project_file_path):
    
    import Metashape
    
    doc = Metashape.Document()
    doc.open(project_file_path)
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
            lon, lat, alt, yaw, pitch, roll, omega, phi, kappa = None, None, None, None, None, None, None, None, None
            
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
