import os
import glob


"""
Agisoft Metashape processing pipeline.
"""

# TODO add logging output

### SETUP
# TODO import of hsfm.metapipe should prompt for licence if not found.

def authentication():
    METAHSAPE_LICENCE_FILE = '/opt/metashape-pro/uw_agisoft.lic'
    metashape_licence_file_symlink = os.path.join(os.getcwd(),
                                                  os.path.basename(METAHSAPE_LICENCE_FILE))
    if not os.path.exists(metashape_licence_file_symlink):
        os.symlink(METAHSAPE_LICENCE_FILE,
                   metashape_licence_file_symlink)
#
#
authentication()
import Metashape



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
               tiepoint_limit          = 4000):

    """
    image_matching_accuracy = Highest/High/Medium/Low/Lowest -> 0/1/4/8/16
    densecloud_quality      = Ultra/High/Medium/Low/Lowest   -> 1/2/4/8/16
    """

    # TODO
    # check if project already exists and prompt for overwrite or pickup where left off
    # doc.open(output_path + project_name + ".psx")

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
            output_path):

    doc = Metashape.Document()
    doc.open(output_path + project_name + ".psx")
    doc.read_only = False

    chunk = doc.chunk

    chunk.buildDem(source_data=Metashape.DenseCloudData, 
                   interpolation=Metashape.DisabledInterpolation)

    doc.save()

    chunk.exportRaster(output_path + project_name + "_DEM.tif", 
                       source_data= Metashape.ElevationData,
                       image_format=Metashape.ImageFormatTIFF, 
                       format=Metashape.RasterFormatTiles, 
                       nodata_value=-32767, 
                       save_kml=False, 
                       save_world=True)

def images2ortho(project_name,
                 output_path):

    doc = Metashape.Document()
    doc.open(output_path + project_name + ".psx")
    doc.read_only = False

    chunk = doc.chunk

    chunk.buildOrthomosaic(surface_data=Metashape.ElevationData)

    doc.save()

    chunk.exportRaster(output_path + project_name + "_orthomosaic.tif",
                       source_data= Metashape.OrthomosaicData)

