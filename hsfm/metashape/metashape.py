import os
import glob
import pandas as pd
import sys

import hsfm

"""
Agisoft Metashape processing pipeline.
"""

# TODO add logging output

### SETUP
# TODO handle import Metashape with class structure


def authentication(METASHAPE_LICENCE_FILE):
    metashape_licence_file_symlink = os.path.join(
        os.getcwd(), os.path.basename(METASHAPE_LICENCE_FILE)
    )
    if not os.path.exists(metashape_licence_file_symlink):
        os.symlink(METASHAPE_LICENCE_FILE, metashape_licence_file_symlink)


def images2las(
    project_name,
    images_path,
    images_metadata_file,
    output_path,
    focal_length=None,
    pixel_pitch=None,
    camera_model_xml_file=None,
    crs="EPSG::4326",
    image_matching_accuracy=4,
    densecloud_quality=4,
    keypoint_limit=80000,
    tiepoint_limit=8000,
    rotation_enabled=True,
    export_point_cloud=True,
):

    # Levels from https://www.agisoft.com/forum/index.php?topic=11697.msg52455#msg52455
    """
    image_matching_accuracy = Highest/High/Medium/Low/Lowest -> 0/1/2/4/8
    densecloud_quality      = Ultra/High/Medium/Low/Lowest   -> 1/2/4/8/16
    """

    import Metashape

    # PROJECT SETUP

    # TODO
    # check if project already exists and prompt for overwrite or pickup where left off
    # doc.open(output_path + project_name + ".psx")

    os.makedirs(output_path)
    metashape_project_file = os.path.join(output_path, project_name + ".psx")
    report_file = os.path.join(output_path, project_name + "_report.pdf")
    point_cloud_file = os.path.join(output_path, project_name + ".las")

    doc = Metashape.Document()
    doc.save(metashape_project_file)

    crs = Metashape.CoordinateSystem(crs)

    chunk = doc.chunk
    if len(doc.chunks):
        chunk = doc.chunk
    else:
        chunk = doc.addChunk()

    metashape_metadata_df = pd.read_csv(images_metadata_file)
    image_file_names = list(metashape_metadata_df["image_file_name"].values)

    # can pass directory or list of image files if spread accross directories
    if isinstance(images_path, type("")):
        image_file_paths = sorted(glob.glob(os.path.join(images_path, "*.tif")))
    elif isinstance(images_path, type([])):
        image_file_paths = images_path

    image_files_subset = []
    for img_fn in image_file_names:
        for img_fp in image_file_paths:
            if img_fn in img_fp:
                image_files_subset.append(img_fp)
    chunk.addPhotos(image_files_subset)

    # DEFINE EXTRINSICS
    chunk.importReference(
        images_metadata_file,
        columns="nxyzXYZabcABC",  # from metashape py api docs
        delimiter=",",
        format=Metashape.ReferenceFormatCSV,
    )

    chunk.crs = crs
    chunk.updateTransform()

    for i, v in enumerate(chunk.cameras):
        v.reference.rotation_enabled = rotation_enabled

    #     # DEFINE INTRINSICS
    #     if isinstance(focal_length, type(None)) and isinstance(camera_model_xml_file, type(None)):
    if isinstance(camera_model_xml_file, type(None)):
        # try to grab a focal length for every camera from metadata in case run on mix match of cameras
        try:
            df_tmp = pd.read_csv(images_metadata_file)
            #             focal_length  = df_tmp['focal_length'].values[0]
            focal_lengths = df_tmp["focal_length"].values
            print("Focal length:", focal_length)
        except:
            #             print('No focal length specified.')
            pass

    if not isinstance(camera_model_xml_file, type(None)):
        calib = Metashape.Calibration()
        calib.load(camera_model_xml_file)
        for i, v in enumerate(chunk.cameras):
            #             v.sensor.calibration = calib
            v.sensor.user_calib = calib
            #             v.sensor.fixed_calibration = True
            v.sensor.fixed_params = ["F", "K1", "K2", "K3"]
    #             v.sensor.fixed_params=['F','Cx','Cy','K1','K2','K3','P1','P2']

    elif not isinstance(focal_length, type(None)):
        print("Focal length:", focal_length)
        for i, v in enumerate(chunk.cameras):
            # #             Optionally assign seperate camera model to each image
            #             sensor = chunk.addSensor()
            #             sensor.label = "Calibration Group "+str(i)
            #             sensor.type = Metashape.Sensor.Type.Frame
            #             sensor.width = v.photo.image().width
            #             sensor.height = v.photo.image().height
            #             v.sensor = sensor
            #             v.sensor.focal_length = focal_length
            v.sensor.focal_length = focal_lengths[i]
            v.sensor.fixed_params = ["F"]

    if not isinstance(pixel_pitch, type(None)):
        for i, v in enumerate(chunk.cameras):
            v.sensor.pixel_height = pixel_pitch
            v.sensor.pixel_width = pixel_pitch
    else:
        print("Please specify pixel pitch.")
        sys.exit()

    doc.save()

    # BUNDLE ADJUSTMENT

    #     for i,v in enumerate(chunk.cameras):
    #         v.sensor.photo_params = ['Cx', 'Cy']

    chunk.matchPhotos(
        downscale=image_matching_accuracy,
        generic_preselection=True,
        reference_preselection=False,
        keypoint_limit=keypoint_limit,
        tiepoint_limit=tiepoint_limit,
    )

    chunk.alignCameras()

    #     chunk.optimizeCameras(fit_f=False,
    #                           fit_k1=True,
    #                           fit_k2=True,
    #                           fit_k3=True)

    doc.save()

    # FILTER SPARSE CLOUD AND OPTIMIZE CAMERAS

    #     threshold = 3
    #     f = Metashape.PointCloud.Filter()
    #     f.init(chunk, criterion = Metashape.PointCloud.Filter.ImageCount)
    #     f.removePoints(threshold)
    #     chunk.optimizeCameras()

    #     threshold = 15
    #     f = Metashape.PointCloud.Filter()
    #     f.init(chunk, criterion = Metashape.PointCloud.Filter.ReconstructionUncertainty)
    #     f.removePoints(threshold)
    #     chunk.optimizeCameras()

    #     threshold = 5
    #     f = Metashape.PointCloud.Filter()
    #     f.init(chunk, criterion = Metashape.PointCloud.Filter.ProjectionAccuracy)
    #     f.removePoints(threshold)
    #     chunk.optimizeCameras()

    #     threshold = 1
    #     f = Metashape.PointCloud.Filter()
    #     f.init(chunk, criterion = Metashape.PointCloud.Filter.ReprojectionError)
    #     f.removePoints(threshold)
    #     chunk.optimizeCameras()

    #     doc.save()

    # BUILD DENSE CLOUD

    chunk.buildDepthMaps(
        downscale=densecloud_quality, filter_mode=Metashape.MildFiltering
    )
    chunk.buildDenseCloud()
    doc.save()

    # EXPORT

    chunk.exportReport(report_file)
    if export_point_cloud:
        chunk.exportPoints(
            path=point_cloud_file, format=Metashape.PointsFormatLAS, crs=chunk.crs
        )

    return metashape_project_file, point_cloud_file


def oc32dem(project_name, output_path, split_in_blocks=False, resolution=2):

    import Metashape

    doc = Metashape.Document()
    doc.open(output_path + project_name + ".psx")
    doc.read_only = False

    chunk = doc.chunk

    chunk.buildDem(
        source_data=Metashape.DenseCloudData,
        interpolation=Metashape.DisabledInterpolation,
    )

    doc.save()

    chunk.exportRaster(
        output_path + project_name + "_DEM.tif",
        source_data=Metashape.ElevationData,
        image_format=Metashape.ImageFormatTIFF,
        format=Metashape.RasterFormatTiles,
        nodata_value=-32767,
        save_kml=False,
        save_world=False,
        split_in_blocks=split_in_blocks,
        resolution=resolution,
    )


def images2ortho(
    project_name, output_path, build_dem=True, split_in_blocks=False, iteration=0
):

    import Metashape

    ortho_file = os.path.join(output_path, project_name + "_orthomosaic.tif")

    doc = Metashape.Document()
    doc.open(output_path + project_name + ".psx")
    doc.read_only = False

    chunk = doc.chunk

    if build_dem:
        chunk.buildDem(source_data=Metashape.DenseCloudData)

    chunk.buildOrthomosaic(surface_data=Metashape.ElevationData)

    doc.save()

    chunk.exportRaster(
        ortho_file,
        source_data=Metashape.OrthomosaicData,
        split_in_blocks=split_in_blocks,
    )


def get_estimated_camera_centers(metashape_project_file):

    import Metashape

    doc = Metashape.Document()
    doc.open(metashape_project_file)
    chunk = doc.chunk

    images = []
    lons = []
    lats = []
    alts = []
    yaws = []
    pitches = []
    rolls = []
    omegas = []
    phis = []
    kappas = []

    T = chunk.transform.matrix

    for camera in chunk.cameras:
        image = camera.label

        try:
            lon, lat, alt = chunk.crs.project(
                T.mulp(camera.center)
            )  # estimated camera positions
            m = chunk.crs.localframe(
                T.mulp(camera.center)
            )  # transformation matrix to the LSE coordinates in the given point
            R = m * T * camera.transform * Metashape.Matrix().Diag([1, -1, -1, 1])
            row = list()
            for j in range(0, 3):  # creating normalized rotation matrix 3x3
                row.append(R.row(j))
                row[j].size = 3
                row[j].normalize()
            R = Metashape.Matrix([row[0], row[1], row[2]])
            yaw, pitch, roll = Metashape.utils.mat2ypr(
                R
            )  # estimated orientation angles
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


def update_ba_camera_metadata(
    metashape_project_file, metashape_metadata_csv, image_file_extension=".tif"
):
    """
    Returns dataframe with bundle adjusted camera positions and camera positions for cameras
    that were not able to be aligned.
    """

    metashape_metadata_df = pd.read_csv(metashape_metadata_csv)

    metashape_export = hsfm.metashape.get_estimated_camera_centers(
        metashape_project_file
    )
    (
        images,
        lons,
        lats,
        alts,
        yaws,
        pitches,
        rolls,
        omegas,
        phis,
        kappas,
    ) = metashape_export
    images = [s + image_file_extension for s in images]

    dict = {
        "image_file_name": images,
        "lon": lons,
        "lat": lats,
        "alt": alts,
        "lon_acc": 1000,
        "lat_acc": 1000,
        "alt_acc": 1000,
        "yaw": yaws,
        "pitch": pitches,
        "roll": rolls,
        "yaw_acc": 180,
        "pitch_acc": 10,
        "roll_acc": 10,
    }

    ba_cameras_df = pd.DataFrame(dict)

    unaligned_cameras_file_names = ba_cameras_df[ba_cameras_df.isnull().any(axis=1)][
        "image_file_name"
    ].values

    ba_cameras_df = ba_cameras_df.dropna().reset_index(drop=True)
    #     for i in unaligned_cameras_file_names:
    #         ba_cameras_df[ba_cameras_df['image_file_name'].str.contains(i)] = \
    #         metashape_metadata_df[metashape_metadata_df['image_file_name'].str.contains(i)].values

    unaligned_cameras_df = metashape_metadata_df[
        metashape_metadata_df["image_file_name"].isin(unaligned_cameras_file_names)
    ]
    unaligned_cameras_df = unaligned_cameras_df.reset_index(drop=True)

    return ba_cameras_df, unaligned_cameras_df


def determine_clusters(metashape_project_file):
    import Metashape

    print("Determining if seperate image cluster subsets present.")

    # adapted from https://www.agisoft.com/forum/index.php?topic=6989.0

    doc = Metashape.Document()
    doc.open(metashape_project_file)
    chunk = doc.chunk

    point_cloud = chunk.point_cloud
    points = point_cloud.points
    point_proj = point_cloud.projections
    npoints = len(points)

    photo_matches = {}

    for photo in chunk.cameras:
        try:
            total = set()  # only valid
            point_index = 0
            proj = point_proj[photo]

            for cur_point in proj:
                track_id = cur_point.track_id
                while point_index < npoints and points[point_index].track_id < track_id:
                    point_index += 1
                    if (
                        point_index < npoints
                        and points[point_index].track_id == track_id
                    ):
                        if point_cloud.points[point_index].valid:
                            total.add(point_index)
            photo_matches[photo] = total
        except:
            print("Skipping unaligned image:", photo)

    m = []
    for i in range(0, len(chunk.cameras) - 1):
        if chunk.cameras[i] not in photo_matches.keys():
            continue
        for j in range(i + 1, len(chunk.cameras)):
            if chunk.cameras[j] not in photo_matches.keys():
                continue
            matches = photo_matches[chunk.cameras[i]] & photo_matches[chunk.cameras[j]]
            if len(matches) > 3:
                try:
                    pos_i = chunk.crs.project(
                        chunk.transform.matrix.mulp(chunk.cameras[i].center)
                    )
                    pos_j = chunk.crs.project(
                        chunk.transform.matrix.mulp(chunk.cameras[j].center)
                    )
                    m.append((chunk.cameras[i].label, chunk.cameras[j].label))
                except:
                    continue
            else:
                continue

    subsets = hsfm.core.find_sets(m)
    return subsets
