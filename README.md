# Historical Structure from Motion
Automated SfM processing of historical aerial photographs for land surface elevation change detection and quantitative analysis.


### Features

#### Historical Image Preprocessing
- download imagery from archive url and process in-memory or save to disk
- automated fiducial marker template detection
- modular image enhancement and distortion correction functions to optimize SfM processing
- automated calculation and cropping about principal point
- exception flow to launch manual fiducial marker selection app
- various QA/QC routines to enter exception flows and attempt alternate approaches, as well as plot and report final quality of preprocessed images


#### Historical Camera Recreation
- pinhole camera model generation given
  - xyz camera center position
  - focal length
  - pixel pitch
  - flight heading
- alternate flow to launch manual heading selection app

#### NASA Ames Stereo Pipeline (ASP) Processing
- wrappers for multiple ASP functions in the SfM workflow
 - cam_gem
 - bundle_adjust
 - parrallel_stereo
 - dem_mosaic
 - point2dem
 - pc_align
 - geodiff
- use of [bare](https://github.com/friedrichknuth/bare) for quality control monitoring during ASP operations


#### Plotting / Geospatial Analysis Functions and Tools
- various plotting functions for intermediate and final DEM products
- convenience wrappers around common gdal operations
- tool to extract and generate elevation profile along transect

#### Reference Imagery and DEMs
- automated SRTM reference DEM download given bounding box
- automated Google Satellite basemap download as geotif given bounding box

### Examples
See [notebooks](./examples/) for processing examples.

### Installation
```
$ git clone https://github.com/friedrichknuth/hsfm.git
$ cd ./hsfm
$ conda create -f environment.yml
$ conda activate hsfm
$ pip install -e .
```

Download and install the [NASA AMes Stereo Pipeline](https://ti.arc.nasa.gov/tech/asr/groups/intelligent-robotics/ngt/stereo/)

Download and install the [Agisoft Metashape Python API](https://agisoft.freshdesk.com/support/solutions/articles/31000148930-how-to-install-metashape-stand-alone-python-module)

### Contributing

_hsfm_ contains modular libraries and tools that can be adapted to process various types of historical imagery. At this time, many of the methods have been customized to process NAGAP imagery, but can be deconstructed into more generalized upstream methods and classes.

For contribution guidelines and high-level TODO list please click [here](./CONTRIBUTING.md).

### References
NASA Ames Stereo Pipeline [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1345235.svg)](https://doi.org/10.5281/zenodo.1345235)
 
Beyer, Ross A., Oleg Alexandrov, and Scott McMichael. "The Ames Stereo Pipeline: NASA's open source software for deriving and processing terrain data." Earth and Space Science 5.9 (2018): 537-548.

Shean, David E., et al. "An automated, open-source pipeline for mass production of digital elevation models (DEMs) from very-high-resolution commercial stereo satellite imagery." ISPRS Journal of Photogrammetry and Remote Sensing 116 (2016): 101-117.