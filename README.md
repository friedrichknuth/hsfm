# Historical Structure from Motion 

Automated SfM processing of historical aerial photographs for land surface elevation change detection and quantitative analysis. 
[![DOI](https://zenodo.org/badge/202800494.svg)](https://zenodo.org/badge/latestdoi/202800494)


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

Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)  

After installing Miniconda set up [Mamba](https://mamba.readthedocs.io/en/latest/installation.html) (optional but recommended)
```
$ conda install mamba -n base -c conda-forge
```
Clone the repo and set up the conda environment  

```
$ git clone https://github.com/friedrichknuth/hsfm.git
$ cd ./hsfm
$ mamba env create -f environment.yml
$ conda activate hsfm
$ pip install -e .
```

Download and install the [NASA AMes Stereo Pipeline](https://stereopipeline.readthedocs.io/en/latest/installation.html)

Download and install the [Agisoft Metashape Python API](https://agisoft.freshdesk.com/support/solutions/articles/31000148930-how-to-install-metashape-stand-alone-python-module)

Check your installation
```
$ cd ./examples/scripts
$ python -u batch_pipeline.py
```

### Contributing

_hsfm_ contains modular libraries and tools that can be adapted to process various types of historical imagery. At this time, many of the methods have been customized to process NAGAP imagery, but can be deconstructed into more generalized upstream methods and classes.

For contribution guidelines and high-level TODO list please click [here](./CONTRIBUTING.md).

### Related software

[Historical Image Pre-Processing (HIPP)](https://github.com/friedrichknuth/hipp) [![DOI](https://zenodo.org/badge/287390486.svg)](https://zenodo.org/badge/latestdoi/287390486)

[NASA Ames Stereo Pipeline](https://stereopipeline.readthedocs.io/en/latest/introduction.html) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1345235.svg)](https://doi.org/10.5281/zenodo.1345235)

### References

Knuth, F., Shean, D., Bhushan, S., Schwat, E., Alexandrov, O., McNeil, C., Dehecq, A., Florentine, C. and O’Neel, S., 2023. Historical Structure from Motion (HSfM): Automated processing of historical aerial photographs for long-term topographic change analysis. Remote Sensing of Environment, 285, p.113379. [https://doi.org/10.1016/j.rse.2022.113379](https://doi.org/10.1016/j.rse.2022.113379)


 
Beyer, Ross A., Oleg Alexandrov, and Scott McMichael. "The Ames Stereo Pipeline: NASA's open source software for deriving and processing terrain data." Earth and Space Science 5.9 (2018): 537-548.

Shean, David E., et al. "An automated, open-source pipeline for mass production of digital elevation models (DEMs) from very-high-resolution commercial stereo satellite imagery." ISPRS Journal of Photogrammetry and Remote Sensing 116 (2016): 101-117.
