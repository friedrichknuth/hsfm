# Contribution Guidelines

Functions should be organized and added to the following classes, following [PEP8](http://www.python.org/dev/peps/pep-0008/) conventions.

### Libraries

#### hsfm/asp.asp.py 
Wrappers around ASP functions.

#### hsfm/batch.batch.py 
Wrappers around other hsfm functions for batch processing. 

#### hsfm/core.core.py 
Core data wrangling and preprocessing functions. 

#### hsfm/geospatial.geospatial.py
Geospatial data processing functions.

#### hsfm/image.image.py 
Basic image processing functions.

#### hsfm/io.io.py 
Basic io functions.

#### hsfm/plot.plot.py 
Functions to plot various products.

#### hsfm/trig.trig.py 
Basic calculations.

#### hsfm/utils.utils.py 
Wrappers around external command line tools.


### TODO
High priority
- test the existing processing pipeline (end-to-end-notebook) on other NAGAP sites and years. this is in process.

Medium priority
- implement stereo pair matching based on image footprints
- try to implement ip detection and matching against basemap to georectify imagery before intitial camera generation. 
 - initial tests with ASP ipfind ipmatch did not give good results. Likely due to inconsistent image characteristics between basemap geotif and NAGAP image used for testing.

Low priority
- add instructions to contribution guieleines to use black and flake8