FROM continuumio/miniconda

RUN apt-get update -y && \
	apt-get install -y wget  && \
	apt-get install -y build-essential

# Set up conda env
RUN conda --version
ADD environment.yml /root/hsfm/environment.yml
RUN conda env create -f /root/hsfm/environment.yml

# DOWNLOAD ASP AND METASHAPE
RUN wget  https://github.com/NeoGeographyToolkit/StereoPipeline/releases/download/v2.6.2/StereoPipeline-2.6.2-2019-06-17-x86_64-Linux.tar.bz2 -P /root/
RUN wget http://download.agisoft.com/Metashape-1.6.4-cp35.cp36.cp37-abi3-linux_x86_64.whl -P /root/ 

# HSFM Dependencies - I found these necessary to succesfully "import hsfm" in a python script
RUN apt-get install -y libgl1-mesa-glx
RUN /opt/conda/envs/hsfm/bin/pip install scikit-image


# Copy in HSFM source code and agisoft license
COPY .  /root/hsfm/
COPY uw_agisoft.lic /ro/ot/hsfm/

# Set environmental variable for Agisoft license activation
ENV agisoft_LICENSE='/root/hsfm/'

# Set environmental variable for mounted data drive
ENV hsfm_data_dir='/root/hsfm_data'

# Install HSFM (in editable mode) and Metashape to conda env
RUN /opt/conda/envs/hsfm/bin/pip install -e /root/hsfm/
RUN /opt/conda/envs/hsfm/bin/pip install /root/Metashape-1.6.4-cp35.cp36.cp37-abi3-linux_x86_64.whl


# Install ASP library and add to path (required by HSFM because HSFM relies on system calls)
RUN tar xvfj /root/StereoPipeline-2.6.2-2019-06-17-x86_64-Linux.tar.bz2
ENV PATH=${PATH}:/StereoPipeline-2.6.2-2019-06-17-x86_64-Linux/bin
# Add GDAL tools to path (?? required by HSFM and/or ASP ??)
ENV PATH=${PATH}:/opt/conda/envs/hsfm/bin/

# Move working directory to examples folder
WORKDIR /root/hsfm/examples/
# EXPOSE 5053 5147s

# Run an app
ENTRYPOINT ["conda", "run", "-n", "hsfm", "python", "validate_license.py"] 