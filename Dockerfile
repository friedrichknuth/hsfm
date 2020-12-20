FROM continuumio/miniconda

RUN apt-get update -y && \
	apt-get install -y wget
#   && \ apt-get install -y git 
#	&& \ rm -rf /var/lib/apt/lists/*

RUN conda --version

ADD environment.yml /root/hsfm/environment.yml

RUN conda env create -f /root/hsfm/environment.yml

# DOWNLOAD ASP AND METASHAPE
RUN wget  https://github.com/NeoGeographyToolkit/StereoPipeline/releases/download/v2.6.2/StereoPipeline-2.6.2-2019-06-17-x86_64-Linux.tar.bz2 -P /root/
RUN wget http://download.agisoft.com/Metashape-1.6.4-cp35.cp36.cp37-abi3-linux_x86_64.whl -P /root/ 

#I FOUND THESE TO BE NECEESARY TO IMPORT HSFM
RUN apt-get install -y libgl1-mesa-glx
RUN /opt/conda/envs/hsfm/bin/pip install scikit-image

# COPY IN HSFM LIBRARY, LICENSE FILE
COPY .  /root/hsfm/
# COPY metashape_trial.lic .
COPY uw_agisoft.lic .

#  INSTALL HSFM TO CONDA ENV
RUN /opt/conda/envs/hsfm/bin/pip install -e /root/hsfm/

# ENV agisoft_LICENSE='$HOME/metashape_trial.lic'

# INSTALL ASP

RUN tar xvfj /root/StereoPipeline-2.6.2-2019-06-17-x86_64-Linux.tar.bz2
ENV PATH=${PATH}:/StereoPipeline-2.6.2-2019-06-17-x86_64-Linux/bin

#  INSTALL METASHAPE TO CONDA ENV
RUN /opt/conda/envs/hsfm/bin/pip install /root/Metashape-1.6.4-cp35.cp36.cp37-abi3-linux_x86_64.whl

# MOVE TO THIS FOLDER SO THE SCRIPT WORKS RIGHT
WORKDIR /root/hsfm/examples/

# ADD GDAL CLI TOOLS TO PATH
ENV PATH=${PATH}:/opt/conda/envs/hsfm/bin/

RUN apt-get install -y build-essential

EXPOSE 5053 5147


# RUN AN APP

ENTRYPOINT ["conda", "run", "-n", "hsfm", "python", "validate_license.py"] 


# RUN /opt/conda/envs/hsfm/bin/python Metashape_end_to_end_processing_example.py
# ENTRYPOINT ["/opt/conda/envs/hsfm/bin/python Metashape_end_to_end_processing_example.py"]
# ENTRYPOINT ["conda", "run", "-n", "hsfm", "python", "Metashape_end_to_end_processing_example.py"] 