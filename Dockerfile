FROM debian:latest

LABEL "maintainer"="Robbie Capps <rcapps3@student.gsu.edu>"
LABEL "group"="Molkov Lab, Georgia State University"

# Add and set the USER
USER root

# Create and go to the ~/Git directory
WORKDIR /home/$USER/Git
WORKDIR nemati-collab

# Add the fantasia dataset
ADD fantasia ./

# Set the WFDB_PATH variable for building the module
ENV WFDB_PATH ". /home/${USER}/Git/nemati-collab/"

WORKDIR /

RUN apt-get update && apt-get -y upgrade

# Update sources, install python3, git, and wget
RUN apt-get install -y \
    python3 \
    git \
    wget \
    make \
    gcc \
    g++ \
    python3-tk \
    python3-pip \
    cpio

RUN cd /tmp && \
  wget http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/11306/l_mkl_2017.2.174.tgz && \
    tar -xzf l_mkl_2017.2.174.tgz && \
      cd l_mkl_2017.2.174 && \
        sed -i 's/ACCEPT_EULA=decline/ACCEPT_EULA=accept/g' silent.cfg && \
	  ./install.sh -s silent.cfg && \
	    cd .. && \
	      rm -rf *
	      RUN echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/intel.conf && \
	        ldconfig && \
		  echo ". /opt/intel/bin/compilervars.sh intel64" >> /etc/bash.bashrc

# Cython for MKL
RUN pip3 install Cython

# Back to the ~/Git directory
WORKDIR /home/$USER/Git

# Install WFDB (for Physionet data)
RUN git clone https://github.com/MIT-LCP/wfdb.git && cd wfdb && ./configure && make install && make check

WORKDIR /

# Install GSL (for integration of the model)
RUN wget http://mirror.us-midwest-1.nexcess.net/gnu/gsl/gsl-2.3.tar.gz && tar xvf gsl-2.3.tar.gz && cd gsl-2.3 && ./configure && make && make install

# Install necessary python packages
RUN pip3 install numpy scipy requests pandas matplotlib cython
RUN pip3 install wfdb==1.3.7

WORKDIR /home/$USER/Git/nemati-collab/

ADD ./build.sh ./
ADD ./src/ ./
ADD ./scripts/ ./

RUN /bin/bash ./build.sh