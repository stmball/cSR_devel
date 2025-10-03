FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    build-essential \
    wget

# Add deadsnakes PPA
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.7
RUN apt-get install -y python3.7 python3.7-dev python3.7-distutils

# Install pip for Python 3.7
RUN curl -sS https://bootstrap.pypa.io/pip/3.7/get-pip.py | python3.7

# Set python and pip to use Python 3.7
RUN ln -sf /usr/bin/python3.7 /usr/bin/python \
    && ln -sf /usr/local/bin/pip /usr/bin/pip

# Set working directory
WORKDIR /MLapp

RUN git clone https://github.com/stmball/cSR_devel

# Set working directory
WORKDIR /MLapp/cSR_devel

# Update pip
RUN python3.7 -m pip install --upgrade pip

# Install dependencies
RUN python3.7 -m pip install -r requirements.txt

# Run the application
CMD ["python3.7", "-m", "csr.Data", "--help"]