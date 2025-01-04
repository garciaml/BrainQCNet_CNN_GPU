Bootstrap: docker
From: pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

%labels
    Author MelanieGarcia
    Version 1.0
    Description Singularity image for CNN baseline models from BrainQCNet study - GPU version

%files
    . /app

%post
    # Set timezone during build
    export TZ=Europe
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime
    echo $TZ > /etc/timezone

    # Install tzdata package to ensure timezones are available
    apt-get update && apt-get install -y tzdata

    # Install dependencies
    apt-get update && apt-get install -y python3 \
       python3-venv \
       make \
       tk-dev \
       tcl-dev \
       libgl1-mesa-glx \
    && apt-get install -yq libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

    cd /app
    make all

%environment
    export HOME=/app
    export PATH=/app/venv/bin:$PATH
    cd /app

%runscript
    cd /app
    # Define the default run command for the container
    exec venv/bin/python3 run.py "$@"

