# Use the existing Dockerfile as the base image
FROM ubuntu:22.04

# Set the locale
RUN DEBIAN_FRONTEND=noninteractive apt-get clean && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y locales locales-all
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# Install Python and required libraries
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        python3 \
        python3-pip \
        python3-dev \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install SimpleITK numpy matplotlib pytz pandas matplotlib opencv-python

# Copy the Python scripts into the container
COPY extract_params_from_log.py             /app/
COPY extract_params_from_transform_files.py /app/
COPY compute_motion_measures.py             /app/
COPY compute_displacement.py                /app/
COPY generate_motion_plots.py               /app/
COPY monitor_directory.py                   /app/
COPY mjpeg_server_module.py                 /app/

# Set the working directory inside the container
WORKDIR /working

# Set the entry point for the container to run the Python script with command-line arguments
CMD python3 /app/monitor_directory.py /working/ --radius=$HEAD_RADIUS --threshold=$MOTION_THRESH
#CMD python3 /app/mjpeg_server_example.py
