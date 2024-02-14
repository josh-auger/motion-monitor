# Use the existing Dockerfile as the base image
FROM ubuntu:jammy

# Set the locale
RUN DEBIAN_FRONTEND=noninteractive apt-get clean && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y locales locales-all
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# Install Python and required libraries
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install SimpleITK numpy matplotlib

# Copy the Python scripts into the container
COPY extract_params_from_log.py /app/extract_params_from_log.py
COPY compute_displacement.py /app/compute_displacement.py

# Set the working directory inside the container
WORKDIR /data

# Set the entry point for the container to run the Python script with command-line arguments
#ENTRYPOINT ["python3", "extract_params_from_log.py"]
ENTRYPOINT ["python3", "hello_world.py"]

# Set the default command to an empty list
CMD []
