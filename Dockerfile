# Use the shared-base-build (from Dockerfile.base) as the base image
FROM shared-base-build

ENV MPLCONFIGDIR=/tmp/matplotlibdir

# Copy the Python scripts into the container
COPY extract_params_from_log.py             /app/
COPY extract_params_from_transform_files.py /app/
COPY compute_motion_measures.py             /app/
COPY compute_displacement.py                /app/
COPY generate_motion_plots.py               /app/
COPY monitor_directory.py                   /app/
COPY mjpeg_server_module.py                 /app/

# Set the working directory inside the container
WORKDIR /data

# Set the entry point for the container to run the Python script with command-line arguments
#ENTRYPOINT ["python3", "/app/compute_motion_measures.py"]
CMD ["python3", "/app/monitor_directory.py", "/data/"]
#CMD python3 /app/mjpeg_server_example.py
