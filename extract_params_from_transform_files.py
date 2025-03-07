#!/usr/bin/env python3

# Title: extract_params_from_transform_files.py

# Description:
# Read an input directory of transform files (*.tfm, *.txt) and compile transform parameters into a list array.

# Created on: June 2024
# Created by: Joshua Auger (joshua.auger@childrens.harvard.edu), Computational Radiology Lab, Boston Children's Hospital

# Example run command: python3 compile_transform_files.py /path/to/your/directory

import os
import numpy as np
import logging
import json

def parse_transform_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        parameters = None
        fixed_parameters = None

        for line in lines:
            if line.startswith("Parameters:"):
                parameters = list(map(float, line.split(":")[1].strip().split()))
            elif line.startswith("FixedParameters:"):
                fixed_parameters = list(map(float, line.split(":")[1].strip().split()))

        if parameters is not None and fixed_parameters is not None:
            transform_params = np.array(parameters + fixed_parameters)
            return transform_params
        else:
            raise ValueError(f"File {file_path} is missing required parameters.")


def get_ordered_filepaths(directory_path):
    filepaths = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt") or filename.endswith(".tfm"):
            file_path = os.path.join(directory_path, filename)
            filepaths.append(file_path)
    filepaths.sort()  # Order by filename

    return filepaths

def look_for_metadata_file(directory_path):
    metadatafiles = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            metadatafiles.append(os.path.join(directory_path, filename))

    if len(metadatafiles) == 0:
        raise FileNotFoundError(f"No JSON metadata file found in {directory_path}!")
    elif len(metadatafiles) > 1:
        raise ValueError(f"Multiple JSON metadata files found in {directory_path}!")

    return metadatafiles[0]

def get_data_from_transforms(directory_path):
    logging.info("Processing list of transform files...")
    filepaths = get_ordered_filepaths(directory_path)

    transform_list = []
    for file_path in filepaths:
        try:
            transform_params = parse_transform_file(file_path)
            transform_list.append(transform_params)
        except ValueError as e:
            logging.error(e)

    # Transform files do not contain metadata (i.e. SMS factor or number of slices per volume), default value = 1
    # Check for JSON file in directory_path
    # IF *.JSON file exists, read JSON for sms_factor and nslice_per_vol
    # ELSE, set sms_factor and nslices_per_vol to 1
    sms_factor = 1
    nslices_per_vol = 1
    try:
        metadatafile = look_for_metadata_file(directory_path)
        logging.info(f"\tFound metadatafile: {metadatafile}")
        with open(metadatafile, 'r') as f:
            metadata = json.load(f)

        sms_factor = metadata['MultibandAccelerationFactor']
        slice_timings = metadata['SliceTiming']
        nslices_per_vol = len(slice_timings)
    except FileNotFoundError:
        logging.info(f"\tNo metadata file found in {directory_path}. Using default values.")
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error parsing metadatafile: {e}. Using default values.")

    logging.info(f"\tSMS factor = {sms_factor}")
    logging.info(f"\tNum slices per volume: {nslices_per_vol}")
    logging.info(f"\tNum acquisitions per volume: {int(nslices_per_vol / sms_factor)}")

    logging.info(f"Number of extracted parameter sets: {len(transform_list)}")
    if len(transform_list) == 0:
        logging.info(f"\tERROR: No parameter sets found!")
    elif len(transform_list) == 1:
        logging.info(f"\tOnly found one transform file: {transform_list[0]} in directory: {directory_path}.\nCheck input directory and try again.")

    return transform_list, sms_factor, nslices_per_vol


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compile transform parameters from transform files.")
    parser.add_argument("directory_path", type=str, help="Path to the directory containing transform files.")

    args = parser.parse_args()
    directory_path = args.directory_path

    transforms = get_data_from_transforms(directory_path)

    for i, transform in enumerate(transforms):
        print(f"Transform {i}: {transform}")
