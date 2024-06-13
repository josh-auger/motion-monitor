#!/usr/bin/env python3

# Title: extract_params_from_transform_files.py

# Description:
# Read an input directory of transform files (*.tfm, *.txt) and compile transform parameters into a list array.

# Created on: June 2024
# Created by: Joshua Auger (joshua.auger@childrens.harvard.edu), Computational Radiology Lab, Boston Children's Hospital

# Example run command: python3 compile_transform_files.py /path/to/your/directory

import os
import numpy as np

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
            return np.array(parameters + fixed_parameters)
        else:
            raise ValueError(f"File {file_path} is missing required parameters.")


def get_data_from_transforms(directory_path):
    transform_list = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt") or filename.endswith(".tfm"):
            file_path = os.path.join(directory_path, filename)
            try:
                transform_params = parse_transform_file(file_path)
                transform_list.append(transform_params)
            except ValueError as e:
                print(e)

    return transform_list


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compile transform parameters from transform files.")
    parser.add_argument("directory_path", type=str, help="Path to the directory containing transform files.")

    args = parser.parse_args()
    directory_path = args.directory_path

    transforms = get_data_from_transforms(directory_path)

    for i, transform in enumerate(transforms):
        print(f"Transform {i}: {transform}")
