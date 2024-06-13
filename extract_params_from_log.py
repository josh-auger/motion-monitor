#!/usr/bin/env python3

# Run with python3

# Other code dependencies:
# pyhelpers/compute_displacement.py

# Requires log filename argument ($LOG_FILE) from docker run command (start_motion_monitor.sh)

import re
import os
import sys
import argparse
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
from compute_displacement import compute_displacement

def find_lines_with_phrase(log_filename, line_search_phrase="FOR-REPORT", additional_search_phrase=None):
    output_lines = []
    with open(log_filename, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line_search_phrase in line:
                output_line = line
                if additional_search_phrase:
                    next_line_index = i + 1
                    if next_line_index < len(lines):
                        next_line = lines[next_line_index].strip()
                        if additional_search_phrase in next_line:
                            output_line += " " + next_line
                            i = next_line_index  # Move to the line after next_line
                output_lines.append(output_line)
            i += 1
    print("\nNum acquisition lines found:", len(output_lines))
    return output_lines

def extract_numbers_from_lines(lines, number_search_pattern=r'\[(.*?)\]'):
    extracted_numbers = []
    skipped_lines_count = 0
    skipped_lines = []

    for line in lines:
        match = re.search(number_search_pattern, line)
        if match:
            numbers_str = match.group(1)
            # Use a more permissive regex for floating-point numbers
            numbers = [float(num) for num in numbers_str.split()]
            extracted_numbers.append(numbers)
        else:
            skipped_lines_count += 1
            skipped_lines.append((line, "No match found"))

    print("Num extracted parameter sets:", len(extracted_numbers))
    if skipped_lines_count != 0:    # some lines were skipped - we need to know which/why!
        print("Skipped lines (missing end bracket, ']'):", skipped_lines_count)
        for skipped_line, error_message in skipped_lines:
            print(f"Line: {skipped_line}, Error: {error_message}")

    return extracted_numbers

def get_data_from_slimm_log(log_filename):
    # Extract multi-band (SMS) factor value
    sms_factor_line = find_lines_with_phrase(log_filename, "MultiBandFactor", "</value>")
    sms_factor, *_ = extract_numbers_from_lines(sms_factor_line[:1], r'<value>(.*?)<\/value>')
    sms_factor = float(sms_factor[0])  # unpack nested list value storage (used for saving transform parameters)
    print("\nSMS factor = ", sms_factor)

    # Extract number of slices per volume
    num_vol_slices_line = find_lines_with_phrase(log_filename, "Number of slices per volume:")
    nslices_per_vol, *_ = extract_numbers_from_lines(num_vol_slices_line, r'(\d+)\.$')
    nslices_per_vol = float(nslices_per_vol[0])
    print("Num slices per volume:", nslices_per_vol)
    print("Num acquisitions per volume:", int(nslices_per_vol / sms_factor))

    # Find all lines reporting transform parameters
    lines_with_params = find_lines_with_phrase(log_filename, line_search_phrase="FOR-REPORT", additional_search_phrase="Kalman filtering")
    # Extract parameter numbers from the lines and count skipped lines
    transform_list = extract_numbers_from_lines(lines_with_params, number_search_pattern=r'\[(.*?)\]')

    return transform_list, sms_factor, nslices_per_vol


if __name__ == "__main__":
    log_file = sys.argv[1]
    log_filename = "/data/" + log_file

    transform_list, sms_factor, nslices_per_vol = get_data_from_slimm_log(log_filename)







