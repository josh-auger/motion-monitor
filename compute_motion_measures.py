#!/usr/bin/env python3

# Title: compute_motion_measures.py

# Description:
#

# Created on: June 2024
# Created by: Joshua Auger (joshua.auger@childrens.harvard.edu), Computational Radiology Lab, Boston Children's Hospital


import re
import os
import sys
import argparse
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
from compute_displacement import compute_displacement
# from extract_params_from_log import
# from extract_params_from_transform_files import compile_transforms


def create_euler_transform(parameters, rotation_center=[0.0, 0.0, 0.0]):
    euler_transform = sitk.Euler3DTransform()
    euler_transform.SetParameters(parameters)
    euler_transform.SetCenter(rotation_center)
    return euler_transform

def compute_transform_pair_displacement(extracted_numbers):
    num_instances = len(extracted_numbers)
    displacements = []
    for i in range(num_instances - 1):
        parameters_i = extracted_numbers[i]
        parameters_next = extracted_numbers[i + 1]

        transform_i = create_euler_transform(parameters_i)
        transform_next = create_euler_transform(parameters_next)

        displacement_value = compute_displacement(transform_i, transform_next)
        displacements.append(displacement_value)
    displacements.append(0) # final acquisition does not have next transform to compute displacement (yet)

    print("\nNum displacement values:", len(displacements))
    return displacements

def calculate_percent_diff(array1, array2):
    array1 = np.array(array1)
    array2 = np.array(array2)
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length.")
    percent_diff = np.abs((array2 - array1) / ((array1 + array2) / 2)) * 100

    return percent_diff

def check_volume_motion(displacements, sms_factor, num_slices_per_volume, threshold):
    # Calculate number of acquisitions per volume
    num_acquisitions_per_volume = int(num_slices_per_volume / sms_factor)
    total_volumes = (len(displacements) / num_acquisitions_per_volume) + 1 # plus 1 for the initial reference volume
    volumes_above_threshold = 0
    volume_id = []

    for i in range(0, len(displacements), num_acquisitions_per_volume):
        volume_count = i // num_acquisitions_per_volume + 1
        volume_displacements = displacements[i:i + num_acquisitions_per_volume]
        volume_id.extend([volume_count] * len(volume_displacements))
        if any(d > threshold for d in volume_displacements):
            volumes_above_threshold += 1

    print("Completed volumes (+ ref vol):", total_volumes)
    print("Volumes with motion:", volumes_above_threshold)
    print("Volumes without motion:", (total_volumes - volumes_above_threshold))
    return total_volumes, volumes_above_threshold, volume_id

def plot_parameters(extracted_numbers, indices_to_plot=[0, 1, 2, 3, 4, 5], log_filename="", titles=None, y_labels=None, rot_thresh=None, trans_thresh=None):
    valid_indices = all(0 <= index < len(extracted_numbers[0]) for index in indices_to_plot)
    if not valid_indices:
        print("Error: One (or more) indices are out of range.")
        return

    # Create an output folder if it doesn't exist
    log_file_path, log_file_name = os.path.split(log_filename)
    output_folder = os.path.join(log_file_path, f"{os.path.splitext(log_file_name)[0]}_outputs")
    os.makedirs(output_folder, exist_ok=True)

    num_indices = len(indices_to_plot)
    fig, axes = plt.subplots(num_indices, 1, figsize=(8, 4 * num_indices))
    subplot_colors = ['b', 'g', 'r', 'c', 'm', 'y']  # Default colors for subsequent plots

    for i, index in enumerate(indices_to_plot):
        numbers_to_plot = [numbers[index] for numbers in extracted_numbers]
        if num_indices > 1:
            ax = axes[i]
        else:
            ax = axes

        color = subplot_colors[i % len(subplot_colors)]  # Use modulo to cycle through colors
        ax.plot(numbers_to_plot, marker='o', linestyle='-', color=color, alpha=0.7)
        if titles is not None:
            ax.set_title(titles[i])
        else:
            ax.set_title(f'{index + 1}-th parameter of each registration instance')
        ax.set_xlabel('Acquisition (slice timing) group')
        if y_labels is not None:
            ax.set_ylabel(y_labels[i])
        else:
            ax.set_ylabel(f'Parameter {index + 1}')
        ax.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)

        # Plot dashed lines for threshold values
        if rot_thresh is not None and i < 3:  # Check if rotation plot and threshold is specified
            ax.axhline(y=rot_thresh, color='r', linestyle='--', alpha=0.7,
                       label=f'Rotation Threshold ({rot_thresh})')
            ax.axhline(y=-rot_thresh, color='r', linestyle='--', alpha=0.7)
        elif trans_thresh is not None and i >= 3:  # Check if translation plot and threshold is specified
            ax.axhline(y=trans_thresh, color='r', linestyle='--', alpha=0.7,
                       label=f'Translation Threshold ({trans_thresh})')
            ax.axhline(y=-trans_thresh, color='r', linestyle='--', alpha=0.7)
    plt.tight_layout()

    base_name, _ = os.path.splitext(log_file_name)
    plot_filename = os.path.join(output_folder, f"{base_name}_parameters.png")
    counter = 1
    while os.path.exists(plot_filename):
        plot_filename = os.path.join(output_folder, f"{base_name}_parameters_{counter}.png")
        counter += 1
    plt.savefig(plot_filename)
    print(f"\n\nParameters plot saved to: {plot_filename}")
    plt.ion()
    plt.show(block=False)   # plt.show() is otherwise a blocking function pausing code execution until fig is closed

def plot_parameter_distributions(extracted_numbers):
    # Generate a frequency histogram of each transform parameter, overlaid onto the same figure
    # y-axis = frequency
    # x-axis = mm or degrees for translation and rotation, respectively
    # Indicate the lower and upper bounds of +/- 2 stddev (95% of data) and +/- 3 stddev (99.7%)
    # save PNG of parameter distribution

def plot_displacements(displacements, log_filename, threshold=None, total_volumes=None, volumes_above_threshold=None):
    # Create an output folder if it doesn't exist
    log_file_path, log_file_name = os.path.split(log_filename)
    output_folder = os.path.join(log_file_path, f"{os.path.splitext(log_file_name)[0]}_outputs")
    os.makedirs(output_folder, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(displacements, marker='o', linestyle='-', color='b', alpha=0.7, label='Displacements (mm)')
    plt.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
    if threshold is not None:
        plt.axhline(y=threshold, color='r', linestyle='--', linewidth=3, alpha=1.0, label=f'Threshold = {threshold} mm')

    plt.title('Displacement Tracking for : ' + log_file_name)
    plt.xlabel('Acquisition (slice timing) group')
    plt.ylabel('Displacement (mm)')
    plt.legend(loc='upper left')
    plt.tight_layout()

    # Display number of acquisitions and cumulative displacement
    cumulative_sum = sum(displacements)
    total_sets = len(displacements)
    text = f'Number of Acquisitions: {total_sets}\nCumulative Displacement (mm): {cumulative_sum:.3f}'
    if total_volumes is not None and volumes_above_threshold is not None:
        text += f'\nTotal Collected Volumes: {total_volumes:.3f}\nVolumes with Motion: {volumes_above_threshold:.3f}\nVolumes without Motion: {(total_volumes - volumes_above_threshold):.3f}'
    plt.text(0.5, 0.9, text, ha='center', va='center', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.7))

    base_name, _ = os.path.splitext(log_file_name)
    plot_filename = os.path.join(output_folder, f"{base_name}_displacements.png")
    counter = 1
    while os.path.exists(plot_filename):
        plot_filename = os.path.join(output_folder, f"{base_name}_displacements_{counter}.png")
        counter += 1
    plt.savefig(plot_filename)
    print(f"Displacements plot saved to: {plot_filename}")
    plt.ion()
    plt.show(block=True)


def construct_data_table(extracted_numbers, displacements, volume_ID):
    # print("Shapes of arrays:")
    # print("extracted_numbers:", extracted_numbers.shape)
    # print("displacements:", displacements.shape)
    # print("volume_ID:", volume_ID.shape)
    if len(extracted_numbers) != len(displacements) or len(displacements) != len(volume_ID) or len(extracted_numbers) != len(volume_ID):
        raise ValueError("Length of input arrays must be the same for concatenation.")

    # combine transform parameters, slice displacements, and associated volume number into one numpy array table
    data_table = np.hstack((extracted_numbers, displacements[..., np.newaxis], volume_ID[..., np.newaxis]))
    data_table_headers = ['X_rotation(rad)','Y_rotation(rad)','Z_rotation(rad)','X_translation(mm)','Y_translation(mm)','Z_translation(mm)','Slice_displacement(mm)', 'Volume_number']
    return data_table, data_table_headers


def export_values_csv(data_table, data_table_headers, log_filename):
    if len(data_table_headers) != data_table.shape[1]:
        raise ValueError("Number of headers must match number of columns in data table!")

    log_file_path, log_file_name = os.path.split(log_filename)
    output_folder = os.path.join(log_file_path, f"{os.path.splitext(log_file_name)[0]}_outputs")
    os.makedirs(output_folder, exist_ok=True)

    base_name, _ = os.path.splitext(log_file_name)
    csv_filename = os.path.join(output_folder, f"{base_name}_datatable.csv")

    header_string = ','.join(data_table_headers)
    np.savetxt(csv_filename, data_table, delimiter=",", header=header_string, comments='')
    print(f"Data table saved to: {csv_filename}")



if __name__ == "__main__":
    input_file = sys.argv[1]
    input_filename = "/data/" + input_file

    # IF input_file = .log file, THEN extract transform parameters from log file
    # IF input_file = .tfm or .txt file, THEN extract transform parameters from transform file(s)
    # return list of arrays of all transform parameters as "transform_list"

    # Calculate displacement between acquisitions
    displacements = compute_transform_pair_displacement(transform_list)

    # Calculate cumulative displacement
    cumulative_disp = sum(displacements)
    print("Cumulative sum of displacement:", cumulative_disp)

    # Calculate motion per minute estimate

    # Calculate tSNR?

    # Check displacements of each volume against motion threshold
    pixel_size = 2.4
    threshold_value = 0.75 # pixel_size*0.25  # threshold for acceptable motion (in mm)
    total_volumes, volumes_above_threshold, volume_id = check_volume_motion(displacements, sms_factor, num_vol_slices, threshold_value)

    # Plot transform parameters
    indices_to_plot = [0, 1, 2, 3, 4, 5] # remember base 0 indexing!
    titles = ['X-axis Rotation', 'Y-axis Rotation', 'Z-axis Rotation', 'X-axis Translation', 'Y-axis Translation', 'Z-axis Translation']
    y_labels = ['X Rotation (rad)', 'Y Rotation (rad)', 'Z Rotation (rad)', 'X Translation (mm)', 'Y Translation (mm)', 'Z Translation (mm)']
    rot_thresh = threshold_value / 50 # angle corresponding to arc length of threshold value, where radius = 50 mm
    plot_parameters(extracted_numbers, indices_to_plot, log_filename, titles, y_labels, rot_thresh, threshold_value)
    # plot_parameter_distributions(extracted_numbers)

    # Plot displacements
    plot_displacements(displacements, log_filename, threshold=threshold_value, total_volumes=total_volumes, volumes_above_threshold=volumes_above_threshold)

    # Export table of motion data (.csv file)
    data_table, data_table_headers = construct_data_table(np.array(extracted_numbers), np.array(displacements), np.array(volume_id))
    export_values_csv(data_table, data_table_headers, log_filename)