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
from matplotlib.gridspec import GridSpec
import SimpleITK as sitk
import numpy as np
from compute_displacement import compute_displacement
from extract_params_from_log import get_data_from_slimm_log
from extract_params_from_transform_files import get_data_from_transforms


def create_euler_transform(parameters, rotation_center=[0.0, 0.0, 0.0]):
    euler_transform = sitk.Euler3DTransform()
    euler_transform.SetParameters(parameters)
    euler_transform.SetCenter(rotation_center)
    return euler_transform

def compute_transform_pair_displacement(extracted_numbers, radius):
    num_instances = len(extracted_numbers)
    displacements = []
    for i in range(num_instances):
        if i == 0:
            parameters_previous = extracted_numbers[i]
        else:
            parameters_previous = extracted_numbers[i - 1]

        parameters_i = extracted_numbers[i]

        transform_previous = create_euler_transform(parameters_previous)
        transform_i = create_euler_transform(parameters_i)

        displacement_value = compute_displacement(transform_previous, transform_i, radius)
        displacements.append(displacement_value)

    print("\nNumber of displacement values:", len(displacements))
    return displacements

def compute_displacement(transform1, transform2, radius=50, outputfile=None):
    A0 = np.asarray(transform2.GetMatrix()).reshape(3, 3)
    c0 = np.asarray(transform2.GetCenter())
    t0 = np.asarray(transform2.GetTranslation())

    A1 = np.asarray(transform1.GetInverse().GetMatrix()).reshape(3, 3)
    c1 = np.asarray(transform1.GetInverse().GetCenter())
    t1 = np.asarray(transform1.GetInverse().GetTranslation())

    combined_mat = np.dot(A0,A1)
    combined_center = c1
    combined_translation = np.dot(A0, t1+c1-c0) + t0+c0-c1
    combined_affine = sitk.AffineTransform(combined_mat.flatten(), combined_translation, combined_center)

    # Save composed transform to outputfile
    if outputfile:
        sitk.WriteTransform(combined_affine, outputfile)

    versorrigid3d = sitk.VersorRigid3DTransform()
    versorrigid3d.SetCenter(combined_center)
    versorrigid3d.SetTranslation(combined_translation)
    versorrigid3d.SetMatrix(combined_mat.flatten())

    euler3d = sitk.Euler3DTransform()
    euler3d.SetCenter(combined_center)
    euler3d.SetTranslation(combined_translation)
    euler3d.SetMatrix(combined_mat.flatten())

    # Compute displacement (Tisdall et al. 2012)
    # print(f"\tHead radius (mm) : {radius}")
    params = np.asarray( euler3d.GetParameters() )
    # print("\tComposed parameters (Euler3D) : ", params)

    theta = np.abs(np.arccos(0.5 * (-1 + np.cos(params[0]) * np.cos(params[1]) + \
                                    np.cos(params[0]) * np.cos(params[2]) + \
                                    np.cos(params[1]) * np.cos(params[2]) + \
                                    np.sin(params[0]) * np.sin(params[1]) * np.sin(params[2]))))
    drot = radius * np.sqrt((1 - np.cos(theta)) ** 2 + np.sin(theta) ** 2)
    dtrans = np.linalg.norm(params[3:])
    displacement = drot + dtrans

    # print("\tDisplacement : ", displacement)

    return displacement

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

    print("Total collected volumes (+ reference):", total_volumes)
    print("Volumes with motion:", volumes_above_threshold)
    print("Volumes without motion:", (total_volumes - volumes_above_threshold))
    return total_volumes, volumes_above_threshold, volume_id


def create_output_file(input_filepath, new_filename_string="", file_extension=""):
    # Create an output folder, if it doesn't exist
    input_parentdir, input_filename = os.path.split(input_filepath)
    output_folder = os.path.join(input_parentdir, f"{os.path.splitext(input_filename)[0]}_outputs")
    os.makedirs(output_folder, exist_ok=True)

    # Create new output filename
    base_name, _ = os.path.splitext(input_filename)
    output_filename = f"{base_name}_{new_filename_string}.{file_extension}"
    output_filepath = os.path.join(output_folder, output_filename)
    counter = 1
    while os.path.exists(output_filepath):
        new_output_filename = f"{base_name}_{new_filename_string}_{counter}.{file_extension}"
        output_filepath = os.path.join(output_folder, new_output_filename)
        counter += 1

    return output_filepath


def plot_parameters(extracted_numbers, input_filepath="", titles=None, y_labels=None, trans_thresh=0.75, radius=50):
    indices_to_plot = [0,1,2,3,4,5]     # which parameters to plot (column indices)
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
        rot_thresh = trans_thresh / radius # assume head radius = 50 mm
        if rot_thresh is not None and i < 3:  # Check if rotation plot and threshold is specified
            ax.axhline(y=rot_thresh, color='r', linestyle='--', alpha=0.7,
                       label=f'Rotation Threshold ({rot_thresh})')
            ax.axhline(y=-rot_thresh, color='r', linestyle='--', alpha=0.7)
        elif trans_thresh is not None and i >= 3:  # Check if translation plot and threshold is specified
            ax.axhline(y=trans_thresh, color='r', linestyle='--', alpha=0.7,
                       label=f'Translation Threshold ({trans_thresh})')
            ax.axhline(y=-trans_thresh, color='r', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_filename = create_output_file(input_filepath, "parameters","png")
    plt.savefig(plot_filename)
    print(f"\nParameters plot saved as: {plot_filename}")
    plt.ion()
    plt.show(block=False)   # plt.show() is otherwise a blocking function pausing code execution until fig is closed

def plot_parameter_distributions(transform_list, input_filepath="", offset=0.008):
    x_rotation = []
    y_rotation = []
    z_rotation = []
    x_translation = []
    y_translation = []
    z_translation = []

    for transform in transform_list:
        x_rotation.append(np.degrees(transform[0]))
        y_rotation.append(np.degrees(transform[1]))
        z_rotation.append(np.degrees(transform[2]))
        x_translation.append(transform[3])
        y_translation.append(transform[4])
        z_translation.append(transform[5])

    plt.figure(figsize=(10, 6))
    params = ['x_rotation', 'y_rotation', 'z_rotation', 'x_translation', 'y_translation', 'z_translation']
    data = [x_rotation, y_rotation, z_rotation, x_translation, y_translation, z_translation]
    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    for i, (param, values, color) in enumerate(zip(params, data, colors)):
        hist_values, bins = np.histogram(values, bins=100)
        hist_values = hist_values / len(values)  # Normalize by total count
        bins_center = (bins[:-1] + bins[1:]) / 2
        plt.plot(bins_center, hist_values, label=param, color=color, alpha=1.0)

        # Calculate the 99% coverage range and plot it
        lower_bound = np.percentile(values, 0.5)
        upper_bound = np.percentile(values, 99.5)
        plt.hlines(-0.005 - i * offset, lower_bound, upper_bound, colors=color, linewidth=6)

    plt.legend()
    plt.xlabel('mm/degrees')
    plt.ylabel('Normalized frequency')
    plt.title('Distribution of motion parameters : ' + input_filepath)

    plot_filename = create_output_file(input_filepath, "parameters_distribution","png")
    plt.savefig(plot_filename)
    print(f"Parameters distribution plot saved as: {plot_filename}")

    plt.ion()
    plt.show(block=False)

def plot_displacements(displacements, input_filepath, threshold=None, total_volumes=None, volumes_above_threshold=None):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 1]})

    # Left subplot: displacement values per acquisition group
    axs[0].plot(displacements, marker='o', linestyle='-', color='b', alpha=0.7, label='Displacements (mm)')
    axs[0].grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
    if threshold is not None:
        axs[0].axhline(y=threshold, color='r', linestyle='--', linewidth=3, alpha=1.0,
                       label=f'Threshold = {threshold} mm')

    axs[0].set_title('Displacement Tracking : ' + input_filepath)
    axs[0].set_xlabel('Acquisition (slice timing) group')
    axs[0].set_ylabel('Displacement (mm)')
    axs[0].legend(loc='upper left')

    # Display number of acquisitions and cumulative displacement on the left plot
    cumulative_sum = sum(displacements)
    total_sets = len(displacements)
    text = f'Number of Acquisitions: {total_sets}\nCumulative Displacement (mm): {cumulative_sum:.3f}'
    if total_volumes is not None and volumes_above_threshold is not None:
        text += f'\nTotal Collected Volumes: {total_volumes:.3f}\nVolumes with Motion: {volumes_above_threshold:.3f}\nVolumes without Motion: {(total_volumes - volumes_above_threshold):.3f}'
    axs[0].text(0.5, 0.9, text, ha='center', va='center', transform=axs[0].transAxes,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.7))

    # Right subplot: boxplot of all displacement values
    axs[1].boxplot(displacements, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='skyblue', color='black', alpha=1.0),
                   whiskerprops=dict(color='black'), capprops=dict(color='black'), medianprops=dict(color='black'))

    axs[1].set_title('Displacement distribution')
    axs[1].set_ylabel('Displacement (mm)')
    axs[1].set_xticks([1])
    axs[1].set_xticklabels([''])

    # Adjust the width of the right subplot
    plt.subplots_adjust(wspace=0.4)  # Increase or decrease the value to adjust the space between subplots

    plt.tight_layout()

    plot_filename = create_output_file(input_filepath, "displacements", "png")
    plt.savefig(plot_filename)
    print(f"Displacements plot saved as: {plot_filename}")
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


def export_values_csv(data_table, data_table_headers, input_filepath):
    if len(data_table_headers) != data_table.shape[1]:
        raise ValueError("Number of headers must match number of columns in data table!")

    csv_filename = create_output_file(input_filepath, "datatable", "csv")

    header_string = ','.join(data_table_headers)
    np.savetxt(csv_filename, data_table, delimiter=",", header=header_string, comments='')
    print(f"Data table saved as: {csv_filename}")



if __name__ == "__main__":
    input_filename = sys.argv[1]
    input_filepath = "/data/" + input_filename

    if os.path.isfile(input_filepath):
        if input_filepath.endswith(".log"):
            print("Processing log file...")
            transform_list, sms_factor, nslices_per_vol = get_data_from_slimm_log(input_filepath)
        elif input_filepath.endswith(".txt") or input_filepath.endswith(".tfm"):
            print("Processing directory of transform files...")
            directory_path = os.path.dirname(input_filepath)
            transform_list = get_data_from_transforms(directory_path)
        else:
            raise ValueError("Unsupported file extension. Please provide a .log, .txt, or .tfm file.")
    else:
        raise ValueError("The input path is not a valid file.")


    # USER-SPECIFIED VALUES
    radius = 50     # head radius assumption (mm)
    pixel_size = 2.4  # in mm
    threshold_value = 0.75  # pixel_size*0.25  # threshold for acceptable motion (mm)
    print(f"\nHead radius (mm) : {radius}")
    print(f"\nMotion threshold (mm) : {threshold_value}")

    # Calculate displacement between acquisitions
    displacements = compute_transform_pair_displacement(transform_list, radius)

    # Calculate cumulative displacement
    cumulative_disp = sum(displacements)
    print(f"Cumulative sum of displacement: {cumulative_disp} mm")

    # Calculate motion per minute estimate

    # Calculate tSNR?

    # Check displacements of each volume against motion threshold
    total_volumes, volumes_above_threshold, volume_id = check_volume_motion(displacements, sms_factor, nslices_per_vol, threshold_value)

    # Plot transform parameters
    titles = ['X-axis Rotation', 'Y-axis Rotation', 'Z-axis Rotation', 'X-axis Translation', 'Y-axis Translation', 'Z-axis Translation']
    y_labels = ['X Rotation (rad)', 'Y Rotation (rad)', 'Z Rotation (rad)', 'X Translation (mm)', 'Y Translation (mm)', 'Z Translation (mm)']
    plot_parameters(transform_list, input_filepath, titles, y_labels, threshold_value, radius)
    plot_parameter_distributions(transform_list, input_filepath)

    # Plot displacements
    plot_displacements(displacements, input_filepath, threshold=threshold_value, total_volumes=total_volumes, volumes_above_threshold=volumes_above_threshold)

    # Export table of motion data (.csv file)
    data_table, data_table_headers = construct_data_table(np.array(transform_list), np.array(displacements), np.array(volume_id))
    export_values_csv(data_table, data_table_headers, input_filepath)