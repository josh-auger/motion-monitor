#!/usr/bin/env python3

# Title: compute_motion_measures.py

# Description:
# For an array list of transform parameters, various measures that characterize motion are computed and reported.
# See USER-SPECIFIED VALUES in __main__ function to alter motion threshold or head radius assumption

# Created on: June 2024
# Created by: Joshua Auger (joshua.auger@childrens.harvard.edu), Computational Radiology Lab, Boston Children's Hospital

import re
import os
import sys
import argparse
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import logging
import datetime
import pytz
from compute_displacement import compute_displacement
from extract_params_from_log import get_data_from_slimm_log
from extract_params_from_transform_files import get_data_from_transforms

def setup_logging(input_filepath, start_time):
    log_filename = create_output_file(input_filepath, f"motion_monitor", "log", start_time)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Logging to : {log_filename}")
    return log_filename

def get_rotation_center(transform_list):
    rotation_center = [0.0, 0.0, 0.0]  # Default rotation center

    if len(transform_list[0]) > 6:
        potential_rotation_center = transform_list[0][-3:]
        if all(np.array_equal(potential_rotation_center, row[-3:]) for row in transform_list):
            rotation_center = potential_rotation_center
            transform_list = [row[:-3] for row in transform_list]  # Remove rotation center columns

    logging.info(f"Center of rotation : {rotation_center}")
    return rotation_center, transform_list

def create_euler_transform(parameters, rotation_center):
    euler_transform = sitk.Euler3DTransform()
    euler_transform.SetParameters(parameters)
    euler_transform.SetCenter(rotation_center)
    return euler_transform

def compute_transform_pair_displacement(transform_list, rotation_center, radius):
    num_instances = len(transform_list)
    displacements = []

    for i in range(num_instances):
        if i == 0:
            parameters_previous = transform_list[i]
        else:
            parameters_previous = transform_list[i - 1]

        parameters_i = transform_list[i]

        transform_previous = create_euler_transform(parameters_previous, rotation_center)
        transform_i = create_euler_transform(parameters_i, rotation_center)

        displacement_value = compute_displacement(transform_previous, transform_i, radius)
        displacements.append(displacement_value)

    logging.info(f"Number of displacement values : {len(displacements)}")
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
    params = np.asarray( euler3d.GetParameters() )
    # logging.info(f"\tComposed parameters (Euler3D) : {params}")
    theta = np.abs(np.arccos(0.5 * (-1 + np.cos(params[0]) * np.cos(params[1]) + \
                                    np.cos(params[0]) * np.cos(params[2]) + \
                                    np.cos(params[1]) * np.cos(params[2]) + \
                                    np.sin(params[0]) * np.sin(params[1]) * np.sin(params[2]))))
    drot = radius * np.sqrt((1 - np.cos(theta)) ** 2 + np.sin(theta) ** 2)
    dtrans = np.linalg.norm(params[3:])
    displacement = drot + dtrans

    # logging.info(f"\tDisplacement : {displacement}")
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
    motion_flag = []

    for i in range(0, len(displacements), num_acquisitions_per_volume):
        volume_count = i // num_acquisitions_per_volume + 1
        volume_displacements = displacements[i:i + num_acquisitions_per_volume]
        volume_id.extend([volume_count] * len(volume_displacements))

        current_motion_flag = [1 if d > threshold else 0 for d in volume_displacements]
        motion_flag.extend(current_motion_flag)
        if any(current_motion_flag):
            volumes_above_threshold += 1

    logging.info("")
    logging.info(f"Number of displacements above threshold : {sum(motion_flag)}")
    logging.info(f"Total collected volumes (+ reference) : {total_volumes}")
    logging.info(f"Volumes with motion : {volumes_above_threshold}")
    logging.info(f"Volumes without motion : {(total_volumes - volumes_above_threshold)}")
    return total_volumes, volumes_above_threshold, volume_id, motion_flag


def calculate_motion_per_minute(displacements, acquisition_time):
    # logging.info(f"Acquisition time (sec) : {acquisition_time}")
    cumulative_disp = sum(displacements)
    total_sets = len(displacements)

    motion_per_minute = (cumulative_disp / total_sets) * (60 / acquisition_time)

    logging.info(f"Average motion per minute (mm/min) : {motion_per_minute}")
    return motion_per_minute


def create_output_file(input_filepath, new_filename_string="", file_extension="", start_time=None):
    # Create an output folder, if it doesn't exist
    input_parentdir, input_filename = os.path.split(input_filepath)
    output_folder = os.path.join(input_parentdir, f"{os.path.splitext(input_filename)[0]}_outputs")
    os.makedirs(output_folder, exist_ok=True)

    # Use start_time if provided, else use the current time
    if start_time is None:
        start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create new output filename
    base_name, _ = os.path.splitext(input_filename)
    output_filename = f"{base_name}_{new_filename_string}_{start_time}.{file_extension}"
    output_filepath = os.path.join(output_folder, output_filename)

    return output_filepath


def plot_parameters(extracted_numbers, input_filepath="", output_filename="", titles=None, y_labels=None, trans_thresh=0.75, radius=50):
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

    plt.savefig(output_filename)
    logging.info("")
    logging.info(f"Parameters plot saved as : {output_filename}")
    plt.ion()
    plt.show(block=False)   # plt.show() is otherwise a blocking function pausing code execution until fig is closed

def plot_parameter_distributions(transform_list, input_filepath="", output_filename="", offset_percent=0.03):
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

    max_hist_value = 0
    histograms = []
    n_bins = int(np.sqrt(len(transform_list)))      # square root heuristic choice for nbins

    # Calculate normalized frequency counts and find max histogram value
    for values in data:
        hist_values, bins = np.histogram(values, bins=n_bins)
        hist_values = hist_values / len(values)  # Normalize by total count
        histograms.append((hist_values, bins))
        max_hist_value = max(max_hist_value, max(hist_values))

    offset = offset_percent * max_hist_value    # define bar offsets based on max histogram value

    for i, (param, (hist_values, bins), color) in enumerate(zip(params, histograms, colors)):
        bins_center = (bins[:-1] + bins[1:]) / 2
        plt.plot(bins_center, hist_values, label=param, color=color, alpha=1.0)
        # plt.bar(bins_center, hist_values, width=(bins[1] - bins[0]), alpha=0.5, label=param, color=color)

        # Calculate 99% coverage range to display
        values = data[i]
        lower_bound = np.percentile(values, 0.5)
        upper_bound = np.percentile(values, 99.5)
        plt.hlines(-(0.6 * offset) - (i * offset), lower_bound, upper_bound, colors=color, linewidth=6)

    y_ticks = plt.gca().get_yticks()
    y_ticks = y_ticks[y_ticks >= 0]
    plt.gca().set_yticks(y_ticks)

    plt.legend()
    plt.xlabel('mm/degrees')
    plt.ylabel('Normalized frequency')
    plt.title('Distribution of motion parameters : ' + input_filepath)

    plt.savefig(output_filename)
    logging.info(f"Parameters distribution plot saved as : {output_filename}")

    plt.ion()
    plt.show(block=False)

def plot_displacements(displacements, input_filepath="", output_filename="", threshold=None, total_volumes=None, volumes_above_threshold=None):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3.5, 1]})

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

    plt.savefig(output_filename)
    logging.info(f"Displacements plot saved as : {output_filename}")
    plt.ion()
    plt.show(block=True)


def construct_data_table(transform_list, displacements, volume_ID, motion_flag):
    if len(transform_list) != len(displacements) or len(displacements) != len(volume_ID) or len(transform_list) != len(volume_ID):
        raise ValueError("Length of input arrays must be the same for concatenation.")

    data_table = np.hstack((transform_list, displacements[..., np.newaxis], volume_ID[..., np.newaxis], motion_flag[..., np.newaxis]))
    data_table_headers = ['X_rotation(rad)','Y_rotation(rad)','Z_rotation(rad)','X_translation(mm)','Y_translation(mm)','Z_translation(mm)','Displacement(mm)', 'Volume_number', 'Motion_flag']
    return data_table, data_table_headers


def export_values_csv(data_table, data_table_headers, output_filename):
    if len(data_table_headers) != data_table.shape[1]:
        raise ValueError("Number of headers must match number of columns in data table!")

    header_string = ','.join(data_table_headers)
    np.savetxt(output_filename, data_table, delimiter=",", header=header_string, comments='')
    logging.info(f"Data table saved as : {output_filename}")


if __name__ == "__main__":
    input_filename = sys.argv[1]
    # Check if input_filename is just a filename and not a file path
    if os.path.basename(input_filename) != input_filename:
        raise ValueError("Scurvy dogs, we messed up! \n"
        "The input argument should be a single filename (with .log, .txt, or .tfm extension) and NOT a file path. \n"
        "Fix that blunder and give it another go, matey!")
    input_filepath = "/data/" + input_filename

    start_time = datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y%m%d_%H%M%S')  # Default to Eastern timezone!
    setup_logging(input_filepath, start_time)
    logging.info("Ahoy! Welcome to the motion-monitor, let's start monitoring some motion!")
    logging.info(f"Input filepath : {input_filepath}")
    logging.info("")

    # Determine input type for pre-processing
    if os.path.isfile(input_filepath):
        if input_filepath.endswith(".log"):
            transform_list, sms_factor, nslices_per_vol = get_data_from_slimm_log(input_filepath)
        elif input_filepath.endswith(".txt") or input_filepath.endswith(".tfm"):
            directory_path = os.path.dirname(input_filepath)
            transform_list = get_data_from_transforms(directory_path)
            sms_factor = 1      # assume these val=1 for kooshball sequence for now...
            nslices_per_vol = 1
            logging.info(f"\tNo scan metadata found. Defaulting to:")
            logging.info(f"\tSMS factor = {sms_factor}")
            logging.info(f"\tNum slices per volume: {nslices_per_vol}")
            logging.info(f"\tNum acquisitions per volume: {int(nslices_per_vol / sms_factor)}")
        else:
            raise ValueError("Arrr! Unsupported file extension. Please provide a .log, .txt, or .tfm file.")
    else:
        raise ValueError("Arrr! The input path is not a valid file.")

    logging.info("")
    logging.info("Calculating motion measures from transform parameters...")

    # ---------- USER-SPECIFIED VALUES ----------
    radius = 50     # spherical head radius assumption (mm)
    threshold_value = 0.75  # threshold for acceptable motion (mm)
    acquisition_time = 4.2  # time between acquisitions/registration instances (sec)
    logging.info("")
    logging.info(f"User-specified values:")
    logging.info(f"\tHead radius (mm) : {radius}")
    logging.info(f"\tMotion threshold (mm) : {threshold_value}")
    logging.info(f"\tAcquisition time (sec): {acquisition_time}")
    logging.info("")

    # Displacement between acquisitions
    rotation_center, transform_list = get_rotation_center(transform_list)
    displacements = compute_transform_pair_displacement(transform_list, rotation_center, radius)

    # Cumulative displacement
    cumulative_disp = sum(displacements)
    logging.info(f"Cumulative sum of displacement (mm) : {cumulative_disp}")

    # Average motion per minute estimate
    motion_per_min = calculate_motion_per_minute(displacements, acquisition_time)

    # Check displacements of each volume against motion threshold
    total_volumes, volumes_above_threshold, volume_id, motion_flag = check_volume_motion(displacements, sms_factor, nslices_per_vol, threshold_value)

    # Plot transform parameters
    titles = ['X-axis Rotation', 'Y-axis Rotation', 'Z-axis Rotation', 'X-axis Translation', 'Y-axis Translation', 'Z-axis Translation']
    y_labels = ['X Rotation (rad)', 'Y Rotation (rad)', 'Z Rotation (rad)', 'X Translation (mm)', 'Y Translation (mm)', 'Z Translation (mm)']
    params_filename = create_output_file(input_filepath, "parameters", "png", start_time)
    plot_parameters(transform_list, input_filepath, params_filename, titles, y_labels, threshold_value, radius)
    params_dist_filename = create_output_file(input_filepath, "parameters_distribution", "png", start_time)
    plot_parameter_distributions(transform_list, input_filepath, params_dist_filename)

    # Plot displacements
    disp_filename = create_output_file(input_filepath, "displacements", "png", start_time)
    plot_displacements(displacements, input_filepath, disp_filename, threshold=threshold_value, total_volumes=total_volumes, volumes_above_threshold=volumes_above_threshold)

    # Export table of motion data (.csv file)
    data_table, data_table_headers = construct_data_table(np.array(transform_list), np.array(displacements), np.array(volume_id), np.array(motion_flag))
    csv_filename = create_output_file(input_filepath, "datatable", "csv", start_time)
    export_values_csv(data_table, data_table_headers, csv_filename)

    logging.info("")
    logging.info("...motion has been monitored. Fair winds and fortune on the motion of the ocean to ye!")