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
from matplotlib.ticker import MultipleLocator

from compute_displacement import compute_displacement
from extract_params_from_log import get_data_from_slimm_log
from extract_params_from_transform_files import get_data_from_transforms
from extract_params_from_transform_files import look_for_metadata_file

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

    logging.info("")
    logging.info(f"\tCenter of rotation : {rotation_center}")
    logging.info("")
    return rotation_center, transform_list

def create_euler_transform(parameters, rotation_center):
    """
    Convert Versor rigid 3D transform parameters into a Euler3DTransform. The motion-monitor assumes that the
    extracted transform parameters are for a Versor Rigid 3D transform, to be converted to Euler format.
    """
    if len(parameters) != 6:
        raise ValueError("Parameters must be a list or tuple of 6 elements.")

    # Extract versor (rotation) and translation components
    versor = parameters[:3]
    translation = parameters[3:]

    # Create a VersorTransform to interpret the versor
    versor_transform = sitk.VersorRigid3DTransform()
    versor_transform.SetParameters(parameters)
    versor_transform.SetCenter(rotation_center)

    # Extract Euler angles (in radians) from the VersorTransform
    euler_angles = versor_transform.GetMatrix()
    euler_angles = np.array(euler_angles).reshape(3, 3)  # Convert to 3x3 matrix

    # Convert rotation matrix to Euler angles (ZYX convention)
    sy = np.sqrt(euler_angles[0, 0] ** 2 + euler_angles[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(euler_angles[2, 1], euler_angles[2, 2])
        y = np.arctan2(-euler_angles[2, 0], sy)
        z = np.arctan2(euler_angles[1, 0], euler_angles[0, 0])
    else:
        x = np.arctan2(-euler_angles[1, 2], euler_angles[1, 1])
        y = np.arctan2(-euler_angles[2, 0], sy)
        z = 0

    # Create the Euler3DTransform
    euler_transform = sitk.Euler3DTransform()
    euler_transform.SetRotation(x, y, z)  # Angles are in radians
    euler_transform.SetTranslation(translation)
    euler_transform.SetCenter(rotation_center)
    return euler_transform

def compute_transform_pair_displacement(transform_list, rotation_center, radius):
    num_instances = len(transform_list)
    displacements = []
    euler_transform_list = []
    zero_parameters = [0] * len(transform_list[0])

    for i in range(num_instances):
        if i == 0:
            parameters_previous = zero_parameters
        else:
            parameters_previous = transform_list[i - 1]

        parameters_i = transform_list[i]
        logging.info(f"\tPrior transform parameters : {parameters_previous}")
        logging.info(f"\tCurrent transform parameters : {parameters_i}")

        # ASSUMPTION: the transform parameters are for a Versor Rigid 3D transform where the first 3 elements are the
        # components of the versor representation of 3D rotation and the last 3 parameters defines the translation in
        # each dimension.

        # Convert Versor rigid 3D parameters into Euler transform objects for computing displacement
        transform_previous = create_euler_transform(parameters_previous, rotation_center)
        transform_i = create_euler_transform(parameters_i, rotation_center)

        # Compute displacement from Euler transforms
        displacement_value = compute_displacement(transform_previous, transform_i, radius)
        displacements.append(displacement_value)
        euler_transform_list.append(transform_i.GetParameters())

    logging.info("")
    logging.info(f"Number of displacement values : {len(displacements)}")
    return displacements, euler_transform_list

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
    logging.info(f"\tComposed parameters (Euler3D) : {params}")
    theta = np.abs(np.arccos(0.5 * (-1 + np.cos(params[0]) * np.cos(params[1]) + \
                                    np.cos(params[0]) * np.cos(params[2]) + \
                                    np.cos(params[1]) * np.cos(params[2]) + \
                                    np.sin(params[0]) * np.sin(params[1]) * np.sin(params[2]))))
    drot = radius * np.sqrt((1 - np.cos(theta)) ** 2 + np.sin(theta) ** 2)
    dtrans = np.linalg.norm(params[3:])
    displacement = drot + dtrans

    logging.info(f"\tDisplacement : {displacement}")
    logging.info("")
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
    # num_acquisitions_per_volume = 1     # Workaround for VVR in SLIMM logs where 1 registration = 1 volume
    total_volumes = (len(displacements) / num_acquisitions_per_volume)

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
    logging.info(f"Acceptable motion threshold (mm) :  {threshold}")
    logging.info(f"Number of displacement values above threshold : {sum(motion_flag)}")
    logging.info("")
    logging.info(f"Total registered volumes : {total_volumes}")
    logging.info(f"Volumes with motion : {volumes_above_threshold}")
    logging.info(f"Volumes without motion : {(total_volumes - volumes_above_threshold)}")
    return total_volumes, volumes_above_threshold, volume_id, motion_flag


def calculate_motion_per_minute(displacements):
    cumulative_disp = sum(displacements)
    total_sets = len(displacements)

    motion_per_acquisition = (cumulative_disp / total_sets)

    logging.info(f"Average motion per acquisition : {motion_per_acquisition}")
    logging.info(f"\tIf acquisition time (sec) is known, then motion per minute = (motion per acquisition) * (60 / acquisition time)")
    return motion_per_acquisition

def calculate_cumulative_displacement(displacements):
    cumulative_displacement = np.cumsum(displacements)

    return cumulative_displacement


def create_output_file(input_filepath, new_filename_string="", file_extension="", start_time=None):
    # Create an output folder, if it doesn't exist
    input_parentdir, input_filename = os.path.split(input_filepath)
    output_folder = os.path.join(input_parentdir, f"{os.path.splitext(input_filename)[0]}_motionmonitor_outputs")
    os.makedirs(output_folder, exist_ok=True)

    # Use start_time if provided, else use the current time
    if start_time is None:
        start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create new output filename
    base_name, _ = os.path.splitext(input_filename)
    output_filename = f"{base_name}_{new_filename_string}_{start_time}.{file_extension}"
    output_filepath = os.path.join(output_folder, output_filename)

    return output_filepath


def plot_parameters(extracted_numbers, series_name="", output_filename="", titles=None, y_labels=None, trans_thresh=0.75, radius=50):
    """
    IMPORTANT NOTE!
    The motion-monitor assumes the three rotation parameters from the transform files/logs are in
    radians and converts the rotations to degrees when reporting values. The transform parameters are otherwise
    reported exactly as they appear in the provided parameter list.
        IF the transforms are of type Versor Rigid 3D, then the rotation parameters are the components of a 3D rotation
        versor (vector part of a unit quaternion) in radians.
        IF the transforms are of type Euler 3D, then the rotation parameters are rigid 3D rotations in radians.
    """

    indices_to_plot = [0,1,2,3,4,5]     # which parameters to plot (column indices)
    fig, axes = plt.subplots(3, 2, figsize=(18,12))
    subplot_colors = ['b', 'g', 'r', 'c', 'm', 'y']  # Default colors for subsequent plots
    fig.suptitle(f"motion parameters : {series_name}", y=0.98)

    for i, index in enumerate(indices_to_plot):
        row = i % 3  # Three rows
        col = 0 if i < 3 else 1  # Rotations in left column, Translations in right column
        ax = axes[row, col]

        numbers_to_plot = [numbers[index] for numbers in extracted_numbers]
        if i < 3:
            numbers_to_plot = np.degrees(numbers_to_plot)

        color = subplot_colors[i % len(subplot_colors)]  # Cycle through colors
        ax.plot(numbers_to_plot, marker='o', linestyle='-', color=color, alpha=0.5)

        if titles is not None:
            ax.set_title(titles[i])
        else:
            ax.set_title(f'{index + 1}-th parameter of each registration instance')

        ax.set_xlabel('Acquisition (slice timing) group')
        ax.set_ylabel(y_labels[i] if y_labels else f'Parameter {index + 1}')
        ax.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
        ax.set_xlim(left=0) # Ensure x-axis starts at 0

        # Plot dashed lines for threshold values
        rot_thresh = np.degrees(trans_thresh / radius)  # Assume head radius = 50 mm
        if i < 3:  # Rotation parameters
            ax.axhline(y=rot_thresh, color='r', linestyle='--', alpha=0.7, label=f'Rotation Threshold ({rot_thresh})')
            ax.axhline(y=-rot_thresh, color='r', linestyle='--', alpha=0.7)
        else:  # Translation parameters
            ax.axhline(y=trans_thresh, color='r', linestyle='--', alpha=0.7, label=f'Translation Threshold ({trans_thresh})')
            ax.axhline(y=-trans_thresh, color='r', linestyle='--', alpha=0.7)

        ax.relim()
        ax.autoscale_view()

    plt.tight_layout()

    plt.savefig(output_filename)
    logging.info("")
    logging.info(f"Parameters plot saved as : {output_filename}")
    plt.ion()
    plt.show(block=False)   # plt.show() is otherwise a blocking function pausing code execution until fig is closed

def plot_parameter_distributions(transform_list, series_name="", output_filename="", offset_percent=0.03):
    x_rotation = []
    y_rotation = []
    z_rotation = []
    x_translation = []
    y_translation = []
    z_translation = []

    # Rotation parameters are converted from radians to degrees
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
    plt.title('Distribution of motion parameters : ' + series_name)
    plt.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)

    plt.savefig(output_filename)
    logging.info(f"Parameters distribution plot saved as : {output_filename}")

    plt.ion()
    plt.show(block=False)

def plot_displacements(displacements, series_name="", output_filename="", threshold=None, total_volumes=None, volumes_above_threshold=None):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3.5, 1]})

    # Left subplot: displacement values per acquisition group
    axs[0].plot(displacements, marker='o', linestyle='-', color='b', alpha=0.7, label='Displacements (mm)')
    axs[0].grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
    if threshold is not None:
        axs[0].axhline(y=threshold, color='r', linestyle='--', linewidth=3, alpha=1.0,
                       label=f'Threshold = {threshold} mm')

    axs[0].set_title('Displacement Tracking : ' + series_name)
    axs[0].set_xlabel('Acquisition Instance')
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

def plot_cumulative_displacement(cumulative_displacements, series_name="", output_filename="", threshold=None, total_volumes=None, volumes_above_threshold=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot cumulative displacement values per acquisition group
    ax.plot(cumulative_displacements, marker='o', linestyle='-', color='b', alpha=0.7, label='Cumulative Displacements (mm)')
    ax.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)

    if threshold is not None:
        x = np.arange(len(cumulative_displacements))
        y = threshold * x
        ax.plot(x, y, color='r', linestyle='--', linewidth=1, alpha=1.0,
                label=f'Acceptable Accumulation (slope = {threshold})')

    ax.set_title('Cumulative Displacement Tracking : ' + series_name)
    ax.set_xlabel('Acquisition Instance')
    ax.set_ylabel('Cumulative Displacement (mm)')
    ax.legend(loc='upper left')

    # Display number of acquisitions and cumulative displacement on the plot
    total_sets = len(cumulative_displacements)
    final_cumulative_displacement = cumulative_displacements[-1]
    text = f'Number of Acquisitions: {total_sets}\nFinal Cumulative Displacement (mm): {final_cumulative_displacement:.3f}'
    if total_volumes is not None and volumes_above_threshold is not None:
        text += f'\nTotal Collected Volumes: {total_volumes:.3f}\nVolumes with Motion: {volumes_above_threshold:.3f}\nVolumes without Motion: {(total_volumes - volumes_above_threshold):.3f}'
    ax.text(0.5, 0.9, text, ha='center', va='center', transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.7))

    plt.tight_layout()

    plt.savefig(output_filename)
    logging.info(f"Cumulative Displacements plot saved as : {output_filename}")
    plt.ion()
    plt.show(block=True)

def plot_motion_flags(volume_numbers, motion_flags, series_name, motion_flags_filename, sms_factor, nslices_per_vol, threshold_value):
    """
    Plot the motion flags with an overlaid bar chart to group each volume.
        - Scatter plot shows each individual motion flag event
        - Bar chart denotes all slice groups within each volume, and if any slice group triggered a motion flag.
    """
    # Dynamically set the figure size
    unique_volumes = np.unique(volume_numbers)
    figure_width = min(25, len(unique_volumes) * 0.08)
    plt.figure(figsize=(figure_width, 4))

    # Scatter plot of all motion flags
    plt.scatter(range(len(motion_flags)), motion_flags, label=f"Motion flags ({len(motion_flags)})", color='blue', alpha=0.5, s=10)

    # Condense motion flags to per-volume level
    slice_group_size = nslices_per_vol / sms_factor
    logging.info(f"unique volumes = {len(unique_volumes)}")
    logging.info(f"slice group size = {slice_group_size}")

    volume_motion_flag = np.zeros(len(unique_volumes))
    for i, vol in enumerate(unique_volumes):
        indices = np.where(volume_numbers == vol)[0]
        volume_motion_flag[i] = 1 if np.any(motion_flags[indices] == 1) else 0.05  # Flag motion or set baseline

    # Bar chart of per-volume motion flags
    bar_positions = [idx * slice_group_size for idx in range(len(unique_volumes))]  # Position bars by slice groups
    plt.bar(bar_positions, volume_motion_flag, width=slice_group_size, color='black', alpha=0.3,
            edgecolor='black', linewidth=1, align='edge', label=f"Volumes ({len(unique_volumes)})")

    # Report number of volumes flagged with motion in sub-title
    flagged_volumes = np.sum(volume_motion_flag == 1)
    subtitle = f"Total registered volumes: {len(unique_volumes)}, Volumes with motion: {flagged_volumes}"
    plt.suptitle(subtitle, fontsize=10, x=0.5, y=0.83)  # Add subtitle with specific position

    # Configure axis ticks and labels
    plt.ylabel(f"Motion Flag \n(threshold = {threshold_value} mm)")
    plt.yticks([0.05, 1], ["No Motion", "Motion"])  # Custom labels for motion flags
    plt.xlabel("Volume Number")
    plt.xlim(min(bar_positions) - slice_group_size, max(bar_positions) + slice_group_size)
    plt.gca().tick_params(axis='x', which='minor', pad=0, length=3)
    minor_tick_positions = [pos + slice_group_size / 2 for pos in bar_positions]  # Center of each bar
    plt.gca().xaxis.set_minor_locator(plt.FixedLocator(minor_tick_positions))  # Minor ticks centered at every volume bar
    plt.gca().tick_params(axis='x', which='major', pad=0, length=7)
    major_tick_positions = [pos + slice_group_size / 2 for pos in bar_positions]  # Center of each bar
    plt.gca().xaxis.set_major_locator(plt.FixedLocator(major_tick_positions[::4]))  # Major ticks every 4th volume
    major_tick_labels = [str(int(vol)) for vol in unique_volumes]  # Corresponding volume numbers
    plt.xticks(major_tick_positions[::4], major_tick_labels[::4], rotation=50, fontsize=10)  # Show every 4th label

    # Final plot settings
    plt.title(f'Volume Motion Flags: {series_name}', fontsize=12, x=0.5, y=1.1)
    plt.legend(loc="upper left", framealpha=0.7)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, color='gray', alpha=0.4)
    plt.tight_layout()

    plt.tight_layout()
    plt.savefig(motion_flags_filename)
    logging.info(f"Motion flags plot saved as: {motion_flags_filename}")
    plt.ion()
    plt.show(block=True)


def construct_data_table(transform_list, displacements, cumulative_displacements, volume_ID, motion_flag):
    if len(transform_list) != len(displacements) or len(displacements) != len(volume_ID) or len(transform_list) != len(volume_ID) or len(displacements) != len(cumulative_displacements):
        raise ValueError("Length of input arrays must be the same for concatenation.")

    data_table = np.hstack((transform_list, displacements[..., np.newaxis], cumulative_displacements[..., np.newaxis], volume_ID[..., np.newaxis], motion_flag[..., np.newaxis]))
    data_table_headers = ['X_rotation(rad)','Y_rotation(rad)','Z_rotation(rad)','X_translation(mm)','Y_translation(mm)','Z_translation(mm)','Displacement(mm)', 'Cumulative_displacement(mm)', 'Volume_number', 'Motion_flag']
    return data_table, data_table_headers


def export_values_csv(data_table, data_table_headers, output_filename):
    if len(data_table_headers) != data_table.shape[1]:
        raise ValueError("Number of headers must match number of columns in data table!")

    header_string = ','.join(data_table_headers)
    np.savetxt(output_filename, data_table, delimiter=",", header=header_string, comments='')
    logging.info(f"Data table saved as : {output_filename}")


if __name__ == "__main__":
    input_filename = sys.argv[1]
    # Check that input_filename is just a filename and NOT a file path
    if os.path.basename(input_filename) != input_filename:
        raise ValueError("Scurvy dogs, we messed up! \n"
        "The input argument should be a single filename (with .log, .txt, or .tfm extension) and NOT a file path. \n"
        "Also, double-check that the parent directory listed in the bash script is correct. \n"
        "Fix this blunder and give it another go, matey!")
    input_filepath = "/data/" + input_filename

    start_time = datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('US/Eastern')).strftime('%Y%m%d_%H%M%S')  # Default to Eastern timezone!
    setup_logging(input_filepath, start_time)
    logging.info("Ahoy! Welcome to the motion-monitor, let's start monitoring some motion!")
    logging.info(f"Input filepath : {input_filepath}")
    logging.info("")

    # Determine input type for pre-processing
    if os.path.isfile(input_filepath):
        if input_filepath.endswith(".log"):
            transform_list, sms_factor, nslices_per_vol, series_name = get_data_from_slimm_log(input_filepath)
        elif input_filepath.endswith(".txt") or input_filepath.endswith(".tfm"):
            directory_path = os.path.dirname(input_filepath)
            transform_list, sms_factor, nslices_per_vol, series_name = get_data_from_transforms(directory_path)
        else:
            raise ValueError("Arrr! Unsupported file extension. Please provide a .log, .txt, or .tfm file as input.")
    else:
        raise ValueError("Arrr! The input path is not a valid file.")

    # Read in remaining user-specified values
    try:
        radius = float(sys.argv[2])     # spherical head radius assumption (mm)
    except (IndexError, ValueError):
        print("Radius not specified. Using default value.")
        radius = 50.0

    try:
        threshold_value = float(sys.argv[3])    # threshold of acceptable motion (mm)
    except (IndexError, ValueError):
        print("Threshold value not specified. Using default value.")
        threshold_value = 0.6

    logging.info("")
    logging.info(f"User-specified values:")
    logging.info(f"\tHead radius (mm) : {radius}")
    logging.info(f"\tMotion threshold (mm) : {threshold_value}")

    # Displacement between acquisitions
    logging.info("")
    logging.info("Tally ho! Calculating motion measures from the acquired transform parameters...")
    rotation_center, transform_list = get_rotation_center(transform_list)
    displacements, euler_transform_list = compute_transform_pair_displacement(transform_list, rotation_center, radius)

    # Cumulative displacement
    cumulative_disp = sum(displacements)
    logging.info(f"Cumulative sum of displacement (mm) : {cumulative_disp}")

    # Average motion per acquisition
    motion_per_acquisition = calculate_motion_per_minute(displacements)

    # Check displacements of each volume against motion threshold
    total_volumes, volumes_above_threshold, volume_id, motion_flag = check_volume_motion(displacements, sms_factor, nslices_per_vol, threshold_value)

    # Plot transform parameters
    titles = ['X-axis Rotation', 'Y-axis Rotation', 'Z-axis Rotation', 'X-axis Translation', 'Y-axis Translation', 'Z-axis Translation']
    y_labels = ['X Rotation (deg)', 'Y Rotation (deg)', 'Z Rotation (deg)', 'X Translation (mm)', 'Y Translation (mm)', 'Z Translation (mm)']
    params_filename = create_output_file(input_filepath, f"{series_name}_parameters", "png", start_time)
    plot_parameters(euler_transform_list, series_name, params_filename, titles, y_labels, threshold_value, radius)
    params_dist_filename = create_output_file(input_filepath, f"{series_name}_parameters_distribution", "png", start_time)
    plot_parameter_distributions(euler_transform_list, series_name, params_dist_filename)

    # Plot displacements
    disp_filename = create_output_file(input_filepath, f"{series_name}_displacements", "png", start_time)
    plot_displacements(displacements, series_name, disp_filename, threshold=threshold_value, total_volumes=total_volumes, volumes_above_threshold=volumes_above_threshold)

    # Plot cumulative displacement over time
    cumulative_displacements = calculate_cumulative_displacement(displacements)
    cum_disp_filename = create_output_file(input_filepath, f"{series_name}_displacements_cumulative", "png", start_time)
    plot_cumulative_displacement(cumulative_displacements, series_name, cum_disp_filename, threshold=threshold_value, total_volumes=total_volumes, volumes_above_threshold=volumes_above_threshold)

    # Plot per-volume binary motion flag
    motion_flags_filename = create_output_file(input_filepath, f"{series_name}_motion_flags", "png", start_time)
    plot_motion_flags((np.array(volume_id)[..., np.newaxis]), (np.array(motion_flag)[..., np.newaxis]), series_name, motion_flags_filename, sms_factor, nslices_per_vol, threshold_value)

    # Export table of motion data (.csv file)
    data_table, data_table_headers = construct_data_table(np.array(euler_transform_list), np.array(displacements), np.array(cumulative_displacements), np.array(volume_id), np.array(motion_flag))
    csv_filename = create_output_file(input_filepath, f"{series_name}_datatable", "csv", start_time)
    export_values_csv(data_table, data_table_headers, csv_filename)

    logging.info("")
    logging.info("...motion has been monitored. Come back soon!")