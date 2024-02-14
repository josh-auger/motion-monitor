#!/usr/bin/env python3

# Run with python3

# Other code dependencies:
# pyhelpers/compute_displacement.py

# Example run command:
# python3 extract_params_from_log.py [filepath to log file, including filename]
# python3 extract_params_from_log.py /home/jauger/GitHubRepos/sms-mi-reg/pyhelpers/test_data/slimm_logs_test/slimm_nar_2024-01-17_11.59.53.log

import re
import os
import sys
import argparse
import matplotlib.pyplot as plt
import SimpleITK as sitk
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

    return extracted_numbers, skipped_lines, skipped_lines_count

def create_euler_transform(parameters, rotation_center=[0.0, 0.0, 0.0]):
    euler_transform = sitk.Euler3DTransform()
    euler_transform.SetParameters(parameters)
    euler_transform.SetCenter(rotation_center)
    return euler_transform

def compute_transform_pairs(extracted_numbers):
    num_instances = len(extracted_numbers)
    displacements = []
    for i in range(num_instances - 1):
        parameters_i = extracted_numbers[i]
        parameters_next = extracted_numbers[i + 1]

        transform_i = create_euler_transform(parameters_i)
        transform_next = create_euler_transform(parameters_next)

        displacement_value = compute_displacement(transform_i, transform_next)
        displacements.append(displacement_value)
    return displacements

def check_volume_motion(displacements, sms_factor, num_slices_per_volume, threshold):
    # Calculate the number of acquisitions per volume
    num_acquisitions_per_volume = int(num_slices_per_volume / sms_factor)
    total_volumes = (len(displacements) / num_acquisitions_per_volume) + 1 # plus 1 for the initial reference volume
    volumes_above_threshold = 0

    # Iterate over the displacement values
    for i in range(0, len(displacements), num_acquisitions_per_volume):
        volume_displacements = displacements[i:i + num_acquisitions_per_volume]
        if any(d > threshold for d in volume_displacements):
            volumes_above_threshold += 1

    return total_volumes, volumes_above_threshold

def plot_parameters(extracted_numbers, indices_to_plot=[0, 1, 2, 3, 4, 5], log_filename="", titles=None, y_labels=None, rot_thresh=None, trans_thresh=None):
    log_file_path, log_file_name = os.path.split(log_filename)

    # Create an output folder if it doesn't exist
    output_folder = os.path.join(log_file_path, f"{os.path.splitext(log_file_name)[0]}_outputs")
    os.makedirs(output_folder, exist_ok=True)

    num_indices = len(indices_to_plot)

    # Check if all specified indices are valid
    valid_indices = all(0 <= index < len(extracted_numbers[0]) for index in indices_to_plot)
    if not valid_indices:
        print("Error: One (or more) indices are out of range.")
        return

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

    # Save the plot with a .png extension
    base_name, _ = os.path.splitext(log_file_name)
    plot_filename = os.path.join(output_folder, f"{base_name}_parameters.png")
    counter = 1
    while os.path.exists(plot_filename):
        plot_filename = os.path.join(output_folder, f"{base_name}_parameters_{counter}.png")
        counter += 1
    plt.savefig(plot_filename)
    print(f"\nParameters plot saved as: {plot_filename}")

    plt.ion()
    plt.show(block=False)   # plt.show() is otherwise a blocking function pausing code execution until fig is closed

def plot_displacements(displacements, log_filename, threshold=None, total_volumes=None, volumes_above_threshold=None):
    log_file_path, log_file_name = os.path.split(log_filename)

    # Create an output folder if it doesn't exist
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
    total_sets = len(displacements) + 1  # Total number of sets (including first acquisition group)
    text = f'Number of Acquisitions: {total_sets}\nCumulative Displacement (mm): {cumulative_sum:.3f}'
    if total_volumes is not None and volumes_above_threshold is not None:
        text += f'\nTotal Collected Volumes: {total_volumes:.3f}\nVolumes with Motion: {volumes_above_threshold:.3f}\nVolumes without Motion: {(total_volumes - volumes_above_threshold):.3f}'
    plt.text(0.5, 0.9, text, ha='center', va='center', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.7))


    # Save the plot with a .png extension
    base_name, _ = os.path.splitext(log_file_name)
    plot_filename = os.path.join(output_folder, f"{base_name}_displacements.png")
    counter = 1
    while os.path.exists(plot_filename):
        plot_filename = os.path.join(output_folder, f"{base_name}_displacements_{counter}.png")
        counter += 1
    plt.savefig(plot_filename)
    print(f"Displacements plot saved as: {plot_filename}")

    plt.ion()
    plt.show(block=True)


if __name__ == "__main__":
    # # Parse command-line arguments
    # parser = argparse.ArgumentParser(description="Extract parameters from log file and plot specified indices.")
    # parser.add_argument("log_filename", type=str, help="Path to the log file")
    # args = parser.parse_args()
    # log_filename = args.log_filename

    log_file = sys.argv[1]
    log_filename = "/data/" + log_file

    # Extract multi-band (SMS) factor value
    sms_factor_line = find_lines_with_phrase(log_filename, "MultiBandFactor", "</value>")
    sms_factor, *_ = extract_numbers_from_lines(sms_factor_line[:1],r'<value>(.*?)<\/value>')
    sms_factor = float(sms_factor[0][0]) # unpack nested list value storage (used for saving transform parameters)

    # Extract number of slices per volume
    num_vol_slices_line = find_lines_with_phrase(log_filename,"Number of slices per volume:")
    num_vol_slices, *_ = extract_numbers_from_lines(num_vol_slices_line,r'(\d+)\.$')
    num_vol_slices = float(num_vol_slices[0][0])

    # Find all lines reporting transform parameters
    lines_with_params = find_lines_with_phrase(log_filename, line_search_phrase = "FOR-REPORT", additional_search_phrase = "Kalman filtering")

    # Extract numbers from the lines and count skipped lines
    extracted_numbers, skipped_lines, skipped_lines_count = extract_numbers_from_lines(lines_with_params, number_search_pattern = r'\[(.*?)\]')

    # Compose transforms and calculate displacement between acquisitions
    displacements = compute_transform_pairs(extracted_numbers)

    # Establish thresholds for motion
    pixel_size = 2.4
    threshold_value = 0.75 # pixel_size*0.25  # threshold for acceptable motion allowed

    # Check each volume for motion
    total_volumes, volumes_above_threshold = check_volume_motion(displacements, sms_factor, num_vol_slices, threshold_value)

    # Plot the specified number in each set
    indices_to_plot = [0, 1, 2, 3, 4, 5] # remember base 0 indexing!
    titles = ['X-axis Rotation', 'Y-axis Rotation', 'Z-axis Rotation', 'X-axis Translation', 'Y-axis Translation', 'Z-axis Translation']
    y_labels = ['X Rotation (rads)', 'Y Rotation (rads)', 'Z Rotation (rads)', 'X Translation (mm)', 'Y Translation (mm)', 'Z Translation (mm)']
    rot_thresh = threshold_value / 50 # angle corresponding to arc length of 0.6 mm, radius = 50 mm
    trans_thresh = threshold_value # 25% of the pixel width
    plot_parameters(extracted_numbers, indices_to_plot, log_filename, titles, y_labels, rot_thresh, trans_thresh)

    # Plot displacements
    plot_displacements(displacements, log_filename, threshold=threshold_value, total_volumes=total_volumes, volumes_above_threshold=volumes_above_threshold)

    # Print results
    # print("Lines with specified phrase(s):")
    # for line in lines_with_phrases:
    #     print(line)
    print("\nTotal number of lines found:", len(lines_with_params))
    # print("\nExtracted parameters (first 5 sets):")
    # for numbers_set in extracted_numbers[:5]:
    #     print(numbers_set)
    print("Number of extracted parameter sets:", len(extracted_numbers))
    print("Skipped lines (missing end bracket, ']'):", skipped_lines_count)
    for skipped_line, error_message in skipped_lines:
        print(f"Line: {skipped_line}, Error: {error_message}")
    # print("\nDisplacements:")
    # for displacement_value in displacements[:5]:
    #     print(displacement_value)
    print("\nNumber of displacement values:", len(displacements))
    cumulative_disp = sum(displacements)
    print("Cumulative sum of displacement:", cumulative_disp)


    print("\nSMS factor = ", sms_factor)
    print("Num slices per volume:", num_vol_slices)
    print("Num acquisitions per volume:", int(num_vol_slices/sms_factor))
    print("Completed volumes (+ ref vol):", total_volumes)
    print("Volumes with motion:", volumes_above_threshold)
    print("Volumes without motion:", (total_volumes - volumes_above_threshold))