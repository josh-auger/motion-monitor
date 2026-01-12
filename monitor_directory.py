#!/usr/bin/env python3

# Title: process_queue_directory.py

# Description:
# Monitors a specified input directory for image files being written into the directory. New image files are identified
# and sent to sms-mi-reg for mutual information-based image registration.

# Created on: March 2025
# Created by: Joshua Auger (joshua.auger@childrens.harvard.edu), Computational Radiology Lab, Boston Children's Hospital

import os
import sys
import csv
import time
import logging
import argparse
import subprocess
import SimpleITK as sitk
import numpy as np
import pandas as pd
from datetime import datetime
import json
from generate_motion_plots import (
    plot_parameters_combined,
    plot_displacements,
    plot_cumulative_displacement,
)


def setup_logging():
    """Configure logging to save logs with a timestamped filename in /working/."""
    log_filename = os.path.join("/working/", f"log_motion_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging will be saved to : {log_filename}")


def reset_logging():
    """Reset logging configuration to save logs to a new logfile with a fresh timestamp."""
    log_filename = os.path.join("/working/", f"log_motion_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    # Remove all existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Reconfigure logging with new file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Log reset. Logging will be saved to : {log_filename}")


def select_input_transform(identityTransform_filepath, counter, reference_volume_flag):
    """Identify prior alignment transform file (.tfm) as input initialization transform in next registration call."""
    if reference_volume_flag == 0:
        logging.info(f"Still calibrating reference volume. Input transform for registration : {identityTransform_filepath}")
        return identityTransform_filepath

    prior_transform_filepath = f"/working/alignTransform_{counter - 1:04d}.tfm"
    chosen_transform = prior_transform_filepath if os.path.exists(prior_transform_filepath) else identityTransform_filepath
    # chosen_transform = identityTransform_filepath
    logging.info(f"Input transform for registration : {chosen_transform}")
    return chosen_transform


def compose_transform_pair(transform1, transform2):
    """Calculate the composed transform between two given input transforms."""
    A0 = np.asarray(transform2.GetMatrix()).reshape(3, 3)
    c0 = np.asarray(transform2.GetCenter())
    t0 = np.asarray(transform2.GetTranslation())

    A1 = np.asarray(transform1.GetInverse().GetMatrix()).reshape(3, 3)
    c1 = np.asarray(transform1.GetInverse().GetCenter())
    t1 = np.asarray(transform1.GetInverse().GetTranslation())

    combined_mat = np.dot(A0,A1)
    combined_center = c1
    combined_translation = np.dot(A0, t1+c1-c0) + t0+c0-c1

    euler3d = sitk.Euler3DTransform()
    euler3d.SetCenter(combined_center)
    euler3d.SetTranslation(combined_translation)
    euler3d.SetMatrix(combined_mat.flatten())
    return euler3d


def calculate_displacement(euler3d_transform, radius=50):
    """Calculate framewise displacement from a Euler 3D transform (following Tisdall et al. 2012)"""
    # assumes Euler3D transform as input
    params = np.asarray(euler3d_transform.GetParameters())
    # logging.info(f"\tParameters (Euler3D) : {params}")

    # displacement calculation
    theta = np.abs(np.arccos(0.5 * (-1 + np.cos(params[0]) * np.cos(params[1]) + \
                                    np.cos(params[0]) * np.cos(params[2]) + \
                                    np.cos(params[1]) * np.cos(params[2]) + \
                                    np.sin(params[0]) * np.sin(params[1]) * np.sin(params[2]))))
    drot = radius * np.sqrt((1 - np.cos(theta)) ** 2 + np.sin(theta) ** 2)
    dtrans = np.linalg.norm(params[3:])
    displacement = drot + dtrans
    return displacement


def motion_table_to_dataframe(motion_table):
    """
    Convert the internal motion_table (list of dicts)
    into a pandas DataFrame suitable for plotting/export.
    """
    if not motion_table:
        return pd.DataFrame()

    df = pd.DataFrame(motion_table)
    column_order = [
        "reg_index",
        "X_rotation(rad)", "Y_rotation(rad)", "Z_rotation(rad)",
        "X_translation(mm)", "Y_translation(mm)", "Z_translation(mm)",
        "Displacement(mm)",
        "Cumulative_displacement(mm)",
        "Motion_flag",
    ]
    return df[column_order]


def convert_versor_to_euler(transform):
    """Convert a Versor Rigid 3D transform into an equivalent Euler 3D transform."""
    center = transform.GetCenter()
    translation = transform.GetTranslation()
    rotation_matrix = transform.GetMatrix()

    # Extract Euler angles (in radians) from the Versor rotation matrix
    R = np.array(rotation_matrix).reshape(3, 3)  # Convert to 3x3 matrix

    # Convert to Euler angles (ZYX convention)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    # Create the Euler3DTransform
    euler_transform = sitk.Euler3DTransform()
    euler_transform.SetRotation(x, y, z)  # Angles are in radians
    euler_transform.SetTranslation(translation)
    euler_transform.SetCenter(center)
    return euler_transform


def load_metadata_from_json(json_filepath):
    if not os.path.isfile(json_filepath):
        logging.error(f"JSON file not found: {json_filepath}")
        return {}

    with open(json_filepath, "r") as f:
        metadata = json.load(f)
    logging.info(f"Metadata loaded from JSON file: {json_filepath}")
    return metadata


def monitor_directory(input_dir, head_radius, motion_threshold):
    """Monitor directory for new image files without deleting any."""
    # Initialization
    # -------------------------------------------
    if not os.path.exists(input_dir):
        logging.error(f"Directory '{input_dir}' does not exist.")
        return

    VALID_EXTENSIONS = {'.json', '.tfm', '.closeM'}  # JDA: ONLY read incoming files with listed valid extensions!
    logging.info(f"Monitoring directory [{VALID_EXTENSIONS}] : {input_dir} ...")

    # Reset all monitoring state variables
    # -------------------------------------------
    def reset_variables():
        return {
            "metadata_filepath": None,
            "itemcount": 0,
            "volcount": 0,
            "prior_transform": None,
            "seen_files": set(),
            "previous_filesize": 0,
            "begintime": None,
            "slice_timings": [],
            "protocol_name": None,
            "ngroups": 0,
            "sms_factor": 0,
            "motion_table": [],
            "cumulative_displacement": 0.0,
            "motion_flag_count": 0
        }
    state = reset_variables()

    # Helper functions
    # -------------------------------------------
    def list_new_files():
        """Return list of valid, non-seen files sorted by modification time,
        ignoring files that disappear during the scan."""
        valid_files = []
        for f in os.listdir(input_dir):
            # Skip wrong extensions
            if os.path.splitext(f)[1] not in VALID_EXTENSIONS:
                continue
            # Skip files that have already been processed
            if f in state["seen_files"]:
                continue

            full_path = os.path.join(input_dir, f)
            try:    # attempt to ping for modification time
                mtime = os.path.getmtime(full_path)
            except FileNotFoundError:
                # File disappeared between os.listdir() and os.path.getmtime() (i.e. end of sequence consolidation)
                logging.info(f"Unable to ping file for mod time : {full_path}")
                continue

            valid_files.append((f, mtime))

        # Sort by mtime: oldest → newest
        valid_files.sort(key=lambda x: x[1])
        # Return only new files not yet processed
        return [f for (f, _) in valid_files]

    def wait_for_complete_write(filepath, max_checks=200, delay=0.005):
        """Poll file size until it is stable for one check. True if stable, False if file disappeared."""
        checks = 0
        last_size = -1
        while checks < max_checks:
            if not os.path.exists(filepath):
                logging.warning(f"File no longer exists : {filepath}")  # file moved/deleted (i.e. end of sequence consolidation)
                return False
            size_now = os.path.getsize(filepath)
            if size_now == last_size:
                return True
            last_size = size_now
            checks += 1
            time.sleep(delay)
        # timed out waiting for stability, still proceed but warn
        logging.warning(f"File write did not stabilize after {max_checks} checks for {filepath}; proceeding anyway.")
        return True

    def store_metadata_filepath(filepath):
        """Handle incoming series metadata JSON."""
        if state["metadata_filepath"] is None:
            logging.info(f"Found series metadata file : {os.path.basename(filepath)}")
            state["metadata_filepath"] = filepath
        else:
            logging.info(f"Skipping {filepath} because metadata already loaded.")
        state["seen_files"].add(os.path.basename(filepath))
        return

    def get_slice_timings_from_metadata(metadata_object):
        """Extract the list of image slice acquisition times from series metadata object."""
        if "SliceTiming" not in metadata_object:
            logging.warning(f"'SliceTiming' field not found in series metadata!")
            return {}

        slice_timing_dict = metadata_object["SliceTiming"]
        # Convert to list ordered by slice index
        slice_timing = [slice_timing_dict[str(i)] if str(i) in slice_timing_dict else slice_timing_dict[i]
                        for i in sorted(map(int, slice_timing_dict.keys()))]

        logging.info(f"Slice timings (ordered by index): {slice_timing}")
        state["slice_timings"] = slice_timing

        if slice_timing is not None:
            logging.info(f"Number of slices per volume : {len(slice_timing)}")
            state["ngroups"] = len(np.unique(slice_timing))
            state["sms_factor"] = len(slice_timing) / state["ngroups"]
            logging.info(f"Number of slice groups : {state['ngroups']}")
            logging.info(f"Number of slices per group (sms factor) : {state['sms_factor']}")
        return slice_timing

    def get_protocol_name_from_metadata(metadata_object):
        """Extract series protocol name from series metadata object."""
        protocol_name = metadata_object["measurementInformation"]["protocolName"]
        protocol_name = protocol_name.replace(" ", "_") # Replace spaces with underscores
        state["protocol_name"] = protocol_name
        logging.info(f"Series protocol name : {state['protocol_name']}")

    def get_counters_from_filename(filepath):
        """Extract volume count string and slice group count string from transform filename."""
        basefilename = os.path.basename(filepath)
        match = re.search(r"_(\d{4})-(\d{4})", basefilename)
        if not match:
            raise ValueError(f"Filename does not match expected pattern: {basefilename}")

        if state["volcount"] != int(match.group(1)):    # only update volcount if it is a new volume
            state["volcount"] = int(match.group(1))
            state["volume_motion_flag"] = 0

        state["groupcount"] = int(match.group(2))
        return

    def track_framewise_displacement(current_transform_filepath, prior_transform_filepath, head_radius, motion_threshold):
        """Maintain cumulative ledger of calculated framewise displacements between transform pairs."""
        prior_transform = convert_versor_to_euler(sitk.ReadTransform(prior_transform_filepath))
        current_transform = convert_versor_to_euler(sitk.ReadTransform(current_transform_filepath))
        combined_transform = compose_transform_pair(prior_transform, current_transform)
        prior_params = prior_transform.GetParameters()
        current_params = current_transform.GetParameters()
        framewise_displacement = calculate_displacement(combined_transform, head_radius)
        motion_flag = 1 if framewise_displacement > motion_threshold else 0

        state["motion_flag_count"] += motion_flag
        state["cumulative_displacement"] += framewise_displacement

        row = {
            "reg_index": state["itemcount"],
            "X_rotation(rad)": current_params[0],
            "Y_rotation(rad)": current_params[1],
            "Z_rotation(rad)": current_params[2],
            "X_translation(mm)": current_params[3],
            "Y_translation(mm)": current_params[4],
            "Z_translation(mm)": current_params[5],
            "Displacement(mm)": framewise_displacement,
            "Cumulative_displacement(mm)": state["cumulative_displacement"],
            "Motion_flag": motion_flag
        }
        state["motion_table"].append(row)

        def format_params(params, precision=4):
            return "(" + ", ".join(f"{p:.{precision}g}" for p in params) + ")"

        logging.info(f"=================================")
        logging.info(f"===== MOTION SUMMARY : {state['itemcount']:04d} =====")
        logging.info(f"\tPrior parameters (Euler) : {format_params(prior_params, 4)}")
        logging.info(f"\tCurrent parameters (Euler) : {format_params(current_params, 4)}")
        logging.info(f"\tFramewise displacement (mm) : {framewise_displacement:04f}")
        logging.info(f"\tCumulative displacement (mm) : {state['cumulative_displacement']:04f}")
        logging.info(f"\tCumulative motion flags : {state['motion_flag_count']}")
        logging.info(f"=================================")

        if state["itemcount"] % (10 * state["ngroups"]) == 0:
            motion_df = motion_table_to_dataframe(state["motion_table"])
            if not motion_df.empty:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                plot_parameters_combined(
                    motion_df,
                    output_filename=os.path.join("/working/", f"motionTracker_parameters_{timestamp}.png"),
                    trans_thresh=motion_threshold, )
                plot_displacements(
                    motion_df,
                    output_filename=os.path.join("/working/", f"motionTracker_framewise_displacement_{timestamp}.png"),
                    threshold=motion_threshold, )


    def export_motion_table_csv(output_dir):
        """Export motion_table to CSV if it exists and is non-empty."""
        motion_table = state.get("motion_table")
        if not motion_table:
            logging.debug("Motion table empty — skipping CSV export.")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"motion_table_{timestamp}.csv")
        fieldnames = motion_table[0].keys()
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(motion_table)
            logging.info(f"Motion table exported: {csv_path}")
        except Exception as e:
            logging.error(f"Failed to export motion table CSV: {e}")

    def handle_reset_trigger(filepath):
        """Handle CLOSE-trigger file."""
        logging.info(f"Reset trigger detected : {os.path.basename(filepath)}")
        try:
            os.remove(filepath)
        except Exception as e:
            logging.error(f"Failed to delete reset trigger file {filepath}: {e}")

        # Export motion table BEFORE wiping state
        motion_df = motion_table_to_dataframe(state["motion_table"])
        if not motion_df.empty:
            plot_parameters_combined(
                motion_df,
                output_filename=os.path.join("/working/", "motionTracker_parameters.png"),)
            plot_displacements(
                motion_df,
                output_filename=os.path.join("/working/", "motionTracker_framewise_displacement.png"),)

        export_motion_table_csv(output_dir=input_dir)

        # Reset all state
        nonlocal_state = reset_variables()
        state.update(nonlocal_state)
        logging.info("\n\n---- Motion-monitor reset ----")
        reset_logging()


    # Main monitoring loop
    # =====================================================================
    while True:
        new_files = list_new_files()
        if not new_files:
            time.sleep(0.005)
            continue

        if new_files:
            logging.info(f"Found {len(new_files)} new file(s) to process")

            # Start timer
            if state["begintime"] is None:
                state["begintime"] = time.time()
                logging.info(f"Started monitoring at : {datetime.now()}")

            # Process each new file
            # -------------------------------------------
            for fname in new_files:
                new_filepath = os.path.join(input_dir, fname)
                ext = os.path.splitext(fname)[1]

                # Handle CLOSE trigger file(s)
                if ext == ".closeM":
                    handle_reset_trigger(new_filepath)
                    continue

                # Handle series metadata file
                if ext == ".json":
                    store_metadata_filepath(new_filepath)
                    continue

                # Handle pointer file
                if ext == ".tfm":
                    state["itemcount"] += 1

                    if state["itemcount"] == 1:     # first transform = ref vol identity transform
                        # After first ref volume we can process series metadata for slice timings
                        metadata_object = load_metadata_from_json(state["metadata_filepath"])
                        get_slice_timings_from_metadata(metadata_object)
                        get_protocol_name_from_metadata(metadata_object)

                    get_counters_from_filename(new_filepath)    # update volume count and slice group count from filename

                    if not wait_for_complete_write(new_filepath):
                        state["seen_files"].add(fname)
                        continue

                    # Track new motion parameters
                    if state["prior_transform"] is None:
                        state["prior_transform"] = new_filepath
                    track_framewise_displacement(new_filepath, state["prior_transform"], head_radius, motion_threshold)

                    # Mark file as processed
                    state["seen_files"].add(fname)
                    state["prior_transform"] = new_filepath    # set current transform to be the prior seen transform

                    # Logging
                    logging.info(f"Total processed files : {state['itemcount']}")
                    logging.info(f"Total elapsed time (sec) : {time.time() - state['begintime']:.3f}")
                    logging.info("...")


def main():
    parser = argparse.ArgumentParser(add_help=False)  # Disable default help
    parser.add_argument("input_directory", nargs="?", help="Path to the directory containing image files.")
    parser.add_argument("--radius", type=float, default=50, help="Motion threshold (mm)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Motion threshold (mm)")
    args = parser.parse_args()

    monitor_directory(args.input_directory, args.radius, args.threshold)

if __name__ == "__main__":
    setup_logging()
    main()