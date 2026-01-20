#!/usr/bin/env python3

# Title: process_queue_directory.py

# Description:
# Monitors a specified input directory for image files being written into the directory. New image files are identified
# and sent to sms-mi-reg for mutual information-based image registration.

# Created on: March 2025
# Created by: Joshua Auger (joshua.auger@childrens.harvard.edu), Computational Radiology Lab, Boston Children's Hospital

import os
import re
import cv2
import sys
import csv
import glob
import time
import socket
import logging
import argparse
import SimpleITK as sitk
import numpy as np
import pandas as pd
from datetime import datetime
import json
import mjpeg_server_module
from generate_motion_plots import (
    plot_parameters_combined,
    plot_displacements,
    plot_motion_dashboard
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
        "Volume_index",
        "Slice_group_index",
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
    """Load metadata object from json series metadata file."""
    if not os.path.isfile(json_filepath):
        logging.error(f"JSON file not found: {json_filepath}")
        return {}

    with open(json_filepath, "r") as f:
        metadata = json.load(f)
    logging.info(f"Metadata loaded from JSON file: {json_filepath}")
    return metadata


def get_host_ip():
    """Get host IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to an external host — no data is sent, just used to determine the local IP
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def push_img_to_stream(img, width, height, jpeg_quality=80):
    img = cv2.resize(img, (width, height))
    success, jpeg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if success:
        mjpeg_server_module.update_frame(jpeg.tobytes())


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
            "groupcount": 0,
            "prior_transform": None,
            "seen_files": set(),
            "previous_filesize": 0,
            "begintime": None,
            "slice_timings": [],
            "protocol_name": None,
            "total_repetitions": 0,
            "ngroups": 0,
            "sms_factor": 0,
            "motion_table": [],
            "cumulative_displacement": 0.0,
            "motion_flag_count": 0,
            "volume_motion_flag": 0,
            "volume_motion_count": 0,
            "last_plotted_volcount": 0
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
        """Extract sequence protocol name from series metadata object."""
        protocol_name = metadata_object["measurementInformation"]["protocolName"]
        protocol_name = protocol_name.replace(" ", "_") # Replace spaces with underscores
        state["protocol_name"] = protocol_name
        logging.info(f"Series protocol name : {state['protocol_name']}")

    def get_repetitions_from_metadata(metadata_object):
        """Extract sequence repetitions (total expected image volumes) from series metadata object."""
        total_repetitions = metadata_object["encoding"][0]["encodingLimits"]["repetition"]["maximum"]
        logging.info(f"Total sequence repetitions : {total_repetitions}")
        state["total_repetitions"] = total_repetitions

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

        if motion_flag == 1 and state["volume_motion_flag"] == 0:   # only raise volume motion flag once per volume
            state["volume_motion_flag"] = 1
            state["volume_motion_count"] += 1

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
            "Volume_index": state["volcount"],
            "Slice_group_index": state["groupcount"],
            "Motion_flag": motion_flag
        }
        state["motion_table"].append(row)

        def format_params(params, precision=4):
            return "(" + ", ".join(f"{p:.{precision}g}" for p in params) + ")"

        logging.info(f"Volume (group) : {state['volcount']} ({state['groupcount']})")
        logging.info(f"Prior Euler parameters ({state['itemcount'] - 1:04d}) : {format_params(prior_params, 4)}")
        logging.info(f"Current Euler parameters ({state['itemcount']:04d}) : {format_params(current_params, 4)}")
        logging.info(f"Framewise displacement (mm) : {framewise_displacement:04f}")
        # logging.info(f"=================================")
        # logging.info(f"===== MOTION SUMMARY : {state['itemcount']:04d} =====")
        # logging.info(f"\tCumulative displacement (mm) : {state['cumulative_displacement']:04f}")
        # logging.info(f"\tCumulative motion flags : {state['motion_flag_count']}")
        # logging.info(f"\tCurrent volume count : {state['volcount']} (slice group {state['groupcount']})")
        # logging.info(f"\tMotion-free volumes : {(state['volcount'] - state['volume_motion_count'])}")
        # logging.info(f"\tVolumes with motion : {state['volume_motion_count']}")
        # logging.info(f"=================================")
        return

    def plot_motion_data():
        motion_df = motion_table_to_dataframe(state["motion_table"])
        # timestamp = time.strftime("%Y%m%d_%H%M%S")
        parameters_filepath = os.path.join("/working/", f"motionMonitor_parameters_{state['protocol_name']}.jpg")
        displacements_filepath = os.path.join("/working/", f"motionMonitor_framewise_displacement_{state['protocol_name']}.jpg")
        dashboard_filepath = os.path.join("/working/", f"motionMonitor_dashboard_{state['protocol_name']}.jpg")
        if not motion_df.empty:
            plot_parameters_combined(
                motion_df,
                output_filename=parameters_filepath,
                protocol_name=state['protocol_name'])
            plot_displacements(
                motion_df,
                output_filename=displacements_filepath,
                protocol_name=state['protocol_name'],
                threshold=motion_threshold,
                num_expected_volumes=state['total_repetitions'],
                num_moved_volumes=state['volume_motion_count'])
            plot_motion_dashboard(
                motion_df,
                output_filename=dashboard_filepath,
                protocol_name=state['protocol_name'],
                threshold=motion_threshold,
                num_expected_volumes=state['total_repetitions'],
                num_moved_volumes=state['volume_motion_count'],
                host_ip=host_ip,
                host_port=PORT)

        img = cv2.imread(dashboard_filepath)
        push_img_to_stream(img, 1600, 900)
        return

    def export_motion_table_csv(output_dir):
        """Export motion_table to CSV if it exists and is non-empty."""
        motion_table = state.get("motion_table")
        if not motion_table:
            logging.debug("Motion table empty — skipping CSV export.")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f"motionMonitor_data_{state['protocol_name']}_{timestamp}.csv")
        fieldnames = motion_table[0].keys()
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(motion_table)
            logging.info(f"Motion table saved as : {csv_path}")
        except Exception as e:
            logging.error(f"Failed to export motion table CSV: {e}")
        return

    def handle_reset_trigger(filepath):
        """Handle CLOSE-trigger file."""
        logging.info(f"Reset trigger detected : {os.path.basename(filepath)}")
        try:
            os.remove(filepath)
        except Exception as e:
            logging.error(f"Failed to delete reset trigger file {filepath}: {e}")

        # Export motion table BEFORE wiping state
        plot_motion_data()
        export_motion_table_csv(output_dir=input_dir)

        # Reset all state
        # temp_seen = state["seen_files"]
        nonlocal_state = reset_variables()
        state.update(nonlocal_state)
        # state["seen_files"] = temp_seen     # DEV: re-assign all seen files to prevent repeat processing, for now
        logging.info("\n\n---- Motion-monitor reset ----")
        return

    # =====================================================================
    # ESTABLISH SERVER
    # =====================================================================
    # http_url = "http://0.0.0.0:8080/stream.mjpg"
    PORT = 8080
    mjpeg_server_module.start_server(PORT)
    host_ip = get_host_ip()
    logging.info(f"MJPEG stream available at: http://{host_ip}:{PORT}/stream.mjpg\n")
    logging.info(f"NOTE: IF running on crlreconmri SSH server, then use crlreconmri IP address: http://10.27.192.112:8080/stream.mjpg")

    # =====================================================================
    # MAIN MONITOR LOOP
    # =====================================================================
    while True:
        new_files = list_new_files()
        if not new_files:
            time.sleep(0.005)
        else:
            # Start timer for new session
            if state["begintime"] is None:
                reset_logging()
                state["begintime"] = time.time()
                logging.info(f"Started monitoring at : {datetime.now()}")

            # Process each new file
            logging.info(f"Found {len(new_files)} new file(s) to process")
            for fname in new_files:
                new_filepath = os.path.join(input_dir, fname)
                ext = os.path.splitext(fname)[1]

                # Handle CLOSE trigger
                if ext == ".closeM":
                    handle_reset_trigger(new_filepath)
                    continue

                # Handle metadata
                if ext == ".json":
                    store_metadata_filepath(new_filepath)
                    continue

                # Handle transform files
                if ext == ".tfm":
                    get_counters_from_filename(new_filepath)
                    state["itemcount"] += 1
                    if state["itemcount"] == 1:
                        metadata_object = load_metadata_from_json(state["metadata_filepath"])
                        get_slice_timings_from_metadata(metadata_object)
                        get_protocol_name_from_metadata(metadata_object)
                        get_repetitions_from_metadata(metadata_object)

                    if not wait_for_complete_write(new_filepath):
                        state["seen_files"].add(fname)
                        continue

                    if state["prior_transform"] is None:
                        state["prior_transform"] = new_filepath

                    track_framewise_displacement(new_filepath, state["prior_transform"], head_radius, motion_threshold)
                    if (state["volcount"] // 10) > (state["last_plotted_volcount"] // 10):
                        plot_motion_data()
                        state["last_plotted_volcount"] = state["volcount"]

                    state["seen_files"].add(fname)
                    state["prior_transform"] = new_filepath

                    logging.info(f"Uptime (sec) : {time.time() - state['begintime']:.3f}")
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