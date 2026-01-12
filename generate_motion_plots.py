#!/usr/bin/env python3

# Title: generate_motion_plots.py

# Description:
# Helper script to generate plots of motion parameters and framewise displacement.

# Created on: December 2025
# Created by: Joshua Auger (joshua.auger@childrens.harvard.edu), Computational Radiology Lab, Boston Children's Hospital

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import csv
import os


def load_motion_csv(csv_filepath):
    """
    Load motion tracking CSV into a pandas DataFrame.
    """
    df = pd.read_csv(csv_filepath)

    required_columns = [
        "X_rotation(rad)", "Y_rotation(rad)", "Z_rotation(rad)",
        "X_translation(mm)", "Y_translation(mm)", "Z_translation(mm)",
        "Displacement(mm)", "Cumulative_displacement(mm)", "Motion_flag"
    ]

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    return df


def plot_parameters_combined(motion_df, output_filename="", trans_thresh=0.60, radius=50):
    """
    motion_df = pandas DataFrame loaded from the motion CSV file
    Plot combined motion parameters with rotations (x, y, z) on one subplot and translations (x, y, z) on another.
    Rotations are converted from radians to degrees.
    """

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    subplot_colors = ['b', 'g', 'r']  # x, y, z colors
    fig.suptitle(f"Motion Parameters", y=0.98)

    # Extract parameter arrays from motion dataframe
    rotations_rad = motion_df[["X_rotation(rad)", "Y_rotation(rad)", "Z_rotation(rad)"]].to_numpy()
    translations = motion_df[["X_translation(mm)", "Y_translation(mm)", "Z_translation(mm)"]].to_numpy()
    # Convert rotations to degrees
    rotations = np.degrees(rotations_rad)

    # Rotation plot
    ax_rot = axes[0]
    for i, label in enumerate(['X', 'Y', 'Z']):
        ax_rot.plot(rotations[:, i], marker='o', linestyle='-', color=subplot_colors[i],
                    alpha=0.6, label=f'{label}-axis')
    ax_rot.set_title('Rotations')
    ax_rot.set_xlabel('Acquisition Index')
    ax_rot.set_ylabel('Rotation (deg)')
    ax_rot.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
    ax_rot.set_xlim(left=0)
    # rot_thresh = np.degrees(trans_thresh / radius)
    # ax_rot.axhline(y=rot_thresh, color='r', linestyle='--', alpha=0.7,
    #                label=f'Rot Threshold = {rot_thresh:.3f}°')
    # ax_rot.axhline(y=-rot_thresh, color='r', linestyle='--', alpha=0.7)
    ax_rot.legend(loc='upper left')

    # Translation plot
    ax_trans = axes[1]
    for i, label in enumerate(['X', 'Y', 'Z']):
        ax_trans.plot(translations[:, i], marker='o', linestyle='-', color=subplot_colors[i],
                      alpha=0.6, label=f'{label}-axis')
    ax_trans.set_title('Translations')
    ax_trans.set_xlabel('Acquisition Index')
    ax_trans.set_ylabel('Translation (mm)')
    ax_trans.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
    ax_trans.set_xlim(left=0)
    # ax_trans.axhline(y=trans_thresh, color='r', linestyle='--', alpha=0.7,
    #                  label=f'Trans Threshold = {trans_thresh:.3f} mm')
    # ax_trans.axhline(y=-trans_thresh, color='r', linestyle='--', alpha=0.7)
    ax_trans.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_filename)
    logging.info(f"Combined parameters plot saved as : {output_filename}")
    plt.ion()
    plt.show(block=False)


def plot_displacements(motion_df, output_filename="", threshold=None, num_moved_volumes=None):
    displacements = motion_df["Displacement(mm)"].to_numpy()
    cumulative = motion_df["Cumulative_displacement(mm)"].to_numpy()
    volume_index = motion_df["Volume_index"]
    motion_flags = motion_df["Motion_flag"]

    fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3.5, 1]})

    # Plot framewise displacements
    axs[0].plot(displacements, marker='o', alpha=0.7, label="Framewise displacement")
    if threshold is not None:
        axs[0].axhline(threshold, color='r', linestyle='--', label=f"Threshold = {threshold} mm")

    axs[0].set_title("Framewise Displacements")
    axs[0].set_xlabel("Index")
    axs[0].set_ylabel("Displacement (mm)")
    axs[0].legend(loc='upper left')
    axs[0].grid(True)

    # Plot motion summary and displacements boxplot
    num_volumes = max(volume_index)
    if num_moved_volumes is None:
        logging.info(f"Number of motion-corrupt volumes not provided. Treating each motion flag separately.")
        num_moved_volumes = motion_flags.sum()
    num_motion_free_volumes = num_volumes - num_moved_volumes
    counts = [num_motion_free_volumes, num_moved_volumes]

    text = (
        f"Number of acquisitions: {len(displacements)}\n"
        f"Motion flags: {motion_flags.sum()}\n"
        f"Cumulative displacement (mm): {cumulative[-1]:.3f}\n"
        f"Number of volumes: {num_volumes}\n"
        f"Motion-free volumes: {num_motion_free_volumes}\n"
        f"Motion-corrupt volumes: {num_moved_volumes}"
    )
    axs[1].text(0.5, 0.9, text, transform=axs[1].transAxes, ha='center', va='center', multialignment='left',
                bbox=dict(facecolor='white', alpha=0.8))
    axs[1].bar([f"Motion-free", f"Motion-corrupt"], counts)
    ymax = max(counts)
    axs[1].set_ylim(0, ymax * 1.4)
    axs[1].set_ylabel("N volumes")
    axs[1].set_title("Volume motion summary")
    for i, val in enumerate([num_motion_free_volumes, num_moved_volumes]):
        axs[1].text(i, val + 0.5, str(val), ha='center', va='bottom')
    axs[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename)
    logging.info(f"Displacements plot saved as : {output_filename}")
    plt.show(block=True)


def plot_cumulative_displacement(motion_df, output_filename="", threshold=None):
    cumulative = motion_df["Cumulative_displacement(mm)"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cumulative, marker='o', alpha=0.7, label="Cumulative displacement")

    if threshold is not None:
        ax.plot(np.arange(len(cumulative)), threshold * np.arange(len(cumulative)),
                linestyle='--', color='r', label="Acceptable accumulation")

    ax.set_title(f"Cumulative Displacement")
    ax.set_xlabel("Acquisition Instance")
    ax.set_ylabel("Cumulative Displacement (mm)")
    ax.legend(loc='upper left')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_filename)
    logging.info(f"Cumulative displacement plot saved as : {output_filename}")
    plt.show(block=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate motion plots from motion CSV")
    parser.add_argument("motion_csv", help="Path to motion tracking CSV")
    parser.add_argument("--series-name", default="", help="Series name for plot titles")
    parser.add_argument("--outdir", default=".", help="Output directory for plots")
    parser.add_argument("--threshold", type=float, default=None, help="Motion threshold (mm)")
    args = parser.parse_args()

    motion_df = load_motion_csv(args.motion_csv)

    plot_parameters_combined(
        motion_df,
        output_filename=os.path.join(args.outdir, "motion_parameters_combined.png"),
        trans_thresh=args.threshold if args.threshold is not None else 0.75
    )

    plot_displacements(
        motion_df,
        output_filename=os.path.join(args.outdir, "framewise_displacement.png"),
        threshold=args.threshold
    )

    plot_cumulative_displacement(
        motion_df,
        output_filename=os.path.join(args.outdir, "cumulative_displacement.png"),
        threshold=args.threshold
    )
