#!/usr/bin/env python3
"""
Title: identify_reference_volume.py

Description:
Identify suitable 3D image volume from a 4D nifti file to use as a reference volume for motion characterization.
Reference volume must be free of intra-volume motion when compared to the subsequent volume acquisition.

The presence of intra-volume motion is determined by computing the normalized cross-correlation (NCC) between slice groups in each volume.

Author: Joshua Auger (joshua.auger@childrens.harvard.edu), Computational Radiology Lab, Boston Children's Hospital
Date of creation: June 22, 2026
"""

import argparse
import json
from collections import defaultdict
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys
import csv


def load_metadata(json_file):
    """
    Load sequence metadata file (.json).
    """
    with open(json_file, "r") as f:
        metadata = json.load(f)

    # Extract protocol name and slice timing array
    protocol_name = metadata.get("ProtocolName", "UNKNOWN")
    slice_timing = metadata.get("SliceTiming")

    if slice_timing is None:
        raise RuntimeError("SliceTiming field not found.")

    return protocol_name, slice_timing


def build_slice_groups(slice_timing):
    """
    Group slices acquired simultaneously.
    """
    groups = defaultdict(list)

    for slice_idx, timing in enumerate(slice_timing):
        groups[timing].append(slice_idx)

    return dict(sorted(groups.items()))


def normalized_cross_correlation(a, b):
    """
    Compute normalized cross-correlation (NCC) between two groups of image slices.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    a = a.ravel()
    b = b.ravel()

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    a -= a_mean
    b -= b_mean

    numerator = np.sum(a * b)

    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))

    if denominator < 1e-12:
        return 0.0

    return numerator / denominator


def compute_group_ncc(volume_a, volume_b, slice_groups):
    """
    Compute NCC for every slice group between two image volumes.
    """
    results = {}

    for group_time, slice_indices in slice_groups.items():
        group_a = volume_a[:, :, slice_indices]
        group_b = volume_b[:, :, slice_indices]
        results[group_time] = normalized_cross_correlation(group_a, group_b)

    return results


def compare_volumes(volume_a, volume_b, slice_groups, threshold):
    """
    Compare all slice group NCC values between two volumes (i, i+1) to determine if volume i passes threshold for intra-volume motion.
    """
    group_results = compute_group_ncc(volume_a, volume_b, slice_groups)

    group_values = np.array(list(group_results.values()))
    min_ncc = np.min(group_values)
    mean_ncc = np.mean(group_values)
    range_ncc = np.max(group_values) - np.min(group_values)

    passed = min_ncc >= threshold

    return (passed, min_ncc, mean_ncc, range_ncc, group_results)


def print_summary(comparison_stats, candidate_volumes):
    """
    Print summary of candidate reference volumes, first stable volume pair, and most stable volume pair.
    """
    logging.info("")
    logging.info("=" * 70)
    logging.info("SUMMARY")
    logging.info("=" * 70)

    if len(candidate_volumes) == 0:
        logging.info("No volume pair passed all slice-group tests.")
        return

    candidate_str = ", ".join(str(v) for v in candidate_volumes)
    logging.info(f"Candidate reference volumes: {candidate_str}")

    first = next(stat for stat in comparison_stats if stat["passed"])

    logging.info("\nFirst stable volume pair:")
    logging.info(f"  Volume {first['volume']} vs Volume {first['volume'] + 1}")
    logging.info(f"  Min NCC   = {first['min_ncc']:.4f}")
    logging.info(f"  Mean NCC  = {first['mean_ncc']:.4f}")
    logging.info(f"  Range NCC = {first['range_ncc']:.4f}")

    best = min(comparison_stats, key=lambda x: x["range_ncc"])

    logging.info("\nMost stable volume pair (smallest NCC range):")
    logging.info(f"  Volume {best['volume']} vs Volume {best['volume'] + 1}")
    logging.info(f"  Min NCC   = {best['min_ncc']:.4f}")
    logging.info(f"  Mean NCC  = {best['mean_ncc']:.4f}")
    logging.info(f"  Range NCC = {best['range_ncc']:.4f}")


def save_slice_group_csv(filename, slice_group_comparison_results):
    """
    Save slice-group NCC comparison results to CSV.
    """
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "volume",
                "next_volume",
                "slice_timing",
                "ncc",
                "threshold",
                "passed",
            ],
        )
        writer.writeheader()
        writer.writerows(slice_group_comparison_results)

    logging.info(f"\nSlice group comparison results saved as: {filename}")


def plot_results(comparison_stats, protocol_name, threshold, save_filename=None,):
    """
    Plot minimum slice group NCC and range of NCC values for each adjacent volume comparison.
    """
    volume_indices = [stat["volume"] for stat in comparison_stats]
    min_ncc_values = [stat["min_ncc"] for stat in comparison_stats]
    range_ncc_values = [stat["range_ncc"] for stat in comparison_stats]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Minimum NCC
    ax1.plot(volume_indices, min_ncc_values, marker="o")
    ax1.axhline(y=threshold, linestyle="--", color="red", label=f"Threshold = {threshold:.3f}")
    ax1.set_ylabel("Minimum slice group NCC")
    ax1.grid(True)
    ax1.legend()

    # NCC Range
    ax2.plot(volume_indices, range_ncc_values,marker="s")
    ax2.set_ylabel("Range slice group NCC")
    ax2.set_xlabel("Volume Comparison Index")
    ax2.grid(True)

    plt.suptitle(f"{protocol_name}\nSlice Group Normalized Cross-Correlation (NCC)")
    plt.tight_layout()
    if save_filename:
        fig.savefig(save_filename, dpi=300, bbox_inches="tight")
        logging.info(f"Plot saved as: {save_filename}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nii", required=True, help="Input 4D nifti scan data file (.nii or .nii.gz)")
    parser.add_argument("--json",required=True,help="Sequence metadata file (.json)")
    parser.add_argument("--threshold",type=float,default=0.99,help="Minimum NCC required to pass")
    parser.add_argument("--plot",action="store_true",help="Plot min NCC and range NCC for all volume comparisons")
    parser.add_argument("--save",action="store_true",help="Save log file (.log), slice group comparison results (.csv), and comparison plot (.png) to current working directory")
    parser.add_argument("--verbose",action="store_true",help="Enable verbose logging of every slice group comparison")
    args = parser.parse_args()
    
    base_name = Path(args.nii).stem
    if base_name.endswith(".nii"):  # Handle .nii.gz correctly
        base_name = Path(base_name).stem

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.save:
        handlers.append(logging.FileHandler(f"{base_name}_refVolCalibration.log", mode="w"))

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers,)

    protocol_name, slice_timing = load_metadata(args.json)
    logging.info(f"\nProtocol name: {protocol_name}")

    slice_groups = build_slice_groups(slice_timing)
    logging.info(f"Detected {len(slice_groups)} slice groups in SliceTiming metadata")

    img = nib.load(args.nii)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise RuntimeError("Expected 4D nifti input.")

    nx, ny, nz, nt = data.shape
    logging.info(f"Shape = {data.shape}\n")

    candidate_volumes = []
    comparison_stats = []
    slice_group_comparison_results = []

    logging.info(f"Comparing volume slices with NCC threshold = {args.threshold:.4f}...")

    for vol_idx in range(nt - 1):
        volume_a = data[:, :, :, vol_idx]
        volume_b = data[:, :, :, vol_idx + 1]

        (passed, min_ncc, mean_ncc, range_ncc, group_results) = compare_volumes(volume_a, volume_b, slice_groups, args.threshold)

        comparison_stats.append(
            {
                "volume": vol_idx,
                "passed": passed,
                "min_ncc": min_ncc,
                "mean_ncc": mean_ncc,
                "range_ncc": range_ncc
            }
        )

        logging.info(f"\nVolume {vol_idx} vs Volume {vol_idx + 1}")
        for group_time, value in group_results.items():
            passed_group = value >= args.threshold
            status = "PASS" if passed_group else "FAIL"
            if args.verbose:
                logging.info(f"  Group {group_time:.4f}   NCC={value:.4f}   {status}")

            slice_group_comparison_results.append({
                "volume": vol_idx,
                "next_volume": vol_idx + 1,
                "slice_timing": group_time,
                "ncc": value,
                "threshold": args.threshold,
                "passed": int(passed_group)
            })

        logging.info(f"  Min NCC   = {min_ncc:.4f}")
        logging.info(f"  Mean NCC  = {mean_ncc:.4f}")
        logging.info(f"  Range NCC = {range_ncc:.4f}")
        if passed:
            candidate_volumes.append(vol_idx)
            logging.info(f"  Reference candidate {vol_idx}: {status}")

    print_summary(comparison_stats, candidate_volumes)

    if args.save:
        save_slice_group_csv(f"{base_name}_refVolCalibration.csv", slice_group_comparison_results)

    if args.plot:
        plot_results(
            comparison_stats,
            protocol_name,
            args.threshold,
            save_filename=(f"{base_name}_refVolCalibration.png" if args.save else None)
        )


if __name__ == "__main__":
    main()