"""
Title: calculate_tsnr.py

Description:
Compute the temporal signal-to-noise ratio (tSNR) of a directory of nifti files and generate the equivalent nifti SNR 3Ds.
Library dependencies: nibabel, matplotlib and scipy

References:
    Adapted from https://gist.github.com/arokem/937534/0dec33cc0c302292642b10bce0a5855737550b1f

Author: Joshua Auger (joshua.auger@childrens.harvard.edu), Computational Radiology Lab, Boston Children's Hospital
Date of creation: March 19, 2025
"""

import os
import sys
from glob import glob
from tqdm import tqdm
import numpy as np
from numpy import nanmean
import matplotlib.pyplot as plt
from scipy.io import savemat
from nibabel import load, save
import nibabel.nifti1 as nifti
import argparse

def usage():
    print("Usage: python3 calculate_tsnr.py [option flags] <nifti_directory>")


def help_me():
    help_text = """ 
    This script calculates the temporal Signal-to-Noise Ratio (tSNR) for a directory of NIfTI files.

    Usage:
        python3 calculate_tsnr.py [option flags] <nifti_directory>

    Input(s):
        nifti_directory   Path to the directory containing NIfTI 3D or 4D image files (*.nii or *.nii.gz).

    Options:
        -h, --help        Display help message and exit script.
        -c, --cat         Concatenate all input NIfTI files and compute tSNR on the combined dataset.
        -v, --verbose     Verbose mode for more detailed logging during processing.
        -p, --plot        Generate and save a bar plot of tSNR values for each processed dataset.

    Processing Modes:
        - If --cat flag is used, all NIfTI files in the directory are concatenated along a new axis (assumed to be time), 
            and a single tSNR image is calculated for the combined dataset. This is ideal for a directory of 3D image volumes.
        - If --cat flag is NOT used, each NIfTI file is processed separately, and an individual tSNR image is calculated 
            for each input file. This is ideal for a directory of 4D scan data.

    Output(s):
        - Generates sub-directory '/TSNR' inside input directory for all output files.
        - For each processed dataset, a tSNR NIfTI image is saved.
        - A MATLAB (.mat) file of the tSNR values is also saved for each dataset.
        - If --plot is specified, a bar plot of all tSNR values is saved as 'plot_tsnr.png'.
    """
    print(help_text)
    sys.exit(0)


def calculate_tsnr(data, affine, output_path, verbose=False):
    """Compute temporal SNR (tSNR) from a NIfTI dataset and save output as .nii.gz and .mat"""
    mean_data = np.mean(data, axis=-1)
    std_data = np.std(data, axis=-1)

    # Avoid division by zero by replacing 0s with NaN before division
    std_data[std_data == 0] = np.nan
    tsnr_map = mean_data / std_data
    tsnr_map[np.isinf(tsnr_map)] = np.nan  # Replace infinities with NaNs
    tsnr_map = np.nan_to_num(tsnr_map, nan=0)  # Replace NaNs with 0
    mean_tsnr = np.nanmean(tsnr_map)
    std_tsnr = np.nanstd(tsnr_map)

    # Save as NIfTI images
    save(Nifti1Image(tsnr_map, affine), output_path + "_tsnr.nii.gz")
    save(Nifti1Image(mean_data, affine), output_path + "_mean_signal.nii.gz")
    save(Nifti1Image(std_data, affine), output_path + "_stddev_signal.nii.gz")

    # Save as .mat files
    savemat(output_path + "_tsnr.mat", {'tsnr': tsnr_map})
    savemat(output_path + "_mean_signal.mat", {'mean': mean_data})
    savemat(output_path + "_std_signal.mat", {'std': std_data})

    if verbose:
        print(f"tSNR image saved: {output_path}_tsnr.nii.gz")
        print(f"Mean signal image saved: {output_path}_mean.nii.gz")
        print(f"Standard deviation image saved: {output_path}_std.nii.gz")
        print(f"tSNR values saved: {output_path}_tsnr.mat")
        print(f"Mean signal values saved: {output_path}_mean.mat")
        print(f"Standard deviation values saved: {output_path}_std.mat")
        print(f"Mean tSNR: {mean_tsnr}")
        print(f"Std-dev tSNR: {std_tsnr:.4f}")

    return mean_tsnr


def process_nifti_files(input_path, concatenate, plot, verbose):
    """Processes a directory of NIfTI files and calculates tSNR."""
    if not os.path.exists(input_path):
        sys.exit(f"Error: Directory '{input_path}' not found.")

    tsnr_path = os.path.join(input_path, "TSNR")    # Create /TSNR sub-directory for output files
    os.makedirs(tsnr_path, exist_ok=True)

    nifti_files = sorted(glob(os.path.join(input_path, "*.nii*")))  # Ingest all nifti files in directory, sorted by filename
    if not nifti_files:
        sys.exit("Error: No NIfTI files found in the directory.")
    if verbose:
        print(f"Processing {len(nifti_files)} NIfTI files from {input_path}...")

    snr_values = []
    if concatenate: # Concatenate all image files together along 4th dimension (i.e. time)
        first_img = load(nifti_files.pop(0))
        data_list = [first_img.get_fdata()]
        affine = first_img.affine
        base_shape = data_list[0].shape

        for file in tqdm(nifti_files, desc="Concatenating NIfTI files", unit="file"):
            new_data = load(file).get_fdata()
            if new_data.shape != base_shape:
                print(f"Skipping {file}: Shape mismatch {new_data.shape} != {base_shape}")
                continue
            data_list.append(new_data)

        combined_data = np.stack(data_list, axis=-1)
        print(f"Concatenated data shape : {combined_data.shape}")
        output_file = os.path.join(tsnr_path, "mean_tsnr")
        snr_values.append(calculate_tsnr(combined_data, affine, output_file, verbose))
        print(f"Concatenated mean tSNR : {snr_values[0]}")

    else:   # Process each image file separately along final dimension (time for 4D nifti files, or z-axis for 3D nifti image volumes)
        for file in tqdm(nifti_files, desc="Processing NIfTI files", unit="file"):
            fname = os.path.basename(file).rsplit('.', 1)[0]
            img = load(file)
            data = img.get_fdata()
            affine = img.affine
            if verbose:
                print(f"Processing {fname} with shape {data.shape}")

            output_file = os.path.join(tsnr_path, f"{fname}_tsnr")
            snr_values.append(calculate_tsnr(data, affine, output_file, verbose))

    if plot:
        generate_plot(snr_values, tsnr_path, verbose)


def generate_plot(snr_values, output_dir, verbose):
    """Generates and saves a bar plot of tSNR values."""
    fig, ax = plt.subplots()
    ax.bar(range(1, len(snr_values) + 1), snr_values)
    ax.set_xticks(np.arange(1, len(snr_values) + 1, 5))  # Major ticks at data points
    ax.set_xticks(np.arange(1, len(snr_values) + 1, 1), minor=True)  # Minor ticks between data points
    ax.set_xlabel("File(s)")
    ax.set_ylabel("Mean tSNR")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

    plot_filename = os.path.join(output_dir, "plot_tsnr.png")
    fig.savefig(plot_filename)
    if verbose:
        print(f"tSNR plot saved: {plot_filename}")


def main():
    parser = argparse.ArgumentParser(add_help=False)  # Disable default help
    parser.add_argument("nifti_directory", nargs="?", help="Path to the directory containing NIfTI files.")
    parser.add_argument("-c", "--cat", action="store_true", help="Concatenate NIfTI files into single dataset before computing tSNR.")
    parser.add_argument("-p", "--plot", action="store_true", help="Generate and save a plot of mean tSNR values of each dataset.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging mode.")
    parser.add_argument("-h", "--help", action="store_true", help="Show help message and exit.")

    args = parser.parse_args()

    if args.help:
        help_me()

    if not args.nifti_directory:
        print("Error: No NIfTI directory provided.\n")
        help_me()

    process_nifti_files(args.nifti_directory, args.cat, args.plot, args.verbose)


if __name__ == "__main__":
    main()