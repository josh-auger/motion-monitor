
# Title: update-confounds-with-motion-monitor.py

# Description:
# Update the motion outlier columns in a confounds file (.tsv) to reflect the motion-flagged volumes from the
# motion-monitor output (.csv).

# Example run command:
# python3 update-confounds-with-motion-monitor.py --confounds <path/to/confounds/file.tsv> --motionmonitor <path/to/motion-monitor/data/table.csv>

# Created on: September 2024
# Created by: Joshua Auger (joshua.auger@childrens.harvard.edu), Computational Radiology Lab, Boston Children's Hospital

import pandas as pd
import numpy as np
import argparse


def condense_motion_flags(motion_df):
    """
    Condense the motion monitor table so that for each unique volume number, there is a single motion flag.
    The motion flag is 1 if any of the rows with the same volume number have a motion_flag of 1, and 0 otherwise.
    """
    condensed_motion_df = motion_df.groupby('Volume_number')['Motion_flag'].max().reset_index()

    # Add initial 0 entry for reference volume (Volume_number = 0, Motion_flag = 0), ONLY IF ref vol was not included in registration!
    # This would be the case for SLIMM logs where the reference volume is not registered with itself
    # Nick's "brute force motion char" script already
    # initial_entry = pd.DataFrame({'Volume_number': [0], 'Motion_flag': [0]})
    # condensed_motion_df = pd.concat([initial_entry, condensed_motion_df], ignore_index=True)

    print("Condensed motion dataframe:")
    print(condensed_motion_df)

    return condensed_motion_df


def create_motion_flag_matrix(condensed_motion_df, total_volumes):
    """
    Create a motion flag matrix where each column corresponds to a volume with a motion flag of 1.
    Each column will have all zeros except for a 1 in the row that corresponds to the volume number with the motion flag.
    """
    flagged_volumes = condensed_motion_df[condensed_motion_df['Motion_flag'] == 1]['Volume_number'].values
    print(f"Number of motion-flagged volumes: {len(flagged_volumes)}")

    # Initialize an empty MxN matrix of zeros (M = total num volumes, N = num flagged volumes)
    print(f"Generating motion flag matrix of size {total_volumes} x {len(flagged_volumes)}")
    motion_flag_matrix = np.zeros((total_volumes, len(flagged_volumes)))

    # Iterate through each flagged volume and set the appropriate row to 1 in the corresponding column
    volume_map = {vol: idx for idx, vol in enumerate(condensed_motion_df['Volume_number'].unique())}    # map to zero-based indices
    for idx, volume_number in enumerate(flagged_volumes):
        mapped_idx = volume_map[volume_number]
        motion_flag_matrix[mapped_idx, idx] = 1

    print("Motion flag matrix:")
    print(motion_flag_matrix)

    return motion_flag_matrix


def remove_motion_outlier_columns(confounds_df):
    """
    Remove columns from the confounds dataframe that contain 'motion_outlier' in the column name.
    Return the modified dataframe and print how many columns were removed.
    """
    motion_outlier_cols = [col for col in confounds_df.columns if 'motion_outlier' in col]
    confounds_df_cleaned = confounds_df.drop(columns=motion_outlier_cols)
    print(f"Number of prior 'motion_outlier' columns removed: {len(motion_outlier_cols)}")

    return confounds_df_cleaned


def append_motion_flag_matrix(confounds_df, motion_flag_matrix):
    """
    Append the motion_flag_matrix as new columns to the confounds dataframe.
    The new columns should be named 'motion_outlier00', 'motion_outlier01', etc.
    """
    # Generate new column names like 'motion_outlier00', 'motion_outlier01', etc.
    num_new_columns = motion_flag_matrix.shape[1]
    new_column_names = [f'motion_outlier{str(i).zfill(2)}' for i in range(num_new_columns)]
    print(f"Number of new 'motion_outlier' columns added: {num_new_columns}")

    # Convert motion_flag_matrix (numpy array) to a DataFrame with column names, concatenate with confounds
    motion_flag_df = pd.DataFrame(motion_flag_matrix, columns=new_column_names)
    confounds_df_extended = pd.concat([confounds_df, motion_flag_df], axis=1)

    return confounds_df_extended


def main():
    parser = argparse.ArgumentParser(description='Update a confounds file with motion-monitor results.')
    parser.add_argument('--confounds', type=str, help='Full filepath to the confounds file (*.tsv).')
    parser.add_argument('--motionmonitor', type=str, help='Full filepath to the motion monitor data table (*.csv).')
    args = parser.parse_args()

    # Step 1: read in input files
    confounds_filepath = args.confounds
    motion_monitor_filepath = args.motionmonitor
    confounds_df = pd.read_csv(confounds_filepath, sep='\t')
    motion_df = pd.read_csv(motion_monitor_filepath)

    # Step 2: condense slicewise motion flags to volume motion flags
    condensed_motion_df = condense_motion_flags(motion_df)
    total_volumes = len(condensed_motion_df['Volume_number'].unique())
    print(f"Total number of volumes from motion-monitor : {total_volumes}")

    # Check that num rows in motion_flag_matrix == num rows in confounds dataframe
    # Account for potential missing reference volume in the motion-monitor
    # This is usually the case when processing SLIMM logs vs Nick's "brute force" retrospective motion_char_full.py
    if (total_volumes + 1) == condensed_motion_df.shape[0]:
        print(f"Volume count MISMATCH : motion-monitor [{total_volumes}] vs. confounds file [{confounds_df.shape[0]}].")
        print(f"Motion-monitor is likely missing the initial reference volume. Adding an initial zero-motion entry.")
        initial_entry = pd.DataFrame({'Volume_number': [0], 'Motion_flag': [0]})
        condensed_motion_df = pd.concat([initial_entry, condensed_motion_df], ignore_index=True)
        total_volumes = len(condensed_motion_df['Volume_number'].unique())

    # Now double-check that the number of volumes from the motion-monitor and confounds file match!
    if total_volumes != confounds_df.shape[0]:
        raise ValueError(f"Volume count MISMATCH : motion-monitor [{total_volumes}] vs. confounds file [{confounds_df.shape[0]}].")
    else:
        print(f"Volume counts match : motion-monitor [{total_volumes}], confounds file [{confounds_df.shape[0]}].")

    # Step 3: create motion flag matrix formatted for confounds file, separate binary column array for each volume motion flag
    motion_flag_matrix = create_motion_flag_matrix(condensed_motion_df, total_volumes)

    # Step 4: update confounds file with new motion outlier columns
    # Remove preexisting 'motion_outlier' columns from the confounds file and print the count
    confounds_df_cleaned = remove_motion_outlier_columns(confounds_df)
    confounds_df_updated = append_motion_flag_matrix(confounds_df_cleaned, motion_flag_matrix)

    new_confounds_filepath = confounds_filepath.replace('.tsv', '_slimm_moco.tsv')
    confounds_df_updated.to_csv(new_confounds_filepath, sep='\t', index=False)
    print(f"Confounds file has been updated and saved as '{new_confounds_filepath}'.")


if __name__ == "__main__":
    main()
