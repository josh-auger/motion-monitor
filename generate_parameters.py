import re
import os
import sys
import argparse
import random
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
from compute_displacement import compute_displacement

def create_uniform_arrays(length, array_length, min_val, max_val):
    """
    Generate a list of number arrays where each array contains numbers sampled from a uniform distribution.

    Args:
        length (int): Length of the list.
        array_length (int): Number of elements in each array.
        min_val (float): Minimum value of the uniform distribution.
        max_val (float): Maximum value of the uniform distribution.

    Returns:
        list: List of number arrays.
    """
    result = []
    for _ in range(length):
        array = [random.uniform(min_val, max_val) for _ in range(array_length)]
        # Add three zeros after the first three numbers
        array += [0, 0, 0]
        result.append(array)
        print("Simulated parameters : ", array)
    return result


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

def compute_motion_score(extracted_numbers, r=50):
    num_instances = len(extracted_numbers)
    displacements = []
    for i in range(num_instances - 1):
        param1 = np.array(extracted_numbers[i])
        param2 = np.array(extracted_numbers[i + 1])
        dp = param2 - param1
        theta = np.abs(np.arccos(0.5 * (-1 + np.cos(dp[0])*np.cos(dp[1])
            + np.cos(dp[0])*np.cos(dp[2]) + np.cos(dp[1])*np.cos(dp[2])
            + np.sin(dp[0])*np.sin(dp[1])*np.sin(dp[2]))))
        drot = r * np.sqrt((1-np.cos(theta))**2 + np.sin(theta)**2)
        dtrans = np.linalg.norm(dp[3:])
        displacement_value = drot + dtrans
        displacements.append(displacement_value)
    return displacements

def calculate_percent_diff(array1, array2):
    array1 = np.array(array1)
    array2 = np.array(array2)
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length.")

    percent_diff = np.abs((array2 - array1) / ((array1 + array2) / 2)) * 100
    return percent_diff

def plot_displacements(displacements, threshold=None, title=None):

    plt.figure(figsize=(10, 6))
    plt.plot(displacements, marker='o', linestyle='-', color='b', alpha=0.7, label='Displacements (mm)')
    plt.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
    if threshold is not None:
        plt.axhline(y=threshold, color='r', linestyle='--', linewidth=3, alpha=1.0, label=f'Threshold = {threshold} mm')

    plt.title('Displacement values : ' + title)
    plt.xlabel('Acquisition (slice timing) group')
    plt.ylabel('Displacement (mm)')
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.ion()
    plt.show(block=False)

def plot_percent_difference(percent_diff):
    plt.figure(figsize=(10, 6))
    plt.plot(percent_diff, marker='o')
    plt.xlabel('Acquisition (slice timing) group')
    plt.ylabel('Percent Difference')
    plt.title('Percent Difference between l2 norm and Yao slimm')
    plt.grid(True)

    plt.show(block=True)


if __name__ == "__main__":

    extracted_numbers = create_uniform_arrays(5000,3,(np.pi/100),(np.pi/10))

    # Compose transforms and calculate displacement between acquisitions
    displacements = compute_transform_pairs(extracted_numbers)
    displacements_slimm = compute_motion_score(extracted_numbers, r=50)
    percent_diff = calculate_percent_diff(displacements_slimm, displacements)

    # Establish thresholds for motion
    threshold_value = 0.75

    # Plot measures
    plot_displacements(displacements, threshold=threshold_value, title='L2 norm')
    plot_displacements(displacements_slimm, threshold=threshold_value, title='Yao SLIMM')
    plot_percent_difference(percent_diff)

    # Print results
    print("\nNum displacement values:", len(displacements))
    cumulative_disp = sum(displacements)
    print("Cumulative sum of displacement:", cumulative_disp)

    print("Mean percent difference:", np.mean(percent_diff))
    print("Std-dev percent difference:", np.std(percent_diff))
    print("Min percent difference:", np.min(percent_diff))
    print("Max percent difference:", np.max(percent_diff))


