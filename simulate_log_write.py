#!/usr/bin/env python3

# Run with python3

from tqdm import tqdm
import os
import time

def generate_unique_filename(filename):
    base_name, extension = os.path.splitext(filename)
    counter = 1
    while True:
        new_filename = f"{base_name}_copy_{counter}{extension}"
        if not os.path.exists(new_filename):
            return new_filename
        counter += 1

def simulate_log(input_log_file, output_log_file=None, delay=None):
    if delay is None:
        delay = 0.02  # Default delay to 20 milliseconds if not specified

    print("Simulating log of:",input_log_file)
    print("Line delay (ms):", (delay*1000))

    # If output log file is not specified, generate unique filename based on input log file
    if not output_log_file:
        output_log_file = generate_unique_filename(input_log_file)

    with open(input_log_file, 'r') as input_file, open(output_log_file, 'w') as output_file:
        start_time = time.perf_counter()  # Start time
        total_lines = sum(1 for _ in input_file)  # Count total lines
        input_file.seek(0)  # Reset file pointer to the beginning

        with tqdm(total=total_lines, desc="Progress", unit="lines") as pbar:
            for line in input_file:
                output_file.write(line)
                output_file.flush()  # Flush the buffer to ensure immediate writing
                time.sleep(delay)    # Introduce delay between each line
                pbar.update(1)       # Update tqdm progress bar

        total_time = time.perf_counter() - start_time
        print(f"Total time taken: {total_time:.2f} seconds")

    print("\nSimulated log writing complete.")


if __name__ == "__main__":
    input_log_file = "/home/jauger/Radiology_Research/SLIMM_data/20240209_SLIMM_logs/slimm_resting_2024-02-09_17.11.40.log"
    output_log_file = None

    delay = 0.0003    # line time delay, in seconds

    simulate_log(input_log_file, output_log_file, delay)
