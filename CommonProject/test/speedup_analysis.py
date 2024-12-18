import re
import sys
from collections import defaultdict

def main(filename):
    # Check if the file exists and is readable
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    # Dictionaries to store elapsed times and speedups per optimization level
    elapsed_times = defaultdict(list)
    speedups = defaultdict(list)

    current_opt_level = None

    # Parse the input file
    for line in lines:
        # Match the optimization level
        opt_match = re.match(r"Compiling in (-O\d)...", line)
        if opt_match:
            current_opt_level = opt_match.group(1)
            continue

        # Match elapsed time and speedup values
        data_match = re.match(r"-\s+Num samples: \d+\s+Elapsed time: ([0-9.]+)s\s+SPEEDUP ([0-9.]+)", line)
        if data_match and current_opt_level:
            elapsed_time = float(data_match.group(1))
            speedup = float(data_match.group(2))
            elapsed_times[current_opt_level].append(elapsed_time)
            speedups[current_opt_level].append(speedup)

    # Calculate and print averages
    print("\nAverage Results:\n")
    for opt_level in sorted(elapsed_times.keys()):
        avg_elapsed_time = sum(elapsed_times[opt_level]) / len(elapsed_times[opt_level])
        avg_speedup = sum(speedups[opt_level]) / len(speedups[opt_level])
        print(f"  {opt_level}:")
        print(f"    Average SPEEDUP: {avg_speedup:.4f}")
        print(f"    Average Elapsed Time: {avg_elapsed_time:.4f}s")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
    else:
        main(sys.argv[1])
