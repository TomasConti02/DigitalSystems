import re
import sys

# this script reads the log file and, for each opt level, calculates the average speedup

def extract_speedups_from_log(file_name):
    speedups = {
        "-O0": [],
        "-O1": [],
        "-O2": [],
        "-O3": []
    }

    with open(file_name, 'r') as file:
        content = file.read()
        
        speedup_pattern = re.compile(r"SPEEDUP\s([0-9\.]+)")
        matches = speedup_pattern.findall(content)
        
        optimization_levels = ['-O0', '-O1', '-O2', '-O3']
        current_level = None
        
        for line in content.splitlines():
            if any(level in line for level in optimization_levels):
                for level in optimization_levels:
                    if level in line:
                        current_level = level
                        break
            if "SPEEDUP" in line and current_level:
                match = re.search(r"SPEEDUP\s([0-9\.]+)", line)
                if match:
                    speedups[current_level].append(float(match.group(1)))
    
    return speedups


def calculate_average_speedup(speedups, file_name):
    print(f"\nAverage SPEEDUP for {file_name}:")
    for level, speedup_list in speedups.items():
        if speedup_list:
            # compute average
            first_10_speedups = speedup_list[:10]
            average_speedup = sum(first_10_speedups) / len(first_10_speedups)
            print(f"  {level}: {average_speedup:.4f}")
        else:
            print(f"  {level}: No data")


def main():
    if len(sys.argv) != 2:
        print("Usage: python speedup_analysis.py <percorso_file_log>")
        sys.exit(1)

    log_file = sys.argv[1]
    
    speedups = extract_speedups_from_log(log_file)
    calculate_average_speedup(speedups, log_file)


if __name__ == "__main__":
    main()
