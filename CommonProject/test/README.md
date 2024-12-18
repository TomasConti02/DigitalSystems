# Test folder #
This folder contains:
- **test.sh**: this script runs a `.cpp` file 10 times for each optimization level (`O0`, `O1`, `O2`, `O3`) and prints the output in a `.txt` file, and executes `speedup_analysis.py`.
- **speedup_analysis.py**: this python file reads the `.txt` file and, for each optimization level, calculates the average speedup and average elapsed time.

## How to run ##
On CLI:
```
  ./test.sh <src_file.cpp> [compile_options]
```
The `.txt` file is saved in this folder. The `.py` file is already executed by the `.sh` file, but you can execute it separately by copying on CLI: 
```
  python3 speedup_analysis.py <file.txt>
```
