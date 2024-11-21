# Test folder #
This folder contains:
- **test.sh**: this script runs a c++ file 10 times for each optimization level (`O0`, `O1`, `O2`, `O3`) and prints the output in a txt file, and executes `speedup_analyzer.py`.
- **speedup_analyzer.py**: this python file reads the txt file and, for each optimization level, calculates the average speedup.
## How to run ##
```
  ./test.sh <src_file.cpp>
```
The txt file is saved in this folder. The py file is already executed by the `.sh` file, but you can execute it separately with `python3 speedup_analyzer.py <file.txt>`.
