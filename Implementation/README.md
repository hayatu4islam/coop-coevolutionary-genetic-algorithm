# Project Implementation

## Directory Structure
### src
All Python files developed for this project can be found here.

###  collected_data
The results produced by the Python scripts are saved here.
Each experiment has its own subdirectory with txt files containing the results.
The subdirectory matlab_plotting_scripts contain the MATLAB scripts used to plot the results.
The subdirectory previous_results contains backups of previous runs of the software.

## Project Usage

###  Installing Prerequisites
It is assumed that you already have Python 3.8+ and MATLAB installed. 

The Python libraries needed to run this project.
Can be installed using the command 

python -m pip install -r requirements.txt

###  Running the Software
The software can be run from the top level directory using the following command:

python ./src/main_data_gather.py <algorithm> <experiment_num> <extension_num>

Where:
    <algorithm> must be replaced with ga, ccga, exga or all to specify which algorithm should be tested.
    <experiment_num> must be replaced with a number specifying the number of experiments which the results should be averaged over.
    <extension_num> can be replaced by either 1 or 2 to specify EXGA_1 or EXGA_2.

