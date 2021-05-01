"""Main Data Gather

The project's main file. All data generated for this project is done so in here.
No plots are produced however. The data is gathered and saved to disk to be plotted
in MATLAB.
"""

import numpy as np
import os
import sys
import threading

from functions import (
    rastrigin,
    rast_dict,
    schwefel,
    schwe_dict,
    griewangk,
    grie_dict,
    ackley,
    ackl_dict,
)

import matplotlib.pyplot as plt

from ga_experiment import GAExperiment
from ccga_experiment import CCGAExperiment


def run_ga_experiment(fitness_func, func_dict, iterations, num_experiments):
    """Runs all standard ga experiments and saves results to disk"""

    # Run Rastragin Experiments

    iteration_data = []
    sum_fitness_data = []

    sum_fitness_data = np.array([0.0] * int(iterations / 100))

    for i in range(0, num_experiments):
        GA = GAExperiment(fitness_func, func_dict, iterations)
        print("GAs generated")
        iteration_data, fitness_data = GA.run_experiment()
        sum_fitness_data += np.array(fitness_data)

        print("Experiment {} Complete".format(i))

    avr_fitness_data = sum_fitness_data / num_experiments

    return iteration_data, avr_fitness_data


def run_ccga_experiment(
    fitness_func, func_dict, iterations, num_experiments, param_num=-1
):
    """Runs all standard ga experiments and saves results to disk"""

    if param_num == -1:
        param_num = func_dict["n"]

    # Run Rastragin Experiments
    GAs = [
        CCGAExperiment(fitness_func, func_dict, iterations, param_num)
        for i in range(0, num_experiments)
    ]

    print("Finished Generating GAs")

    # Compute the number of outputs the GA will generate
    output_size = 0
    while output_size < iterations:
        output_size += param_num * 100

    output_size = int(output_size / 100)

    iteration_data = []
    sum_fitness_data = np.array([0.0] * output_size)

    for i, GA in enumerate(GAs):
        iteration_data, fitness_data = GA.run_experiment()
        sum_fitness_data += np.array(fitness_data)

        print("Experiment {} Complete".format(i))

    avr_fitness_data = sum_fitness_data / num_experiments

    return iteration_data, avr_fitness_data


def write_to_file(iter_data, avr_fitness_data, filepath):
    """Output data given to a file on path given"""

    cwd = os.getcwd()

    # Generate output lines
    out_lines = [
        "{}, {}\n".format(itr, avr_fitness)
        for itr, avr_fitness in zip(iter_data, avr_fitness_data)
    ]

    with open(cwd + "\\" + filepath, "w+") as output_file:
        output_file.writelines(out_lines)


def run_ccga_experiments():
    """Runs all ccga experiments and saves results to disk"""

    print("CCGA Not Yet Implemented!!")


if __name__ == "__main__":

    # Run GA experiments if requested
    if sys.argv[1] in ["ga", "both"]:
        print("GA Experiments")

        # Run standard GA experiments
        print("Rastrigin Experiment")
        rast_iter, rast_avr_fitness = run_ga_experiment(rastrigin, rast_dict, 100000, 5)
        print("Schwefel Experiment")
        schw_iter, schw_avr_fitness = run_ga_experiment(schwefel, schwe_dict, 100000, 5)
        print("Griewangk Experiment")
        grie_iter, grie_avr_fitness = run_ga_experiment(griewangk, grie_dict, 100000, 5)
        print("Ackley Experiment")
        ackl_iter, ackl_avr_fitness = run_ga_experiment(ackley, ackl_dict, 100000, 5)

        # Write data to disk
        output_data_path = "collected_data\\ga\\"

        write_to_file(rast_iter, rast_avr_fitness, output_data_path + "ga_rast.txt")
        write_to_file(schw_iter, schw_avr_fitness, output_data_path + "ga_schw.txt")
        write_to_file(grie_iter, grie_avr_fitness, output_data_path + "ga_grie.txt")
        write_to_file(ackl_iter, ackl_avr_fitness, output_data_path + "ga_ackl.txt")

    # Run CCGA experiments
    if sys.argv[1] in ["ccga", "both"]:
        print("CCGA Experiments")

        output_data_path = "collected_data\\ccga\\"

        # Run standard GA experiments
        print("Ackley Experiment")
        ackl_iter, ackl_avr_fitness = run_ccga_experiment(ackley, ackl_dict, 100000, 5)

        write_to_file(ackl_iter, ackl_avr_fitness, output_data_path + "ccga_ackl.txt")

        print("Rastrigin Experiment")
        rast_iter, rast_avr_fitness = run_ccga_experiment(
            rastrigin, rast_dict, 100000, 5
        )

        write_to_file(rast_iter, rast_avr_fitness, output_data_path + "ccga_rast.txt")

        print("Schwefel Experiment")
        schw_iter, schw_avr_fitness = run_ccga_experiment(
            schwefel, schwe_dict, 100000, 5
        )

        write_to_file(schw_iter, schw_avr_fitness, output_data_path + "ccga_schw.txt")

        print("Griewangk Experiment")
        grie_iter, grie_avr_fitness = run_ccga_experiment(
            griewangk, grie_dict, 100000, 5
        )

        write_to_file(grie_iter, grie_avr_fitness, output_data_path + "ccga_grie.txt")
