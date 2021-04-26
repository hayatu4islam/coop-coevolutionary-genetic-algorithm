"""Main Data Gather

The project's main file. All data generated for this project is done so in here.
No plots are produced however. The data is gathered and saved to disk to be plotted
in MATLAB.
"""

import numpy as np
import os

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


def run_ga_experiment(fitness_func, func_dict, iterations, num_experiments):
    """Runs all standard ga experiments and saves results to disk"""

    # Run Rastragin Experiments
    GAs = [GAExperiment(fitness_func, func_dict, iterations)] * num_experiments

    iteration_data = []
    sum_fitness_data = []

    sum_fitness_data = np.array([0.0] * int(iterations / 100))

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

    # Run standard GA experiments
    rast_iter, rast_avr_fitness = run_ga_experiment(rastrigin, rast_dict, 1000, 5)
    schw_iter, schw_avr_fitness = run_ga_experiment(schwefel, schwe_dict, 1000, 5)
    grie_iter, grie_avr_fitness = run_ga_experiment(griewangk, grie_dict, 1000, 5)
    ackl_iter, ackl_avr_fitness = run_ga_experiment(ackley, ackl_dict, 1000, 5)

    # Write data to disk
    output_data_path = "collected_data\\ga\\"

    write_to_file(rast_iter, rast_avr_fitness, output_data_path + "ga_rast.txt")
    write_to_file(schw_iter, schw_avr_fitness, output_data_path + "ga_schw.txt")
    write_to_file(grie_iter, grie_avr_fitness, output_data_path + "ga_grie.txt")
    write_to_file(ackl_iter, ackl_avr_fitness, output_data_path + "ga_ackl.txt")
