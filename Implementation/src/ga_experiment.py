""" GA Experiment

This file implements a class which is used to complete all standard GA experiments.
The CCGA Experiment class is based on this.
"""

from bitstring import BitArray
from random import randint
from individual import Individual

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


class GAExperiment:
    def __init__(
        self, _fitness_func, _func_dict, _func_param_num=-1, _evaluations=100000
    ):

        self.fitness_func = _fitness_func
        """The function used to evaluate the population, taken from function.py"""

        self.func_dict = _func_dict
        """A dict containing parameters for the chosen fitness function"""

        self.pop_width = None
        """How many bits contained within each pop. 
        Determined from the number of function parameters * 16 bits.
        """

        self.param_num = None
        """The number of function parameters stored in the geonome"""

        if _func_param_num == -1:
            self.pop_width = 16 * self.func_dict.n
            self.param_num = self.func_dict.n
        else:
            self.pop_width = 16 * _func_param_num
            self.param_num = _func_param_num

        self.pop_size = 100
        """Number of individuals in a population"""

        self.evaluations = _evaluations
        """The number of function evaluations completed during the experiment"""

        self.generation = 0
        """The current generation of population"""

        self.pop = self.get_starting_pop()
        """The population of BitArrays that are evolved during this experiment"""

        # Update the fitnesses of the pops to prepare for the start of the algorithm
        self.update_fitnesses()

    def get_starting_pop(self):
        """Generate a starter population of random individuals"""

        # Define a BitArray format string
        def_format = "uint:{}={}".format(self.pop_width, "{}")

        # Generate an array of definition strings with random starting numbers
        def_strings = [
            def_format.format(randint(0, (2 ** (self.pop_width)) - 1))
            for i in range(self.pop_size)
        ]

        # Generate bitarrays/genomes from those definition strings
        pop_bitarrays = [BitArray(def_str) for def_str in def_strings]

        # Assign those bitarrays/genomes to individuals in a population
        population = [Individual(arr) for arr in pop_bitarrays]

        return population

    def update_fitnesses(self):
        """Update the fitness values for every individual in self.pop.

        Splits the pop into parameters and runs them through the function given in
        self.fitness_func.
        """

        # Iterate accross all individuals in the population
        for i, ind in enumerate(self.pop):

            # Split geonome into list of parameters
            bin_param_list = self.get_bin_params(ind)

            # Apply those parameters to the given fitness function
            self.pop[i].fitness = self.fitness_func(bin_param_list, self.param_num)

    def get_bin_params(self, individual: BitArray):
        """Split the individual given into 16 bit chunks for the parameters

        :param individual: The BitArray that defines the individual.
        :returns: A list 16-bit BitArrays, each implementing a parameter.
        """

        # Define a list of empty bit arrays to be overwritten with params
        bin_params = [BitArray("uint:16=0")] * self.param_num

        # Assign slices of individual to bin_params
        for i, ind_index in enumerate(range(0, self.param_num * 16, 16)):
            bin_params[i] = individual.bit_arr[ind_index : ind_index + 16]

        return bin_params

    # def run_experiment(self):
    #     """Run the experiment up to a number of function evaluations given in
    #     evaluations.

    #     :param iterations: The number of function evaluations
    #     :returns:
    #     """


if __name__ == "__main__":
    # Run Some tests
    try:
        # Generate new experiment, check pop and fitness are generated right
        rast_ga_exp = GAExperiment(rastrigin, rast_dict, 5)

        # Test get_bin_params
        ind0 = rast_ga_exp.pop[0]
        bin_params0 = rast_ga_exp.get_bin_params(ind0)
        assert ind0.bit_arr[0:16] == bin_params0[0]
        assert ind0.bit_arr[32:48] == bin_params0[2]

        # Test get update_fitness
        assert ind0.fitness == rastrigin(bin_params0, 5)

    except AssertionError as e:
        print("Assertion Failed!")
        raise (e)

    else:
        print("All Assertion Tests Passed!")
