""" 
GA Experiment
=============

This file implements a class which is used to complete all standard GA experiments.
This includes the extension.
The CCGA Experiment class is based on this.
"""

import sys
import cProfile

from bitstring import BitArray
import numpy as np
import random
import matplotlib.pyplot as plt

from individual import Individual


class GAExperiment:
    def __init__(
        self,
        _fitness_func,
        _func_dict,
        _evaluations=100000,
        _func_param_num=-1,
        _extension=0,
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
            self.pop_width = 16 * self.func_dict["n"]
            self.param_num = self.func_dict["n"]
        else:
            self.pop_width = 16 * _func_param_num
            self.param_num = _func_param_num

        self.pop_size = 100
        """Number of individuals in a population"""

        self.evaluations = _evaluations
        """The number of function evaluations that should be completed during the experiment"""

        self.evaluations_completed = 0
        """The number of evaluations completed so far"""

        self.generation = 0
        """The current generation of population"""

        self.pop = self.get_starting_pop()
        """The population of BitArrays that are evolved during this experiment"""

        self.scaling_window = [0] * 5
        """A record of the worst fitness of the last 5 generations. Used as the 
        baseline for comparing all other fitnesses.
        """

        self.fitness_data = []
        """The data to be exported. The lowest fitness of each generation is added
        to this array."""

        self.evaluation_data = []
        """An array containing the function evaluation number that each corresponding
        element in fitness_data was collected."""

        #### Extension ####

        self.crossover = None
        """A placeholder for the function which performs crossover.
        It is selected based on the value of the _extension argument to be either
        standart two_point_crossover or the extension function n_chunk_crossover.
        """

        if _extension == 0:
            self.crossover = self.two_point_crossover
        elif _extension == 1:
            self.crossover = self.n_chunk_crossover

        self.extension = _extension
        """Determins which extension to use.
        If 0: normal GA.
        If 1: n_chunk crossover
        If 2: n individual crossover""" 

        #### End Extension ####

        # Update the fitnesses of the pops to prepare for the start of the algorithm
        self.update_fitnesses()

    def get_starting_pop(self):
        """Generate a starter population of random individuals"""

        # Define a BitArray format string
        def_format = "uint:{}={}".format(self.pop_width, "{}")

        # Generate an array of definition strings with random starting numbers
        def_strings = [
            def_format.format(random.randint(0, (2 ** (self.pop_width)) - 1))
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
            ind.fitness = self.fitness_func(bin_param_list, self.param_num)

        # Update the track of the number of fitness evaluations completed
        self.evaluations_completed += self.pop_size

        # Add current best fitness to data capture
        self.evaluation_data.append(self.evaluations_completed)
        self.fitness_data.append(min(self.pop, key=lambda pop: pop.fitness).fitness)

    def get_bin_params(self, individual: Individual):
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

    def run_experiment(self):
        """Run the experiment up to a number of function evaluations given in
        evaluations.

        :param iterations: The number of function evaluations
        :returns:
        """

        while self.evaluations_completed < self.evaluations:

            self.generation += 1

            # Update scaling window from previous fitness update
            self.scaling_window.pop()
            self.scaling_window.insert(
                0, max(self.pop, key=lambda pop: pop.fitness).fitness
            )
            self.scaling_factor = max(self.scaling_window)

            # Generate new generation and replace the old one
            self.pop = self.breed_new_population()

            # Update fitness values
            self.update_fitnesses()

        return self.evaluation_data, self.fitness_data

    def breed_new_population(self):
        """Perform proportional fitness selection to generate the next generation.

        Apply crossover to get a new individual and apply mutation to that individual.
        Leave the most fit individual from the previous generation in the new generation.
        """

        # Declare a new array for the new population with space for the previous best
        new_population = []

        # Add the best pop from the previous generation
        new_population.append(min(self.pop, key=lambda ind: ind.fitness))
        # Generate the roulette wheel
        roulette_wheel = self.get_roulette_wheel()

        # Generate new individuals for the rest of the population
        for i in range(0, self.pop_size - 1):

            # Crossover chance of 0.6
            if random.random() < 0.6:

                # Perform normal 2 parent crossover
                if self.extension in [0,1]:
                    ind1 = self.select_new_ind(roulette_wheel)
                    ind2 = self.select_new_ind(roulette_wheel)
                    new_ind = self.crossover(ind1, ind2)
                
                # Perform n parent crossover
                else:
                    parent_list = [self.select_new_ind(roulette_wheel) for i in range(0, self.param_num)]
                    new_ind = self.n_ind_crossover(parent_list)

            else:
                new_ind = self.select_new_ind(roulette_wheel)

            new_population.append(self.mutate(new_ind))

        return new_population.copy()

    def get_roulette_wheel(self):
        """Generate the proportional fitness roulette wheel for the current population

        Perform a cumulative sum of the population fitnesses and divide the result
        through by the max population. This will leave an array of floats between
        0 and 1. By seeing where a random number in that range falls individuals
        can be selected proportionally to their fitness.

        Since a lower fitness value is better we use the scaling window in reverse,
        subtracting the pop fitness from it. This makes smaller pops take up
        bigger chunks of the wheel.
        """

        # Calculate the cumulative sum of all the population fitnesses
        wheel = np.cumsum([self.scaling_factor - pop.fitness for pop in self.pop])

        return wheel.tolist()

    def select_new_ind(self, roulette_wheel: list):
        """Select a new pop using proportional selection/roulette wheel.

        Generate a random number between 0.0 and 1.0 and find the index this
        falls within on the roulette wheel. Return the pop at this index.

        This value is found by finding all values greater than the selected
        value and finding the smallest of those. The index of that value is
        then found in the roulette wheel.
        """

        # Get random float between 0.0 and 1.0
        rand = random.random() * roulette_wheel[-1]

        # Get the index of that value in the wheel
        ind_idx = roulette_wheel.index(min([i for i in roulette_wheel if i >= rand]))

        # Create a new individual with the same genes to stop python assigning every
        # member of the population a link back to a single shared BitArray as their
        # genetic material.
        return Individual(self.pop[ind_idx].bit_arr.copy())

    def two_point_crossover(self, ind1: Individual, ind2: Individual) -> Individual:
        """Perform two point crossover on two individuals and return the child

        Two random numbers between 0 and pop_width are generated.
        ind1 is copied to the child. The slice of ind2 between the two points is
        spliced into the child.
        """

        cross_start = random.randint(0, self.pop_width)
        cross_end = random.randint(cross_start, self.pop_width)

        # Clone first individual
        child = Individual(ind1.bit_arr)

        # Splice in part of the second individual
        child.bit_arr[cross_start:cross_end] = ind2.bit_arr[
            cross_start:cross_end
        ].copy()

        return child

    def mutate(self, ind: Individual):
        """Mutate bits in ind with a chance of 1/len(ind)"""

        # Get mutation chance from array length
        mut_chance = 1 / len(ind.bit_arr)

        # For every bit position in ind
        for i in range(0, len(ind.bit_arr)):

            # If mutation chance met, invert bit at that position
            if random.random() < (mut_chance):
                ind.bit_arr.invert(i)

        return ind

    ####EXTENSION####

    def n_chunk_crossover(self, ind1: Individual, ind2: Individual) -> Individual:
        """Create a new individual by taking parameter chunks from each parent at
        random.

        A combination of n-point crossover and uniform crossover.
        It has property of randomly sampling each parent to make the child that
        uniform crossover uses but, it respects the boundaries of the parameters.
        """

        # First create a blank bitarray to store the new child in.
        new_bit_arr = BitArray("uint:{}=0".format(self.pop_width))

        # Get the bitarrays of the parents
        ind1_bit_arr = ind1.bit_arr
        ind2_bit_arr = ind2.bit_arr

        # Generate an N value array of random numbers to decide which
        # chunks come from which parent
        chunk_rands = np.random.rand(self.param_num).tolist()

        # Assign each chunk in the new bitarray with the equivalent chunk
        # from a randomly selected parent.
        for i, ind_idx in enumerate(range(0, self.param_num * 16, 16)):

            # Define the chunk start and ends
            start = ind_idx
            end = ind_idx + 16

            if chunk_rands[i] < 0.5:
                new_bit_arr[start:end] = ind1_bit_arr[start:end]

            else:
                new_bit_arr[start:end] = ind2_bit_arr[start:end]

        return Individual(new_bit_arr)

    def n_ind_crossover(self, ind_list) -> Individual:
        """Create a new individual by taking parameter chunks from a number of
        parents equal to the number of parameters in being evaluated.

        The is an extension of the n_chunk_crossover designed above. 
        It is designed to mimic the use of multiple populations as seen in 
        the CCGA as closely as possible.
        """

        # First create a blank bitarray to store the new child in.
        new_bit_arr = BitArray("uint:{}=0".format(self.pop_width))

        # Get the bitarrays of the parents
        bit_arrs = [ind.bit_arr for ind in ind_list]

        # Assign each chunk in the new bitarray with the equivalent chunk
        # from a randomly selected parent.
        for i, ind_idx in enumerate(range(0, self.param_num * 16, 16)):

            # Define the chunk start and ends
            start = ind_idx
            end = ind_idx + 16

            # Select a parent at random and take their parameter
            rand_ind_bit_arr = random.choice(bit_arrs)
            new_bit_arr[start:end] = rand_ind_bit_arr[start:end]

        return Individual(new_bit_arr)


from functions import (
    rastrigin,
    rast_dict,
)


def main():
    # Run Some tests
    try:
        # Generate new experiment, check pop and fitness are generated right
        rast_ga_exp = GAExperiment(rastrigin, rast_dict, 10, 5)

        # Test get_bin_params
        ind0 = rast_ga_exp.pop[0]
        bin_params0 = rast_ga_exp.get_bin_params(ind0)
        assert ind0.bit_arr[0:16] == bin_params0[0]
        assert ind0.bit_arr[32:48] == bin_params0[2]

        # Test get update_fitness
        assert ind0.fitness == rastrigin(bin_params0, 5)

        # Test two point crossover
        rast_ga_exp_small = GAExperiment(rastrigin, rast_dict, 10, 2)
        print(
            rast_ga_exp_small.pop[0].bit_arr.bin,
        )
        print(
            rast_ga_exp_small.pop[1].bit_arr.bin,
        )
        print(
            rast_ga_exp_small.two_point_crossover(
                rast_ga_exp_small.pop[0], rast_ga_exp_small.pop[1]
            ).bit_arr.bin
        )

        # Test Mutate
        rast_ga_exp_big = GAExperiment(rastrigin, rast_dict, 10, 20)
        print(rast_ga_exp_big.pop[0].bit_arr)
        print(rast_ga_exp_big.mutate(rast_ga_exp_big.pop[0]).bit_arr)

        # Test extension two point crossover
        rast_ga_ext = GAExperiment(rastrigin, rast_dict, 10, 6)
        ind1 = rast_ga_ext.pop[0]
        ind2 = rast_ga_ext.pop[1]
        child = rast_ga_ext.n_chunk_crossover(ind1, ind2)

        print("N Chunk Cross Over Test")
        print(ind1.bit_arr.hex)
        print(ind2.bit_arr.hex)
        print(child.bit_arr.hex)

        # Test n_ind_crossover
        print("N Ind Crossover test")
        rast_ga_ext = GAExperiment(rastrigin, rast_dict, 10, 4)
        pops = rast_ga_ext.pop[0:3]

        [print(pop.bit_arr.hex) for pop in pops]
        print(rast_ga_ext.n_ind_crossover(pops).bit_arr.hex)

        # Run Rastigin test
        print("Standard Rastrigin Test")

        rast_ga_exp = GAExperiment(rastrigin, rast_dict, 10000)
        test_pop = rast_ga_exp.pop.copy()

        rast_evalus, rast_fitness = rast_ga_exp.run_experiment()

        fig, standard_ax = plt.subplots()
        standard_ax.plot(rast_evalus, rast_fitness)

        pop = rast_ga_exp.pop

        # Run n_chunk_crossover test
        print("n_chunk Rastrigin Test")

        rast_ga_ext = GAExperiment(rastrigin, rast_dict, 10000, _extension=1)

        ext_evalus, ext_fitness = rast_ga_ext.run_experiment()

        standard_ax.plot(ext_evalus, ext_fitness)

        # Run n_ind_crossover test
        print("n_ind Rastrigin Test")

        rast_ga_ext_n_ind = GAExperiment(rastrigin, rast_dict, 10000, _extension=2)
        n_ind_evalus, n_ind_fitness = rast_ga_ext_n_ind.run_experiment()

        standard_ax.plot(n_ind_evalus, n_ind_fitness)

        plt.show()

    except AssertionError as e:
        print("Assertion Failed!")
        raise (e)

    else:
        print("All Assertion Tests Passed!")


def profile():

    # Define experiment
    rast_ga_exp = GAExperiment(rastrigin, rast_dict, 2000)

    # Profile experiment run
    print("Begin Profiling...")
    cProfile.runctx("rast_ga_exp.run_experiment()", globals(), locals())


if __name__ == "__main__":

    if len(sys.argv) > 1:
        if sys.argv[1] == "profile":
            profile()

    else:
        main()
