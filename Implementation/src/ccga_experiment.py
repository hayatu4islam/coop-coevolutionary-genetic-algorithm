""" GA Experiment

This file implements a class which is used to complete all standard GA experiments.
The CCGA Experiment class is based on this.
"""

from bitstring import BitArray
import numpy as np
import random
import matplotlib.pyplot as plt

from individual import Individual


class CCGAExperiment:
    def __init__(
        self, _fitness_func, _func_dict, _evaluations=100000, _func_param_num=-1
    ):

        self.fitness_func = _fitness_func
        """The function used to evaluate the population, taken from function.py"""

        self.func_dict = _func_dict
        """A dict containing parameters for the chosen fitness function"""

        self.pop_width = 16
        """How many bits contained within each pop. 16 bits for all population"""

        self.param_num = None
        """The number of function parameters stored in the geonome"""

        if _func_param_num == -1:
            self.param_num = self.func_dict["n"]
        else:
            self.param_num = _func_param_num

        self.pop_num = self.param_num
        """The number of populations used to store all parameters.
        Defined by param_num"""

        self.pop_size = 100
        """Number of individuals in a population"""

        self.evaluations = _evaluations
        """The number of function evaluations that should be completed during the experiment"""

        self.evaluations_completed = 0
        """The number of evaluations completed so far"""

        self.generation = 0
        """The current generation of population"""

        self.pops = [self.get_starting_pop() for i in range(0, self.pop_num)]
        """A set of populations, each defining one function parameter"""

        self.best_ind_idxs = [0] * self.param_num
        """The indexes the best individuals in each population"""

        self.pops_max_fitness = [10e50] * self.param_num
        """The fitness of the values stored in best_ind_idxs, used for computing
        global fitness each generation"""

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

        # Update the fitnesses of the pops to prepare for the start of the algorithm
        self.set_starting_fitness()

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

    def set_starting_fitness(self):
        """Define the fitness of each individual by selecting random individuals from
        other populations and running them through the fitness function.
        """

        # Iterate through each member of each population
        for pop_num, pop in enumerate(self.pops):
            for ind_num, ind in enumerate(pop):

                # Get a random member from each other population
                # Slot the current one into it
                random_param_set = []

                for i in range(0, self.pop_num):

                    # Add ind under test in the correct position
                    if i == pop_num:
                        random_param_set.append(ind.bit_arr)

                    # Get random ind from the current pop, place it in set
                    else:
                        random_idx = random.randint(0, self.pop_size - 1)
                        rand_pop = self.pops[pop_num][random_idx]
                        random_param_set.append(rand_pop.bit_arr)

                ind.fitness = self.fitness_func(random_param_set, self.param_num)

        # Update the number of function evaluations completed
        self.evaluations_completed += self.pop_num * self.pop_size

        # Find the best ind in each population and save its index and fitness
        self.update_best_inds()

    def update_best_inds(self):
        """Find the index of the best individual in each pop and save it and the fitness
        to global arrays for calculating the fitness of other pops.
        """

        for pop_num, pop in enumerate(self.pops):
            best_ind = min(pop, key=lambda ind: ind.fitness)

            self.best_ind_idxs[pop_num] = pop.index(best_ind)
            self.pops_max_fitness[pop_num] = best_ind.fitness

        # Add current best fitness to data capture
        self.evaluation_data.append(self.evaluations_completed)
        self.fitness_data.append(min(self.pops_max_fitness))

    def update_fitnesses(self):
        """Update the fitness values for every individual in each pop in self.pops

        A list of the best performing pops from last generation is assembled.
        Each ind in each pop is substituted into the relevant position and the
        new set is evaluated.
        This allows each ind to be compared against the best from each other
        pop.
        """

        # Assemble a "Greatest Hits" list of the best ind bit_arrays from each pop
        best_inds_original = [
            self.pops[p_num][i_num].bit_arr.copy()
            for p_num, i_num in enumerate(self.best_ind_idxs)
        ]

        # Iterate accross all individuals in each population
        for pop_num, pop in enumerate(self.pops):

            # Create a copy that can be overwritten
            best_inds_copy = best_inds_original.copy()

            # Test every ind in this pop against the best from the previous generation
            for ind in pop:
                best_inds_copy[pop_num] = ind.bit_arr
                ind.fitness = self.fitness_func(best_inds_copy, self.param_num)

        # Update the list of best performers with values from this generation
        self.update_best_inds()

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
            # self.pop = self.breed_new_population()

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
                ind1 = self.select_new_ind(roulette_wheel)
                ind2 = self.select_new_ind(roulette_wheel)
                new_ind = self.two_point_crossover(ind1, ind2)

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

        # For every bit position in ind
        for i in range(0, len(ind.bit_arr)):

            # If mutation chance met, invert bit at that position
            if random.random() < (1 / len(ind.bit_arr)):
                ind.bit_arr.invert(i)

        return ind


from functions import (
    rastrigin,
    rast_dict,
)

if __name__ == "__main__":
    # Run Some tests
    try:

        # Test the start up of the experiment
        rast_ccga = CCGAExperiment(rastrigin, rast_dict, 2000, 5)

        # Test the populations have been defined correctly
        assert len(rast_ccga.pops) == 5
        assert len(rast_ccga.pops[0]) == 100

        # Test the best pops have been found correctly
        pop3 = rast_ccga.pops[3]
        pop3_best = min(pop3, key=lambda ind: ind.fitness)

        assert pop3.index(pop3_best) == rast_ccga.best_ind_idxs[3]
        assert pop3_best.fitness == rast_ccga.pops_max_fitness[3]

        # Test update fitness
        rast_ccga.update_fitnesses()

    except AssertionError as e:
        print("Assertion Failed!")
        raise (e)

    else:
        print("All Assertion Tests Passed!")
