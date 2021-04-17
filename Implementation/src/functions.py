"""
Functions 
=========

This file implements the four functions that are to be optimised and contains
dictionaries listing parameters for the default implementations of each
function.
"""
from math import exp, sin, cos, sqrt, pi
from numpy.testing import assert_almost_equal
from scipy.interpolate import interp1d
from bitstring import BitArray


def range_map(value: float, interpolator: interp1d) -> float:
    """Maps a value from one range to another using a scipy interpolator

    :param value: The value to map
    :param interpolator: The interpolator used to perform the map
    :returns: The mapped value
    """

    return interpolator(value)


def bin_str_to_int(bin_str: BitArray) -> int:
    """Converts bin_str to an integer between 0 and (2^len(bin_str)) - 1.

    :param bin_str: Binary number represented in a string.
    """

    return bin_str.uint


rast_dict = {
    "n": 20,
    "min": -5.12,
    "max": 5.12,
    "interp": interp1d([0, (2 ** 16) - 1], [-5.12, 5.12]),
}


def rastrigin(chrom_list: list, n=rast_dict["n"], interp=rast_dict["interp"]) -> float:
    """Run the Rastrigin function for the chromasome list given.

    :param chrom_list: A list of BitArrays representing the chromosomes
        for each parameter.
    :param n: The number of parameters
    :returns: The output of the function
    """

    # Convert all chromosomes to decimal values
    dec_chroms = [bin_str_to_int(bin_str) for bin_str in chrom_list]

    # Map those numbers to the function range to get the parameters
    func_params = [range_map(chrom, interp) for chrom in dec_chroms]

    # Run the function
    # Non-sum term
    func_out = 3 * n

    # Sum term
    for x in func_params:
        func_out += (x ** 2) - 3 * cos(2 * pi * x)

    return func_out


schwe_dict = {
    "n": 10,
    "min": -500,
    "max": 500,
    "interp": interp1d([0, (2 ** 16) - 1], [-500, 500]),
    "inv_interp": interp1d([-500, 500], [0, (2 ** 16) - 1]),
}


def schwefel(chrom_list: list, n=schwe_dict["n"], interp=schwe_dict["interp"]) -> float:
    """Run the Schwefel function for the chromasome list given.

    :param chrom_list: A list of BitArrays representing the chromosomes
        for each parameter.
    :param n: The number of parameters
    :returns: The output of the function
    """

    # Convert all chromosomes to decimal values
    dec_chroms = [bin_str_to_int(bin_str) for bin_str in chrom_list]

    # Map those numbers to the function range to get the parameters
    func_params = [range_map(chrom, interp) for chrom in dec_chroms]

    # Run the function
    # Non-sum term
    func_out = 418.9829 * n

    # Sum term
    for x in func_params:
        func_out -= x * sin(sqrt(abs(x)))

    return func_out


grie_dict = {
    "n": 10,
    "min": -600,
    "max": 600,
    "interp": interp1d([0, (2 ** 16) - 1], [-600, 600]),
    "inv_interp": interp1d([-600, 600], [0, (2 ** 16) - 1]),
}


def griewangk(chrom_list: list, n=grie_dict["n"], interp=grie_dict["interp"]) -> float:
    """Run the Griewangk function for the chromasome list given.

    :param chrom_list: A list of BitArrays representing the chromosomes
        for each parameter.
    :param n: The number of parameters
    :returns: The output of the function
    """

    # Convert all chromosomes to decimal values
    dec_chroms = [bin_str_to_int(bin_str) for bin_str in chrom_list]

    # Map those numbers to the function range to get the parameters
    func_params = [range_map(chrom, interp) for chrom in dec_chroms]

    # Run the function
    sum = 0
    product = 1

    for i, x in enumerate(func_params):
        sum += (x ** 2) / 4000
        product *= cos(x / sqrt(i + 1))

    return 1 + sum - product


ackl_dict = {
    "n": 30,
    "min": -30,
    "max": 30,
    "interp": interp1d([0, (2 ** 16) - 1], [-30, 30]),
    "inv_interp": interp1d([-30, 30], [0, (2 ** 16) - 1]),
}


def ackley(chrom_list: list, n=ackl_dict["n"], interp=ackl_dict["interp"]) -> float:
    """Run the Ackley function for the chromasome list given.

    :param chrom_list: A list of BitArrays representing the chromosomes
        for each parameter.
    :param n: The number of parameters
    :returns: The output of the function
    """

    # Convert all chromosomes to decimal values
    dec_chroms = [bin_str_to_int(bin_str) for bin_str in chrom_list]

    # Map those numbers to the function range to get the parameters
    func_params = [range_map(chrom, interp) for chrom in dec_chroms]

    # Calculate sum terms
    sum1 = 0
    sum2 = 0

    dpi = 2 * pi

    for x in func_params:
        sum1 += x ** 2
        sum2 += cos(dpi * x)

    # Calculate exponential terms
    exp_term_1 = exp(-0.2 * sqrt((1 / n) * sum1))
    exp_term_2 = exp((1 / n) * sum2)

    return 20 + exp(1) - 20 * exp_term_1 - exp_term_2


if __name__ == "__main__":

    # Run Some tests
    try:
        # Test range map
        assert range_map(0, rast_dict["interp"]) == -5.12
        assert range_map((2 ** 16) - 1, rast_dict["interp"]) == 5.12

        # Test binary conversion
        bin_str_1 = BitArray("uint:16=2")
        bin_str_2 = BitArray("uint:16={}".format((2 ** 16) - 1))
        bin_str_3 = BitArray("uint:16=0")
        bin_str_4 = BitArray("uint:16=58732")

        assert bin_str_to_int(bin_str_1) == 2
        assert bin_str_to_int(bin_str_2) == (2 ** 16) - 1
        assert bin_str_to_int(bin_str_3) == 0
        assert bin_str_to_int(bin_str_4) == 58732

        # Test Rastrigin Function
        global_min_chrom = int(((2 ** 16) - 1) / 2)

        chroms_1 = [BitArray("uint:16={}".format(global_min_chrom))] * rast_dict["n"]
        assert_almost_equal(rastrigin(chroms_1), 0, 5)

        # Test Schwefel Function
        global_min_chrom = int(schwe_dict["inv_interp"](420.9687))

        chrom_1 = [BitArray("uint:16={}".format(global_min_chrom))] * schwe_dict["n"]
        assert_almost_equal(schwefel(chrom_1), 0, 3)

        # Test Griewagnk Function
        global_min_chrom = int(grie_dict["inv_interp"](0))

        chrom_1 = [BitArray("uint:16={}".format(global_min_chrom))] * grie_dict["n"]
        assert_almost_equal(griewangk(chrom_1), 0, 3)

        # Test Ackley Function
        global_min_chrom = int(ackl_dict["inv_interp"](0))

        chrom_1 = [BitArray("uint:16={}".format(global_min_chrom))] * ackl_dict["n"]
        assert_almost_equal(ackley(chrom_1), 0, 2)

    except AssertionError as e:
        print("Assertion Failed!")
        raise (e)

    else:
        print("All Assertion Tests Passed!")
