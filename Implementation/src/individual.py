"""Individual Class

Simple class holding just a BitArray and the corresponding fitness for that
array. If Python had structs I'd use one of those.
"""

from bitstring import BitArray


class Individual:
    def __init__(self, bit_arr: BitArray):

        self._bit_arr = bit_arr
        """The BitArray defining this individuals chromasome"""

        self._fitness = None
        """The fitness of this individual. Computed externally"""

    @property
    def bit_arr(self):
        return self._bit_arr

    @bit_arr.setter
    def bit_arr(self, bit_arr):
        self._bit_arr = bit_arr

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, fitness):
        self._fitness = fitness


if __name__ == "__main__":
    try:
        bit_arr1 = BitArray("uint:16=4500")
        bit_arr2 = BitArray("uint:16=534")

        fitness1 = 45
        fitness2 = 3453

        ind = Individual(bit_arr1)
        assert ind.bit_arr == bit_arr1

        ind = Individual(bit_arr2)
        assert ind.bit_arr == bit_arr2

        ind = Individual(fitness1)
        assert ind.bit_arr == fitness1

        ind = Individual(fitness2)
        assert ind.bit_arr == fitness2

    except AssertionError as e:
        print("Assertion Failed!")
        raise (e)

    else:
        print("All Assertion Tests Passed!")
