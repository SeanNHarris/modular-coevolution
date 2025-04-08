#  Copyright 2025 BONSAI Lab at Auburn University
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

__author__ = 'Sean N. Harris'
__copyright__ = 'Copyright 2025, BONSAI Lab at Auburn University'
__license__ = 'Apache-2.0'

import random
from typing import TypedDict, Iterable, cast, NotRequired
# from typing_extensions import NotRequired  # TODO: Upgrade to python 3.11 to remove dependency

from modularcoevolution.genotypes.diversity.alternatediversity import genetic_algorithm_diversity
from modularcoevolution.genotypes.basegenotype import BaseGenotype

try:
    import numba
    import numpy
    numba_available = True
except ModuleNotFoundError:
    numba_available = False


class _Parameters(TypedDict):
    values: NotRequired[Iterable[int]]
    alleles: NotRequired[Iterable[int]]


class PermutationGenotype(BaseGenotype):
    genes: list[int]

    def __init__(self, parameters: _Parameters):
        super().__init__()
        if "values" in parameters:
            self.genes = list(parameters["values"])
        elif "alleles" in parameters:
            alleles = list(parameters["alleles"])
            if len(alleles) > len(set(alleles)):
                raise ValueError("Provided alleles should not have duplicates.")
            if numba_available:
                allele_array = numpy.array(alleles)
                self.genes = list(self.random_genotype_numba(allele_array))
            else:
                self.genes = alleles
                random.shuffle(self.genes)
        else:
            raise ValueError("If \"values\" is not provided, \"alleles\" must be.")


    def mutate(self) -> None:
        # Cycle mutation
        cycle_size = random.randrange(2, len(self.genes) // 2)
        indices = random.sample(range(len(self.genes)), cycle_size)  # TODO: Mutation size parameter

        new_values = list()
        for index in indices:
            new_values.append(self.genes[index])
        for i, index in enumerate(indices):
            self.genes[index] = new_values[i - 1]  # Negative indices wrap

    def recombine(self, donor: "PermutationGenotype") -> None:
        # Order 1 Crossover
        retain_start = random.randrange(len(self.genes))
        retain_end = random.randrange(retain_start, len(self.genes)) + 1  # Exclusive

        index = 0
        for gene in donor.genes:
            if gene not in self.genes[retain_start:retain_end]:
                self.genes[index] = gene
                index += 1
            if retain_start <= index < retain_end:
                index = retain_end

    def clone(self, copy_objectives: bool = False) -> "PermutationGenotype":
        parameters = cast(_Parameters, self.get_raw_genotype())
        cloned_genotype = type(self)(parameters)  # TODO: Had to change this to type(self) for inheritance, make sure this is consistent elsewhere. Maybe this function can be partly moved to the base class?
        if copy_objectives:
            for objective in self.objectives:
                cloned_genotype.objectives[objective] = self.objectives[objective]
                cloned_genotype.objective_statistics[objective] = self.objective_statistics[objective]
                cloned_genotype.objectives_counter[objective] = self.objectives_counter[objective]
                cloned_genotype.past_objectives[objective] = self.past_objectives[objective]
            cloned_genotype.evaluated = True
            cloned_genotype.fitness = self.fitness
        cloned_genotype.parent_ids.append(self.id)
        cloned_genotype.creation_method = "Cloning"
        return cloned_genotype

    def __str__(self):
        return str(self.genes)

    def __hash__(self):
        return hash(tuple(self.genes))

    def get_raw_genotype(self) -> _Parameters:
        return {"values": self.genes}

    def diversity_function(self, population, reference=None, samples=None):
        return genetic_algorithm_diversity(population, reference, samples)  # TODO: Permutation-specific diversity?

    @staticmethod
    @numba.njit
    def random_genotype_numba(alleles: numpy.ndarray) -> numpy.ndarray:
        genes = alleles.copy()
        random.shuffle(genes)
        return genes
