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

import math

from modularcoevolution.genotypes.diversity.alternatediversity import genetic_algorithm_diversity
from modularcoevolution.genotypes.basegenotype import BaseGenotype

from typing import TypedDict, Sequence, Union

import random

# TODO: These should probably be parameters somewhere? Either per individual (may be inconsistent), per method (may be tedious), or per class (might be problematic for multiple populations)
MIN_VALUE_DEFAULT = 0
MAX_VALUE_DEFAULT = 10
GENE_MUTATION_RATE_DEFAULT = 0.33
GENE_MUTATION_SIZE_DEFAULT = 0.1


class LinearGenotypeParameters(TypedDict, total=False):
    """
    A `TypedDict` of parameters for a `LinearGenotype`.
    """

    values: list[float]
    """Optional, explicitly set the gene values."""

    length: int
    """Length of the genotype."""
    min_value: Union[float, Sequence[float]]
    """The min allowed value of each gene.
    Passing a ``float`` uses that setting for all genes.
    Passing a sequence will apply the settings to genes in order.
    If the sequence is shorter than :attr:`length`, it will loop back to the beginning of the sequence."""
    max_value: Union[float, Sequence[float]]
    """The max allowed value of each gene, exclusive.
    Passing a ``float`` uses that setting for all genes.
    Passing a sequence will apply the settings to genes in order.
    If the sequence is shorter than :attr:`length`, it will loop back to the beginning of the sequence."""
    loop_genes: Union[bool, Sequence[bool]]
    """Specifies genes where :attr:`max_value` loops to :attr:`min_value` such as angles.
    Passing a ``bool`` uses that setting for all genes.
    Passing a sequence will apply the settings to genes in order.
    If the sequence is shorter than :attr:`length`, it will loop back to the beginning of the sequence.
    Defaults to ``False``."""
    round_genes: Union[bool, Sequence[bool]]
    """Specifies genes that should only take integer values. These genes will be rounded after mutation.
    Passing a ``bool`` uses that setting for all genes.
    Passing a sequence will apply the settings to genes in order.
    If the sequence is shorter than :attr:`length`, it will loop back to the beginning of the sequence.
    Defaults to ``False``."""

    gene_mutation_rate: float
    """The probability of mutation per-gene."""
    gene_mutation_standard_deviation: float
    """The scale of the mutation, as a factor of the gene's range.
    This should be in the range of 0 to 1, with 0.1 as the default."""


class LinearGenotype(BaseGenotype):
    """
    A genotype made from a sequence of numerical values.
    """
    genes: list[float]
    """The values of the genotype"""

    length: int
    """The number of genes in the genotype."""
    min_value: list[float]
    """The min allowed value of each gene."""
    max_value: list[float]
    """The max allowed value of each gene, exclusive."""
    loop_genes: list[bool]
    """Specifies genes where :attr:`max_value` loops to :attr:`min_value` such as angles."""
    round_genes: list[bool]
    """Specifies genes that should only take integer values. These genes will be rounded after mutation."""

    gene_mutation_rate: float
    """The probability of mutation per-gene."""
    gene_mutation_standard_deviation: float
    """The scale of the mutation, as a factor of the gene's range."""

    def __init__(self, parameters: LinearGenotypeParameters):
        super().__init__()
        if 'gene_mutation_rate' in parameters:
            self.gene_mutation_rate = parameters['gene_mutation_rate']
        else:
            self.gene_mutation_rate = GENE_MUTATION_RATE_DEFAULT

        if 'gene_mutation_standard_deviation' in parameters:
            self.gene_mutation_standard_deviation = parameters['gene_mutation_standard_deviation']
        else:
            self.gene_mutation_standard_deviation = GENE_MUTATION_SIZE_DEFAULT

        must_generate = False
        self.genes = None
        if 'values' in parameters:
            self.genes = parameters['values'].copy()
            self.length = len(self.genes)
        elif 'length' in parameters:
            must_generate = True
            self.genes = list()
            self.length = parameters['length']
        if self.genes is None:
            raise TypeError('If \'values\' is not provided, a \'length\' must be.')

        # TODO: Clean this up, since these are all basically the same per parameter
        if 'min_value' in parameters:
            if isinstance(parameters['min_value'], Sequence):
                self.min_value = list()
                for i in range(self.length):
                    self.min_value.append(parameters['min_value'][i % len(parameters['min_value'])])
            else:
                self.min_value = [parameters['min_value'] for _ in range(self.length)]
        else:
            self.min_value = [MIN_VALUE_DEFAULT for _ in range(self.length)]

        if 'max_value' in parameters:
            if isinstance(parameters['max_value'], Sequence):
                self.max_value = list()
                for i in range(self.length):
                    self.max_value.append(parameters['max_value'][i % len(parameters['max_value'])])
            else:
                self.max_value = [parameters['max_value'] for _ in range(self.length)]
        else:
            self.max_value = [MAX_VALUE_DEFAULT for _ in range(self.length)]

        if 'loop_genes' in parameters:
            if isinstance(parameters['loop_genes'], Sequence):
                self.loop_genes = list()
                for i in range(self.length):
                    self.loop_genes.append(parameters['loop_genes'][i % len(parameters['loop_genes'])])
            else:
                self.loop_genes = [parameters['loop_genes'] for _ in range(self.length)]
        else:
            self.loop_genes = [False for _ in range(self.length)]

        if 'round_genes' in parameters:
            if isinstance(parameters['round_genes'], Sequence):
                self.round_genes = list()
                for i in range(self.length):
                    self.round_genes.append(parameters['round_genes'][i % len(parameters['round_genes'])])
            else:
                self.round_genes = [parameters['round_genes'] for _ in range(self.length)]
        else:
            self.round_genes = [False for _ in range(self.length)]

        if must_generate:
            for index in range(parameters['length']):
                self.genes.append(self.random_gene(index))

    def random_gene(self, index: int) -> float:
        gene = random.random() * (self.max_value[index] - self.min_value[index]) + self.min_value[index]
        if self.round_genes[index]:
            gene = math.floor(gene)
        return gene

    def mutate(self) -> None:
        for i in range(len(self.genes)):
            if random.random() < self.gene_mutation_rate:
                # Mutation size is a factor of the range of the gene!
                self.genes[i] += random.gauss(0, self.gene_mutation_standard_deviation) * (self.max_value[i] - self.min_value[i])
                if self.loop_genes[i]:
                    # Modulus in python works correctly for negatives.
                    self.genes[i] = (self.genes[i] - self.min_value[i]) % (self.max_value[i] - self.min_value[i]) + self.min_value[i]
                else:
                    self._reflect_gene(i)

                if self.round_genes[i]:
                    self.genes[i] = round(self.genes[i])
                    if self.loop_genes[i] and self.genes[i] == self.max_value[i]:
                        # The above looping logic will allow genes to round to the max value, so handle this edge case.
                        self.genes[i] = self.min_value[i]
                # TODO: Warn if `round_genes` conflicts with non-integer min or max

        self.creation_method = 'Mutation'

    def _reflect_gene(self, gene_index: int) -> None:
        """
        If the given gene is outside [min, max), reflect it at the boundary.

        Args:
            gene_index: The index of the gene to handle.
        """
        if self.round_genes[gene_index]:
            # For integer genes, the max value is exclusive, so we need to move it.
            max_reflect = self.max_value[gene_index] - 1
        else:
            max_reflect = self.max_value[gene_index]

        gene_range = max_reflect - self.min_value[gene_index]

        # Repeat twice in case the gene is WAY outside the range and reflects to the other bound.
        for _ in range(2):
            if self.genes[gene_index] < self.min_value[gene_index]:
                difference = self.min_value[gene_index] - self.genes[gene_index]
                # If the gene would reflect more than twice, just cancel out the redundant reflections.
                if difference > gene_range * 2:
                    difference %= gene_range * 2
                self.genes[gene_index] = self.min_value[gene_index] + difference
            elif self.genes[gene_index] > max_reflect:
                difference = self.genes[gene_index] - max_reflect
                # If the gene would reflect more than twice, just cancel out the redundant reflections.
                if difference > gene_range * 2:
                    difference %= gene_range * 2
                self.genes[gene_index] = max_reflect - difference

    def recombine_uniform(self, donor: 'LinearGenotype') -> None:
        for i in range(len(self.genes)):
            if random.random() < 0.5:
                self.genes[i] = donor.genes[i]
        self.parent_ids.append(donor.id)
        self.creation_method = 'Recombination'

    def recombine(self, donor: 'LinearGenotype') -> None:
        crossover_point = random.randrange(0, len(self.genes) - 0)  # Can copy whole individual, needed to prevent error on size 1 or 2
        for i in range(crossover_point):
            self.genes[i] = donor.genes[i]
        self.parent_ids.append(donor.id)
        self.creation_method = 'Recombination'

    def clone(self, copy_objectives = None) -> 'LinearGenotype':
        parameters = self.get_parameters()
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
        cloned_genotype.creation_method = 'Cloning'
        return cloned_genotype

    def get_fitness_modifier(self, raw_fitness):
        return 0

    def get_raw_genotype(self):
        return {'values': self.genes}

    def diversity_function(self, population, reference=None, samples=None):
        return genetic_algorithm_diversity(population, reference, samples)

    def __str__(self):
        return str(self.genes)

    def __hash__(self):
        return hash(tuple(self.genes))

    def get_values(self):
        return list(self.genes)

    def get_parameters(self) -> LinearGenotypeParameters:
        parameters = {'min_value': self.min_value, 'max_value': self.max_value, 'gene_mutation_rate': self.gene_mutation_rate, 'gene_mutation_standard_deviation': self.gene_mutation_standard_deviation, 'values': self.genes, 'loop_genes': self.loop_genes, 'round_genes': self.round_genes}
        return parameters
