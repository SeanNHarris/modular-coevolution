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

import math
import random

from modularcoevolution.genotypes.diversity import gpdiversity


def genetic_algorithm_distance(genome_1, genome_2):
    #return math.sqrt(sum([(x-y)**2 for x, y in zip(genome_1.genes, genome_2.genes)]))
    return math.dist(genome_1.genes, genome_2.genes)


# TODO: These are identical to the GP diversity one except for the distance, combine them
def genetic_algorithm_diversity(population, reference=None, samples=None):
    if reference is None and any(not i.fitness for i in population):
        reference = population[0]
    if reference is None:
        reference = max(population, key=lambda x: x.fitness)
    distance_sum = 0
    if samples is None:
        comparisons = population
    else:
        comparisons = random.sample(population, samples)
    for comparison in comparisons:
        distance_sum += genetic_algorithm_distance(reference, comparison)
    return distance_sum / samples


def multiple_genome_distance(multiple_1, multiple_2):
    from modularcoevolution.genotypes.lineargenotype import LinearGenotype
    from modularcoevolution.genotypes.geneticprogramming.gptree import GPTree
    distance_sum = 0
    for member_name in multiple_1.members:
        if member_name not in multiple_2.members:
            raise ValueError("The given genotypes have incompatible structures.")
        member_1 = multiple_1.members[member_name]
        member_2 = multiple_2.members[member_name]
        if isinstance(member_1, GPTree):
            distance_sum += gpdiversity.edit_distance(member_1, member_2)
        elif isinstance(member_1, LinearGenotype):
            distance_sum += genetic_algorithm_distance(member_1, member_2)
        else:
            distance_sum += genetic_algorithm_distance(member_1, member_2)
    return distance_sum / len(multiple_1.members)


def multiple_genome_diversity(population, reference=None, samples=None):
    if reference is None and any(not i.fitness for i in population):
        reference = population[0]
    if reference is None:
        reference = max(population, key=lambda x: x.fitness)
    distance_sum = 0
    if samples is None:
        comparisons = population
        samples = len(population)
    else:
        comparisons = random.sample(population, samples)
    for comparison in comparisons:
        distance_sum += multiple_genome_distance(reference, comparison)
    return distance_sum / samples
