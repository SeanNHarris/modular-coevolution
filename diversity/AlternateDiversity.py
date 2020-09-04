import diversity.GPDiversity as GPDiversity

import math
import random


def genetic_algorithm_distance(genome_1, genome_2):
    return math.sqrt(sum([(x-y)**2 for x, y in zip(genome_1.genes, genome_2.genes)]))


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
    return distance_sum / len(population)


def multiple_genome_distance(multiple_1, multiple_2):
    from alternate_genotypes.GeneticAlgorithm import GeneticAlgorithm
    from GeneticProgramming.GPTree import GPTree
    distance_sum = 0
    for member_1, member_2 in zip(multiple_1.members, multiple_2.members):
        if isinstance(member_1, GPTree):
            distance_sum += GPDiversity.edit_distance(member_1, member_2)
        if isinstance(member_1, GeneticAlgorithm):
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
    else:
        comparisons = random.sample(population, samples)
    for comparison in comparisons:
        distance_sum += multiple_genome_distance(reference, comparison)
    return distance_sum / len(population)
