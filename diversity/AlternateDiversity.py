import math
import sys

import diversity.GPDiversity as GPDiversity


def genetic_algorithm_distance(genome_1, genome_2):
    return math.sqrt(sum([(x-y)**2 for x, y in zip(genome_1.genes, genome_2.genes)]))


def genetic_algorithm_diversity(population, reference=None):
    population = population.copy()
    if reference is None and any(not i.fitnessSet for i in population):
        reference = population[0]
    if reference is None:
        population.sort(key=lambda x: x.fitness, reverse=True)
        reference = population[0]
    distance_sum = 0
    for i in range(len(population)):
        distance_sum += genetic_algorithm_distance(reference, population[i])
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


def multiple_genome_diversity(population, reference=None):
    population = population.copy()
    if reference is None and any(not i.fitnessSet for i in population):
        reference = population[0]
    if reference is None:
        population.sort(key=lambda x: x.fitness, reverse=True)
        reference = population[0]
    distance_sum = 0
    for i in range(len(population)):
        distance_sum += multiple_genome_distance(reference, population[i])
    return distance_sum / len(population)
