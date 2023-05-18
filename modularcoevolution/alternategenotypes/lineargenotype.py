from modularcoevolution.diversity.alternatediversity import genetic_algorithm_diversity
from modularcoevolution.evolution.basegenotype import BaseGenotype

from typing import Iterable

import random

# TODO: These should probably be parameters somewhere? Either per individual (may be inconsistent), per method (may be tedious), or per class (might be problematic for multiple populations)
MIN_VALUE_DEFAULT = 0
MAX_VALUE_DEFAULT = 10
GENE_MUTATION_RATE_DEFAULT = 0.33


class LinearGenotype(BaseGenotype):
    def __init__(self, parameters):
        super().__init__()
        if "gene_mutation_rate" in parameters:
            self.gene_mutation_rate = parameters["gene_mutation_rate"]
        else:
            self.gene_mutation_rate = GENE_MUTATION_RATE_DEFAULT

        must_generate = False
        self.genes = None
        if "values" in parameters:
            self.genes = parameters["values"].copy()
            self.length = len(self.genes)
        elif "length" in parameters:
            must_generate = True
            self.genes = list()
            self.length = parameters["length"]
        if self.genes is None:
            raise TypeError("If \"values\" is not provided, a \"length\" must be.")

        if "min_value" in parameters:
            if isinstance(parameters["min_value"], Iterable):
                self.min_value = list()
                for i in range(self.length):
                    self.min_value.append(parameters["min_value"][i % len(parameters["min_value"])])
            else:
                self.min_value = [parameters["min_value"] for _ in range(self.length)]
        else:
            self.min_value = [MIN_VALUE_DEFAULT for _ in range(self.length)]

        if "max_value" in parameters:
            if isinstance(parameters["max_value"], Iterable):
                self.max_value = list()
                for i in range(self.length):
                    self.max_value.append(parameters["max_value"][i % len(parameters["max_value"])])
            else:
                self.max_value = [parameters["max_value"] for _ in range(self.length)]
        else:
            self.max_value = [MAX_VALUE_DEFAULT for _ in range(self.length)]

        if "loop_genes" in parameters:
            self.loop_genes = parameters["loop_genes"]
        else:
            self.loop_genes = [False for _ in range(self.length)]

        if must_generate:
            for index in range(parameters["length"]):
                self.genes.append(self.random_gene(index))

    def random_gene(self, index):
        return random.random() * (self.max_value[index] - self.min_value[index]) + self.min_value[index]

    def mutate(self):
        for i in range(len(self.genes)):
            if random.random() < self.gene_mutation_rate:
                self.genes[i] = self.genes[i] + random.gauss(0, (self.max_value[i] - self.min_value[i]) / 100)
                if self.loop_genes[i]:
                    self.genes[i] = (self.genes[i] - self.min_value[i]) % (self.max_value[i] - self.min_value[i]) + self.min_value[i]
                else:
                    self.genes[i] = max(self.min_value[i], min(self.genes[i], self.max_value[i]))

        self.creation_method = "Mutation"

    def recombine_uniform(self, donor):
        for i in range(len(self.genes)):
            if random.random() < 0.5:
                self.genes[i] = donor.genes[i]
        self.parent_ids.append(donor.id)
        self.creation_method = "Recombination"

    def recombine(self, donor):
        crossover_point = random.randrange(0, len(self.genes) - 0)  # Can copy whole individual, needed to prevent error on size 1 or 2
        for i in range(crossover_point):
            self.genes[i] = donor.genes[i]
        self.parent_ids.append(donor.id)
        self.creation_method = "Recombination"

    def clone(self, copy_objectives={}):
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
        cloned_genotype.creation_method = "Cloning"
        return cloned_genotype

    def get_fitness_modifier(self, raw_fitness):
        return 0

    def get_raw_genotype(self):
        return {"values": self.genes}

    def diversity_function(self, population, reference=None, samples=None):
        return genetic_algorithm_diversity(population, reference, samples)

    def __str__(self):
        return str(self.genes)

    def __hash__(self):
        return hash(tuple(self.genes))

    def get_values(self):
        return list(self.genes)

    def get_parameters(self):
        parameters = {"min_value": self.min_value, "max_value": self.max_value, "gene_mutation_rate": self.gene_mutation_rate, "values": self.genes, "loop_genes": self.loop_genes}
        return parameters