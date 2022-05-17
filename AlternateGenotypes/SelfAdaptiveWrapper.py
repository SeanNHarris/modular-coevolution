from Evolution.BaseGenotype import *

import random


class SelfAdaptiveWrapper(BaseGenotype):

    def __init__(self, parameters):
        super().__init__()
        if "genotype class" in parameters:
            self.genotype_class = parameters["genotype class"]
            self.genotype = self.genotype_class(parameters)
        elif "genotype" in parameters:
            self.genotype = parameters["genotype"].clone()
            self.genotype_class = self.genotype.__class__

        if "mutation types" in parameters:
            self.mutation_types = parameters["mutation types"]
        else:
            self.mutation_types = {self.genotype.mutate: []}

        self.self_adaptive_parameters = {"mutation rate": random.random(), "mutation amount": random.random(),
                                         "mutation type": random.choice(list(self.mutation_types))}

    def mutate(self):
        mutation_function = self.self_adaptive_parameters["mutation type"]
        mutation_parameters = [self.self_adaptive_parameters[parameter] for parameter in self.mutation_types[mutation_function]]
        mutation_function(self.genotype, *mutation_parameters)
        self.self_adaptive_parameters["mutation rate"] = min(
            max(self.self_adaptive_parameters["mutation rate"] + random.gauss(0, 0.1), 0.1), 1)
        self.self_adaptive_parameters["mutation amount"] = min(
            max(self.self_adaptive_parameters["mutation amount"] + random.gauss(0, 0.1), 0.1), 1)
        if random.random() < .25:
            self.self_adaptive_parameters["mutation type"] = random.choice(list(self.mutation_types))

    def recombine(self, donor):
        self.genotype.recombine(donor.genotype)
        self.self_adaptive_parameters = {parameter: random.choice(
            (self.self_adaptive_parameters[parameter], donor.self_adaptive_parameters[parameter])) for parameter in
                                        self.self_adaptive_parameters}

    def clone(self, copy_objectives=False):
        cloned_genotype = SelfAdaptiveWrapper({"genotype": self.genotype.clone(), "mutation types": self.mutation_types})
        cloned_genotype.self_adaptive_parameters = self.self_adaptive_parameters.copy()
        if copy_objectives:
            for objective in self.objectives:
                cloned_genotype.objectives[objective] = self.objectives[objective]
                cloned_genotype.objective_statistics[objective] = self.objective_statistics[objective]
                cloned_genotype.objectives_counter[objective] = self.objectives_counter[objective]
                cloned_genotype.past_objectives[objective] = self.past_objectives[objective]
        return cloned_genotype

    def __hash__(self):
        return hash(self.genotype)

    def get_raw_genotype(self):
        return {"genotype": self.genotype.get_raw_genotype(), "self-adaptive parameters": self.self_adaptive_parameters}

    def diversity_function(self, population, reference=None, samples=None):
        return self.genotype.diversity_function([individual.genotype for individual in population], reference.genotype, samples)

    def get_fitness_modifier(self, raw_fitness):
        return self.genotype.get_fitness_modifier(raw_fitness)

    def execute(self, context):
        return self.genotype.execute(context)

    def __str__(self):
        string = str(self.genotype)
        string += "\nSelf-adaptive parameters: {0}".format(self.self_adaptive_parameters)
        return string