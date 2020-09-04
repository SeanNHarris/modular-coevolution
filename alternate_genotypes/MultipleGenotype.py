from diversity.AlternateDiversity import multiple_genome_diversity
from Evolution.BaseGenotype import BaseGenotype

import random


class MultipleGenotype(BaseGenotype):
    def __init__(self, parameters):
        super().__init__(parameters)
        if "members" in parameters:
            self.members = dict()
            for name, member in parameters["members"]:
                self.members[name] = member.clone()
            return

        if "subgenotypes" in parameters and "subparameters" in parameters:
            self.members = dict()
            for name in parameters["subgenotypes"]:
                genotype = parameters["subgenotypes"][name]
                subparameters = parameters["subparameters"][name]
                self.members[name] = genotype(subparameters)
        elif "subgenotypes" in parameters or "subparameters" in parameters:
            raise TypeError("Must have both \"subgenotypes\" and \"subparameters\".")
        else:
            raise TypeError("Must provide either \"members\" or both \"subgenotypes\" and \"subparameters\".")

    def mutate(self):
        for name, member in self.members:
            member.mutate()
        self.creation_method = "Mutation"

    def recombine(self, donor):
        for name in self.members:
            assert isinstance(self.members[name], type(donor.members[name]))
            self.members[name].recombine(donor.members[name])
        self.parents.append(donor)
        self.creation_method = "Recombination"

    def clone(self, copy_objectives=False):
        parameters = {"members": self.members}
        cloned_genotype = MultipleGenotype(parameters)
        if copy_objectives:
            for objective in self.objectives:
                cloned_genotype.objectives[objective] = self.objectives[objective]
                cloned_genotype.objective_statistics[objective] = self.objective_statistics[objective]
                cloned_genotype.objectives_counter[objective] = self.objectives_counter[objective]
                cloned_genotype.past_objectives[objective] = self.past_objectives[objective]
        cloned_genotype.parents.append(self)
        cloned_genotype.creation_method = "Cloning"
        for name in self.members:
            cloned_genotype.members[name].parents.append(self.members[name])
            cloned_genotype.members[name].creation_method = "Cloning"
        return cloned_genotype

    def get_fitness_modifier(self):
        return 0

    def get_raw_genotype(self):
        raw_genotype = dict()
        for name, member in self.members:
            raw_genotype[name] = member.get_raw_genotype()


    # TODO: This was probably like this to ensure compliance, but this can be redone with a function class variable, please do so
    def diversity_function(self, population, reference=None, samples=None):
        return multiple_genome_diversity(population, reference, samples)

    def __str__(self):
        string = "(Multiple Genotype)"
        for name in self.members:
            string += "\n" + name + ": " + str(self.members[name])
        return string