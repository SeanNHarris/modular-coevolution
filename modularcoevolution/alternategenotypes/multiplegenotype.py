from typing import TypeVar, Generic

from modularcoevolution.diversity.alternatediversity import multiple_genome_diversity
from modularcoevolution.evolution.basegenotype import BaseGenotype


MemberType = TypeVar("MemberType", bound=BaseGenotype)


class MultipleGenotype(BaseGenotype, Generic[MemberType]):
    members: dict[str | int, MemberType]

    def __init__(self, parameters):
        super().__init__(parameters)
        if "members" in parameters:
            self.members = dict()
            for name, member in parameters["members"].items():
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

    def mutate(self) -> None:
        for name, member in self.members.items():
            member.mutate()
        self.creation_method = "Mutation"

    def recombine(self, donor: "MultipleGenotype") -> None:
        for name in self.members:
            assert isinstance(self.members[name], type(donor.members[name]))
            self.members[name].recombine(donor.members[name])
        self.parent_ids.append(donor.id)
        self.creation_method = "Recombination"

    def clone(self, copy_objectives=False) -> "MultipleGenotype":
        parameters = {"members": self.members}
        cloned_genotype = MultipleGenotype(parameters)
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
        for name in self.members:
            cloned_genotype.members[name].parent_ids.append(self.members[name].id)
            cloned_genotype.members[name].creation_method = "Cloning"
        return cloned_genotype

    def get_fitness_modifier(self, raw_fitness):
        return sum([self.members[member].get_fitness_modifier(raw_fitness) for member in self.members])

    def get_raw_genotype(self):
        raw_genotype = dict()
        for name, member in self.members.items():
            raw_genotype[name] = member.get_raw_genotype()
        return raw_genotype


    # TODO: This was probably like this to ensure compliance, but this can be redone with a function class variable, please do so
    def diversity_function(self, population, reference=None, samples=None):
        return multiple_genome_diversity(population, reference, samples)

    def __str__(self):
        string = "(Multiple Genotype)"
        for name in self.members:
            string += f"\n{name}: {self.members[name]}"
        return string

    def __hash__(self) -> int:
        return hash(tuple(self.members.values()))