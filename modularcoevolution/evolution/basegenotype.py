from typing import ClassVar, NewType

import abc
import math


GenotypeID = NewType("GenotypeID", int)


class BaseGenotype(metaclass=abc.ABCMeta):
    ID_counter: ClassVar[GenotypeID] = GenotypeID(0)

    id: GenotypeID

    # TODO: for all genotypes, accept a parameter list as keywords rather than strings
    def __init__(self, **kwargs):
        self.claim_next_id()  # Sets self.id
        self.fitness = None
        self.objectives = dict()
        self.inactive_objectives = list()
        self.objective_statistics = dict()
        self.fitness_counter = 0
        self.objectives_counter = dict()
        self.evaluated = False
        self.metrics = dict()
        self.past_objectives = dict()

        self.parents = list()
        self.creation_method = "Parthenogenesis"

    @abc.abstractmethod
    def mutate(self):
        pass

    @abc.abstractmethod
    def recombine(self, donor):
        pass

    @abc.abstractmethod
    def clone(self, copy_objectives=False):
        pass

    @abc.abstractmethod
    def __hash__(self):
        pass

    # TODO: Remove this function and associated legacy code, and treat fitness solely as an objective or metric.
    def set_fitness(self, fitness, average=False):
        if self.fitness is None or not average:
            self.fitness = fitness
            self.fitness_counter = 1
        else:
            self.fitness = (self.fitness * self.fitness_counter + fitness) / (self.fitness_counter + 1)
            self.fitness_counter += 1

    def set_objectives(self, objective_list, average_flags=None, inactive_objectives=None):
        if average_flags is None:
            average_flags = dict()
        for objective in objective_list:
            if objective not in average_flags:
                average_flags[objective] = False

        self.evaluated = True

        if inactive_objectives is not None:
            self.inactive_objectives = inactive_objectives

        for objective, value in objective_list.items():
            if objective not in self.objectives or not average_flags[objective]:
                self.objectives[objective] = value
                self.objectives_counter[objective] = 1
                self.objective_statistics[objective] = {"mean": value, "std dev intermediate": 0,
                                                        "standard deviation": 0,
                                                        "minimum": value, "maximum": value}
            else:
                self.objectives[objective] = (self.objectives[objective] * self.objectives_counter[objective] + objective_list[
                    objective]) / (self.objectives_counter[objective] + 1)
                standard_deviation_intermediate = self.objective_statistics[objective]["std dev intermediate"] + (
                            value - self.objectives[objective]) * (value - self.objective_statistics[objective]["mean"])
                # Objectives counter is already subtracting one
                standard_deviation = math.sqrt(standard_deviation_intermediate / self.objectives_counter[objective])
                self.objective_statistics[objective] = {"mean": self.objectives[objective],
                                                        "std dev intermediate": standard_deviation_intermediate,
                                                        "standard deviation": standard_deviation,
                                                        "minimum": min(value,
                                                                       self.objective_statistics[objective]["minimum"]),
                                                        "maximum": max(value,
                                                                       self.objective_statistics[objective]["maximum"])}

                self.objectives_counter[objective] += 1
            if objective not in self.past_objectives:
                self.past_objectives[objective] = list()
            self.past_objectives[objective].append(self.objectives[objective])
            self.metrics["past objectives"] = self.past_objectives

    def get_active_objectives(self):
        return {objective: self.objectives[objective] for objective in self.objectives if
                objective not in self.inactive_objectives}

    def get_fitness_modifier(self, raw_fitness):
        return 0

    @abc.abstractmethod
    def get_raw_genotype(self):
        pass

    @abc.abstractmethod
    def diversity_function(self, population, reference=None, samples=None):
        pass

    def claim_next_id(self):
        self.id = BaseGenotype.ID_counter
        BaseGenotype.ID_counter += 1
