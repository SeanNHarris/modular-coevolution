"""
Todo:
    * Investigate removing fitness as a special value, and making it into an objective named "fitness".
"""

from modularcoevolution.evolution.specialtypes import ObjectiveStatistics

from typing import Any

import abc
import math


class BaseObjectiveTracker(metaclass=abc.ABCMeta):
    """A base class for anything that needs to track objectives or fitness. Formerly part of :class:`.BaseGenotype`.

    """

    objectives: dict[str, float]
    """A dictionary mapping objective names to objective values."""
    fitness: float | None
    """*(Deprecated)* A value summarizing the quality of an individual. Set to `None` if no fitness has been calculated.
    """
    inactive_objectives: list[str]
    """A list of objectives which can be recorded, but will not be used in any calculations for evolution or statistics.
    """
    objective_statistics: dict[str, ObjectiveStatistics]
    """A set of statistics maintained for each objective when multiple objective updates have been performed."""
    fitness_counter: int
    """How many fitness values have been sent, for statistical calculations."""
    objectives_counter: dict[str, int]
    """How many objective values have been sent for each objective, for statistical calculations."""
    evaluated: bool
    """Has this genotype been evaluated at least once? If ``False``, it will not have valid objectives"""
    metrics: dict[str, Any]
    """Additional values to be stored by name for this agent for logging purposes. These can vary by subclass."""
    past_objectives: dict[str, list[float]]
    """Stores the previous values of each objective, to record how it has changed over time for logging purposes."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.fitness = None
        self.objectives = dict()
        self.inactive_objectives = list()
        self.objective_statistics = dict()
        self.fitness_counter = 0
        self.objectives_counter = dict()
        self.evaluated = False
        self.metrics = dict()
        self.past_objectives = dict()

    def set_fitness(self, fitness: float, average: bool = False) -> None:
        """Set the ``fitness`` value, keeping an average if desired.

        Args:
            fitness: A fitness value to store.
            average: If True, the stored fitness value will be averaged across values sent over all calls of this
                method. If False, the stored fitness value will be set to the given value, overwriting previous values.

        """
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
                self.objective_statistics[objective] = {"mean": value, "std_dev_intermediate": 0,
                                                        "standard_deviation": 0,
                                                        "minimum": value, "maximum": value}
            else:
                self.objectives[objective] = (self.objectives[objective] * self.objectives_counter[objective] +
                                              objective_list[objective]) / (self.objectives_counter[objective] + 1)
                standard_deviation_intermediate = self.objective_statistics[objective]["std_dev_intermediate"] + (
                            value - self.objectives[objective]) * (value - self.objective_statistics[objective]["mean"])
                # Objectives counter is already subtracting one
                standard_deviation = math.sqrt(standard_deviation_intermediate / self.objectives_counter[objective])
                self.objective_statistics[objective] = {"mean": self.objectives[objective],
                                                        "std_dev_intermediate": standard_deviation_intermediate,
                                                        "standard_deviation": standard_deviation,
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
