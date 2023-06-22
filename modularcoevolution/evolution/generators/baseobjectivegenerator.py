from modularcoevolution.evolution.baseobjectivetracker import BaseObjectiveTracker
from modularcoevolution.evolution.specialtypes import GenotypeID, EvaluationID
from modularcoevolution.evolution.generators.basegenerator import BaseGenerator

from typing import Callable

import abc

class BaseObjectiveGenerator(BaseGenerator, metaclass=abc.ABCMeta):
    """A :class:`.BaseGenerator` with standardized handling of objectives.

    When extending this class, the

    """

    fitness_function: Callable[[dict[str, float]], float]
    """A function which takes a dictionary of named objectives, and outputs a single fitness value. If unset, fitness
    will be the average of all active objectives."""

    def __init__(self, fitness_function: Callable[[dict[str, float]], float] = None, **kwargs):
        self.fitness_function = fitness_function

    @abc.abstractmethod
    def get_genotype_with_id(self, agent_id) -> BaseObjectiveTracker:
        """Return the agent parameters associated with the given ID.

        To work with the implementation of :meth:`.set_objectives` here, this must return a subclass of
        :class:`.BaseObjectiveTracker`. If no complex genotype object is required, note that a class can inherit from
        :class:`.BaseObjectiveTracker` and :class:`.BaseAgent` simultaneously.

        Args:
            agent_id: The ID of the agent parameter set being requested.

        Returns: The agent parameter set associated with the ID ``agent_id``.

        """
        pass

    def set_objectives(self, agent_id: GenotypeID, objectives: dict[str, float], average_flags: dict[str, bool] = None,
                       average_fitness: bool = False, opponent: GenotypeID = None, evaluation_id: EvaluationID = None,
                       inactive_objectives: dict[str, bool] = None) -> None:
        """Called by a :class:`.BaseEvolutionWrapper` to record objective results from an evaluation
        for the agent with given index.

        This function can be called multiple times for the same agent. When this occurs, ``average_flags``
        will determine if the stored objective values should be overwritten, or store an average of objectives provided
        across each function call.

        Objectives are stored with :meth:`.BaseObjectiveTracker.set_objectives`.
        Fitness values are also calculated and stored in this function, as well as novelty metrics.

        Args:
            agent_id: The index of the agent associated with the objective results.
            objectives: A dictionary keyed by objective name holding the value for each objective.
            average_flags: A dictionary keyed by objective name.
                When the value for an objective is False, the previously stored objective will be overwritten with the
                new one.
                When the value for an objective is True, the stored objective will be an average for this objective
                across each function call.
                Defaults to false for every objective.
            average_fitness: Functions as ``average_flags``, but for a fitness value.
            opponent: The ID of the opponent that resulted in these objective values, if applicable.
            evaluation_id: The ID of evaluation associated with these objective values, for logging purposes.
            inactive_objectives: A dictionary keyed by objective name. Notes that an objective will be marked as
                "inactive" and only stored for logging purposes, rather than treated as a real objective.

        """
        if average_flags is None:
            average_flags = {objective: False for objective in objectives}
        individual = self.get_genotype_with_id(agent_id)
        individual.set_objectives(objectives, average_flags, inactive_objectives)
        if len(individual.get_active_objectives()) > 0:
            if self.fitness_function is not None:
                raw_fitness = self.fitness_function(individual.get_active_objectives())
                fitness_modifier = individual.get_fitness_modifier(raw_fitness)
                fitness = raw_fitness + fitness_modifier
                individual.set_fitness(fitness, average_fitness)
                individual.metrics["quality"] = raw_fitness
            else:
                raw_fitness = sum(individual.get_active_objectives().values()) /\
                                        len(individual.get_active_objectives())
                fitness_modifier = individual.get_fitness_modifier(raw_fitness)
                fitness = raw_fitness + fitness_modifier
                individual.set_fitness(fitness, average_fitness)
                individual.metrics["quality"] = raw_fitness
