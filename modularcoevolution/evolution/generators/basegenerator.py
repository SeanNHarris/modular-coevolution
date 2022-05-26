"""
Todo:
    * Determine why :meth:`~.generate_individual` ever existed, and if it is still needed.
        It's used nowhere else, but might have been needed for CEADS-LIN

"""


from modularcoevolution.evolution.baseagent import BaseAgent

from typing import Any
from typing.io import IO

import abc


class BaseGenerator(metaclass=abc.ABCMeta):
    """The superclass of all agent generators which participate in a :class:`.BaseEvolutionWrapper`, e.g.
    an :class:`.EvolutionGenerator` participating in :class:`.Coevolution`.

    A :class:`.BaseGenerator` maintains a data structure representing a population of agent parameter sets
    (such as genotypes for evolutionary agents).
    The population should not change unless the :meth:`next_generation` method is called.
    Each agent parameter set should be associated with a unique ID, and should remain accessible by this ID
    even if it has been removed from the population (in order to allow comparison with past agents).

    """

    _population_size: int

    def __init__(self):
        self.population_size = -1

    @property
    def population_size(self) -> int:
        """The size of the generator's agent population.

        The inheriting class *must* set a population size, even if it's just one.

        """
        assert self._population_size > 0
        return self._population_size

    @population_size.setter
    def population_size(self, value: int) -> None:
        self._population_size = value

    @abc.abstractmethod
    def get_genotype_with_id(self, agent_id) -> Any:
        """Return the agent parameters associated with the given ID. The type used for storing agent parameters is not
        prescribed by this abstract base class, but is frequently a :class:`.BaseGenotype`.

        Args:
            agent_id: The ID of the agent parameter set being requested.

        Returns: The agent parameter set associated with the ID ``agent_id``.

        """

    @abc.abstractmethod
    def build_agent_from_id(self, agent_id: int, active: bool) -> BaseAgent:
        """Return a new instance of an agent based on the given agent ID.

        Args:
            agent_id: The ID associated with the agent being requested.
            active: Used for the ``active`` parameter in :meth:`.BaseAgent.__init__`.

        Returns: A newly created agent associated with the ID ``agent_id`` and with ``active`` as its activity state.

        """

    @abc.abstractmethod
    def get_individuals_to_test(self) -> list[int]:
        """Get a list of agent IDs in need of evaluation. This can return individuals
        outside of the current generation, if desired (e.g. for a hall of fame).

        Returns: A list of IDs for agents which need to be evaluated.

        """
        pass

    @abc.abstractmethod
    def get_representatives_from_generation(self, generation: int, amount: int, force: bool = False) -> list[int]:
        """Return a set of agent IDs for high-quality representatives of the population from a certain generation,
        for intergenerational comparisons such as CIAO plots.

        Args:
            generation: The generation to choose the representatives from.
            amount: The number of representatives to return.
            force: If False, fewer than ``amount`` agents may be returned if there are not enough good choices.
                If True, exactly ``amount`` agents must be returned, even if this involves duplicates
                or low-quality agents.

        Returns:
            A set of IDs for high-quality agents to represent the requested generation, usually the ``amount`` best.
            If ``force`` is True, the size will be exactly ``amount``.

        """
        pass

    @abc.abstractmethod
    def set_objectives(self, agent_id: int, objectives: dict[str, float], average_flags: dict[str, bool] = None,
                       average_fitness: bool = False, opponent: BaseAgent = None, evaluation_number: int = None,
                       inactive_objectives: dict[str, bool] = None) -> None:
        """Called by a :class:`.BaseEvolutionWrapper` to record objective results from an evaluation
        for the agent with given index.

        Objectives should be stored internally in some way, as the :class:`.BaseEvolutionWrapper` is not required
        to maintain them.

        This function can be called multiple times for the same agent. When this occurs, consult ``average_flags``
        to determine if the stored objective values should be overwritten, or store an average of objectives provided
        across each function call.

        Args:
            agent_id: The ID of the agent associated with the objective results.
            objectives: A dictionary keyed by objective name holding the value for each objective.
            average_flags: A dictionary keyed by objective name.
                When the value for an objective is False, the previously stored objective should be overwritten with the
                new one.
                When the value for an objective is True, the stored objective should be an average for this objective
                across each function call.
                Defaults to false for every objective.
            average_fitness: Functions as ``average_flags``, but for a fitness value.
            opponent: The opponent that resulted in these objective values, if applicable.
            evaluation_number: The ID of evaluation associated with these objective values, for logging purposes.
            inactive_objectives: A dictionary keyed by objective name. Notes that an objective should be marked as
                "inactive" and only stored for logging purposes, rather than treated as a real objective.

        """
        pass

    @abc.abstractmethod
    def next_generation(self, result_log: IO = None, agent_log: IO = None) -> None:
        """Signals the generator that a generation has completed and that the generator may modify its population.

        Changes to the population should only occur as a result of this method being called. However, modifying the
        population at all is optional.

        Two log files may be provided for optional logging of data.

        Args:
            result_log: A log file intended to log the overall state of the population each generation.
            agent_log: A log file intended to log the state of individual agents in the population each generation.

        """
        pass

    def generate_individual(self, parameter_string: str) -> BaseAgent:
        """*Deprecated*, no longer required for implementation.
        Returns a new agent created from the given parameter string, of a type determined by the generator.

        Args:
            parameter_string: A string that can be used to generate an agent.

        Returns: An agent generated from the parameter string.

        """
        pass
