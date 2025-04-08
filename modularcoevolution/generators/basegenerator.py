"""
Todo:
    * Determine why :meth:`~.generate_individual` ever existed, and if it is still needed.
        It's used nowhere else, but might have been needed for CEADS-LIN

"""
#  Copyright 2025 BONSAI Lab at Auburn University
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

__author__ = 'Sean N. Harris'
__copyright__ = 'Copyright 2025, BONSAI Lab at Auburn University'
__license__ = 'Apache-2.0'

from typing import Any, Callable, Generic, TypeVar, Union

import abc

# if TYPE_CHECKING:
from modularcoevolution.agents.baseagent import BaseAgent
from modularcoevolution.genotypes.baseobjectivetracker import BaseObjectiveTracker, MetricConfiguration, MetricTypes, \
    MetricSubmission
from modularcoevolution.utilities.specialtypes import GenotypeID, EvaluationID

AgentType = TypeVar("AgentType", bound=BaseAgent)
MetricFunction = Callable[[dict[str, Any]], MetricTypes]


class BaseGenerator(Generic[AgentType], metaclass=abc.ABCMeta):
    """The superclass of all agent generators which participate in a :class:`.BaseEvolutionManager`, e.g.
    an :class:`.EvolutionGenerator` participating in :class:`.Coevolution`.

    A :class:`.BaseGenerator` must maintain a data structure representing a population of agent parameter sets
    (such as genotypes for evolutionary agents).
    The population should not change unless the :meth:`next_generation` method is called.
    Each agent parameter set should be associated with a unique ID, and should remain accessible by this ID
    even if it has been removed from the population (in order to allow comparison with past agents).

    """

    population_name: str
    """The name of the population being generated. Used as a primary key for logging."""

    metric_configurations: dict[str, MetricConfiguration]
    """A dictionary of registered metric configurations, keyed by metric name."""
    metric_functions: dict[str, Union[MetricFunction, str]]
    """A dictionary of registered metric functions, keyed by metric name.
    A string can be used instead of a function as a shortcut to return the value of a key in the input dictionary."""

    @property
    @abc.abstractmethod
    def population_size(self) -> int:
        """The size of the generator's agent population.

        The inheriting class *must* set a population size, even if it's just one.

        """
        pass

    def __init__(self, population_name: str, *args, **kwargs):
        """
        Args:
            population_name: The name of the population being generated. Used as a primary key for logging.
        """
        self.population_name = population_name
        self.metric_configurations = {}
        self.metric_functions = {}

    @abc.abstractmethod
    def get_genotype_with_id(self, agent_id: GenotypeID) -> BaseObjectiveTracker:
        """Return the agent parameters associated with the given ID.

        To work with the implementation of :meth:`.set_objectives` here, this must return a subclass of
        :class:`.BaseObjectiveTracker`. If no complex genotype object is required, note that a class can inherit from
        :class:`.BaseObjectiveTracker` and :class:`.BaseAgent` simultaneously.

        Args:
            agent_id: The ID of the agent parameter set being requested.

        Returns: The agent parameter set associated with the ID ``agent_id``.

        """
        pass

    @abc.abstractmethod
    def build_agent_from_id(self, agent_id: GenotypeID, active: bool) -> AgentType:
        """Return a new instance of an agent based on the given agent ID.

        Args:
            agent_id: The ID associated with the agent being requested.
            active: Used for the ``active`` parameter in :meth:`.BaseAgent.__init__`.

        Returns: A newly created agent associated with the ID ``agent_id`` and with ``active`` as its activity state.

        """

    @abc.abstractmethod
    def get_individuals_to_test(self) -> list[GenotypeID]:
        """Get a list of agent IDs in need of evaluation. This can return individuals
        outside of the current generation, if desired (e.g. for a hall of fame).

        Returns: A list of IDs for agents which need to be evaluated.

        """
        pass

    def get_mandatory_opponents(self) -> list[GenotypeID]:
        """Get a list of agent IDs which must be evaluated against all opponents.
        An example use case of this is a hall of fame.
        The default implementation returns an empty list.

        Returns: A list of IDs for agents which must be evaluated against all opponents.
        """
        return []

    @abc.abstractmethod
    def get_representatives_from_generation(self, generation: int, amount: int, force: bool = False) -> list[GenotypeID]:
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

    def submit_evaluation(
            self,
            agent_id: GenotypeID,
            evaluation_results: dict[str, Any],
            opponents: list[GenotypeID] = None,
    ) -> None:
        """Called by a :class:`.BaseEvolutionManager` to record objectives and metrics from evaluation results
        for the agent with given index.

        Args:
            agent_id: The ID of the agent associated with the evaluation results.
            evaluation_results: The results of the evaluation.
            opponents: The IDs of the opponents the agent was evaluated against, if any.
        """

        for metric_name in self.metric_configurations:
            if self.metric_configurations[metric_name]['automatic']:
                self.submit_metric(agent_id, metric_name, self.compute_metric(metric_name, evaluation_results), opponents)

    @abc.abstractmethod
    def end_generation(self) -> None:
        """Called by a :class:`.BaseEvolutionManager` to signal that the current generation has ended.

        Sorting the population and any logging of the generation should be performed here.

        This method should not add or remove individuals from the population.

        """
        pass

    @abc.abstractmethod
    def next_generation(self) -> None:
        """Signals the generator that a generation has completed and that the generator may modify its population.

        Changes to the population should only occur as a result of this method being called. However, modifying the
        population at all is optional.

        This function will only be called after :meth:`.end_generation`, so it can be assumed that the population is sorted.

        """
        pass

    def generate_individual(self, parameter_string: str) -> AgentType:
        """*Deprecated*, no longer required for implementation.
        Returns a new agent created from the given parameter string, of a type determined by the generator.

        Args:
            parameter_string: A string that can be used to generate an agent.

        Returns: An agent generated from the parameter string.

        """
        pass

    def register_metric(self,
                        metric_configuration: MetricConfiguration,
                        metric_function: Union[MetricFunction, str, None]) -> None:
        """Register a metric with this generator.

        Args:
            metric_configuration: The metric to register.
            metric_function: A function which computes the metric from the dictionary of evaluation results.
                Alternatively, a string key in the dictionary of evaluation results which contains the metric value.

        """

        metric_name = metric_configuration['name']
        if metric_name in self.metric_configurations:
            raise ValueError(f"Metric \"{metric_name}\" is already registered with this generator.")
        self.metric_configurations[metric_name] = metric_configuration
        self.metric_functions[metric_name] = metric_function
        if metric_function is None and metric_configuration['automatic']:
            raise ValueError("A metric function must be provided for automatic metrics.")

    def submit_metric(
            self,
            agent_id: GenotypeID,
            metric_name: str,
            metric_value: MetricTypes,
            opponents: list[GenotypeID] = None,
    ) -> None:
        """Submit a metric value for an agent to store.
        The metric must have been previously registered using :meth:`.register_metric`.

        Args:
            agent_id: The ID of the agent associated with the metric.
            metric_name: The name of the metric.
            metric_value: The value of the metric.
            opponents: The IDs of the opponents the agent was evaluated against, if any.
        """
        if metric_name not in self.metric_configurations:
            raise ValueError(f"Metric {metric_name} not registered.")

        metric_submission: MetricSubmission = self.metric_configurations[metric_name].copy()
        metric_submission['value'] = metric_value
        if opponents is not None:
            metric_submission['opponents'] = opponents
        self.get_genotype_with_id(agent_id).submit_metric(metric_submission)

    def compute_metric(self, metric_name: str, evaluation_results: dict[str, Any]) -> MetricTypes:
        """Compute the requested metric from the given evaluation results.
        The metric must have been previously registered using :meth:`.register_metric`.

        Args:
            metric_name: The name of the metric.
            evaluation_results: The results of the evaluation.

        Returns:
            The value of the metric.

        """

        if metric_name not in self.metric_configurations:
            raise ValueError(f"Metric {metric_name} not registered.")

        if isinstance(self.metric_functions[metric_name], str):
            return evaluation_results[self.metric_functions[metric_name]]
        else:
            return self.metric_functions[metric_name](evaluation_results)

    def set_objectives(self, agent_id: GenotypeID, objectives: dict[str, float], opponent: GenotypeID = None, evaluation_id: EvaluationID = None) -> None:
        raise NotImplementedError("This method has been removed. Use submit_evaluation instead.")
