import abc
import copy
from typing import Sequence, Any, Union, Literal

from modularcoevolution.generators.archivegenerator import ArchiveGenerator
from modularcoevolution.agents.baseagent import BaseAgent
from modularcoevolution.genotypes.baseobjectivetracker import MetricConfiguration, BaseObjectiveTracker
from modularcoevolution.generators import BaseGenerator
from modularcoevolution.generators.basegenerator import MetricFunction
from modularcoevolution.utilities.specialtypes import GenotypeID
from modularcoevolution.wrappers.baseevolutionwrapper import BaseEvolutionWrapper


class BaseExperiment(metaclass=abc.ABCMeta):
    """A base class providing an interface to define an experiment,
    including its populations, agents, and evaluation function.

    :meth:`create_experiment` is used to initialize an experiment,
    while :meth:`create_archive_generators` is used to evaluate agents from a log during post-experiment analysis.

    Implementations of this class should be *stateless*,
    and should not modify their attributes outside of the ``__init__`` function."""

    config: dict[str, Any]
    """A dictionary of configuration parameters for the experiment from a configuration file."""

    def __init__(self, config: dict[str, Any]):
        """This method creates the experiment object and fixes its parameters, but does not create any generators.

        Args:
            config: A dictionary of configuration parameters for the experiment from a configuration file.
                The configuration file should only contain parameters that can not be fixed
                in the implementation of this class.
        """
        self.config = self._apply_config_defaults(config)

    @abc.abstractmethod
    def evaluate(self, agents: Sequence[BaseAgent], **kwargs) -> Sequence[dict[str, Any]]:
        """Evaluate the agents in the context of the experiment.

        Args:
            agents: An ordered list of agents, e.g. [attacker, defender]
            **kwargs: Any number of keyword arguments that will be passed by ``world_kwargs``.

        Returns:
            A list giving each agent a dictionary of results, e.g.
            `[{'score': 0.7, 'cost': 0.3}, {'score': 0.3, 'cost': 0.5}]`
        """
        pass
    
    @abc.abstractmethod
    def player_populations(self) -> Sequence[int]:
        """Return a list containing the population index which each player is drawn from.
        For a two-player game, this would be either [0, 0] or [0, 1] depending on
        whether the two players are drawn from the same population."""
        # TODO: Synchronize terminology for "population" vs. "generator"
        pass

    @abc.abstractmethod
    def population_names(self) -> Sequence[str]:
        """Return a list containing the name of each population.
        These names will be used in the config file and logs."""
        pass

    @abc.abstractmethod
    def population_agent_types(self) -> Sequence[type]:
        """Return a list containing the agent type of each population."""
        pass

    @abc.abstractmethod
    def _build_metrics(self) -> Sequence['PopulationMetrics']:
        """Create a :class:`.PopulationMetrics` object for each population to define the metrics for this experiment,
        and register any metrics that should be tracked using :meth:`.PopulationMetrics.register_metric`.
        These metrics will be automatically registered with the :class:`.BaseGenerator` objects in corresponding order.

        Returns:
            A list containing a :class:`.PopulationMetrics` object for each population in order.
        """
        pass

    @abc.abstractmethod
    def _create_generators(self) -> Sequence[BaseGenerator]:
        """Create the generators for each population in the experiment.

        Returns:
            A list containing a :class:`.BaseGenerator` object for each population in order.
        """
        pass

    @abc.abstractmethod
    def _create_manager(self, generators: Sequence[BaseGenerator]) -> BaseEvolutionWrapper:
        """Create the evolution/coevolution manager for the experiment.

        Args:
            generators: A list of generators corresponding to each population in the experiment.
                Use this and do not call :meth:`._create_generators` yourself.
        Returns:
            A :class:`.BaseEvolutionWrapper` object built with the provided `generators`.
        """
        pass

    def _apply_config_defaults(self, config: dict[str, Any]):
        """Update the config for each population with any missing default values.

        Args:
            config: A dictionary of configuration parameters containing a `defaults` key.

        Returns:
            The config dictionary, where each population configuration uses the corresponding values from `defaults`
            for parameters that were not explicitly specified.
        """
        updated_config = copy.deepcopy(config)
        if 'default' not in config:
            return updated_config

        if 'populations' not in config:
            updated_config['populations'] = {}
        for population in self.population_names():
            population_config: dict = copy.deepcopy(updated_config['default'])
            sub_configs = ('generator', 'genotype', 'agent')
            if population in updated_config['populations']:
                override_config = updated_config['populations'][population]
                for sub_config in sub_configs:
                    if sub_config not in population_config:
                        population_config[sub_config] = {}
                    if sub_config in override_config:
                        population_config.update(override_config[sub_config])
            updated_config['populations'][population] = population_config
        return updated_config


    def create_experiment(self) -> BaseEvolutionWrapper:
        """Create and initialize the generators and manager for an experiment.

        Returns:
            A :class:`.BaseEvolutionWrapper` object initialized for the experiment.
        """
        generators = self._create_generators()
        expected_names = self.population_names()
        for generator, name in zip(generators, expected_names):
            if generator.population_name != name:
                raise ValueError(f"Population names don't match those defined by population_names. Expected {name}, got {generator.population_name}.")

        metrics = self._build_metrics()
        for generator, population_metrics in zip(generators, metrics):
            for metric_configuration, metric_function in population_metrics.metrics:
                generator.register_metric(metric_configuration, metric_function)

        manager = self._create_manager(generators)
        return manager

    def create_archive_generators(self, genotypes: list[dict[GenotypeID, BaseObjectiveTracker]]) -> list[ArchiveGenerator]:
        """Create a list of :class:`.ArchiveGenerator` objects from a list of genotypes.
        This is used to load archived agents from a log to be evaluated during post-experiment analysis.

        Args:
            genotypes: A list of dictionaries of genotypes keyed by genotype ID (from the log), one for each population.

        Returns:
            A list of :class:`.ArchiveGenerator` objects, one for each population.
        """
        generators = []
        population_names = self.population_names()
        for population_index, population in enumerate(genotypes):
            agent_types = self.population_agent_types()
            generators.append(ArchiveGenerator(population_name=population_names[population_index],
                                               population=population,
                                               agent_class=agent_types[population_index],
                                               agent_parameters=self.config[population_index]['agent_parameters']))
        return generators


class PopulationMetrics:
    """An object which stores the metrics to be used for a population."""

    metrics: list[tuple[MetricConfiguration, MetricFunction]]
    """A list of metrics, where each metric is a tuple of (metric_configuration, metric_function)."""

    def __init__(self):
        self.metrics = []

    def register_metric(self,
                        metric_name: str,
                        metric_function: Union[MetricFunction, str],
                        is_objective: bool = False,
                        repeat_mode: Literal['replace', 'average', 'min', 'max'] = 'average',
                        log_history: bool = False) -> None:
        """
        Registers a metric.
        A metric is any data derived from the evaluation results which should be logged per-individual.
        Objectives are a special type of metric which are used during evolution, denoted by the ``'is_objective'`` configuration flag.
        In single-objective evolution, the objective corresponds to fitness.

        Args:
            metric_name: A string key for storing this metric.
            is_objective: If true, this metric is considered an objective.
                Objectives must be numeric values.
                Objectives are reported to generators, such as fitness for standard evolutionary algorithms.
                Other metrics are just stored for logging and analysis purposes.
            repeat_mode: How to handle multiple submissions of the same metric. The following modes are supported:
                - ``'replace'``: Overwrite the previous value with the new one.
                - ``'average'``: Record the mean of all submitted values. Must be a numeric type.
                - ``'min'``: Record the minimum of all submitted values. Must be a numeric type.
                - ``'max'``: Record the maximum of all submitted values. Must be a numeric type.
            log_history: If true, store a history of all submitted values for this metric.
                Avoid using this unnecessarily, as it can impact the size of the log file.
            metric_function: A function which computes the metric from the dictionary of evaluation results.
                Alternatively, a string key can be provided as a shortcut for a function which returns the result value with this key.
        """
        metric_configuration: MetricConfiguration = {
            'name': metric_name,
            'is_objective': is_objective,
            'repeat_mode': repeat_mode,
            'log_history': log_history,
            'automatic': True
        }
        self.metrics.append((metric_configuration, metric_function))

    def register_fitness_function(self,
                                  fitness_function: Union[MetricFunction, str],
                                  repeat_mode: Literal['replace', 'average', 'min', 'max'] = 'average') -> None:
        """
        Registers a fitness function.
        This is a shortcut for :meth:`register_metric` for single-objective evolution.

        Args:
            fitness_function: A function which computes the fitness from the dictionary of evaluation results.
                Alternatively, a string key can be provided as a shortcut for a function which returns the result value with this key.
            repeat_mode: How to handle multiple submissions of the same metric. The following modes are supported:
                - ``'replace'``: Overwrite the previous value with the new one.
                - ``'average'``: Record the mean of all submitted values. Must be a numeric type.
                - ``'min'``: Record the minimum of all submitted values. Must be a numeric type.
                - ``'max'``: Record the maximum of all submitted values. Must be a numeric type.
        """

        self.register_metric('fitness', fitness_function, is_objective=True, repeat_mode=repeat_mode, log_history=True)