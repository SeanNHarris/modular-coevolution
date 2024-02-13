import abc
import copy
import itertools
import multiprocessing
from functools import partial
from typing import Sequence, Any, Union, Literal, Callable, Protocol

from modularcoevolution.generators.archivegenerator import ArchiveGenerator
from modularcoevolution.agents.baseagent import BaseAgent
from modularcoevolution.genotypes.baseobjectivetracker import MetricConfiguration, BaseObjectiveTracker
from modularcoevolution.generators.basegenerator import BaseGenerator
from modularcoevolution.generators.basegenerator import MetricFunction
from modularcoevolution.utilities import parallelutils
from modularcoevolution.utilities.dictutils import deep_copy_dictionary
from modularcoevolution.utilities.specialtypes import GenotypeID
from modularcoevolution.managers.baseevolutionmanager import BaseEvolutionManager


class EvaluateProtocol(Protocol):
    def __call__(self, agents: Sequence[BaseAgent]) -> Sequence[dict[str, Any]]:
        """Evaluate the agents in the context of the experiment.

        Args:
            agents: An ordered list of agents, e.g. [attacker, defender]

        Returns:
            A list giving each agent a dictionary of results, e.g.
            `[{'score': 0.7, 'cost': 0.3}, {'score': 0.3, 'cost': 0.5}]`
        """
        ...


class BaseExperiment(metaclass=abc.ABCMeta):
    """A base class providing an interface to define an experiment,
    including its populations, agents, and evaluation function.

    :meth:`create_experiment` is used to initialize an experiment,
    while :meth:`create_archive_generators` is used to evaluate agents from a log during post-experiment analysis.

    Implementations of this class should be *stateless*,
    and should not modify their attributes outside of the ``__init__`` function."""

    config: dict[str, Any]
    """A dictionary of configuration parameters for the experiment from a configuration file."""
    agent_types_by_population_name: dict[str, type[BaseAgent]]
    """A dictionary of agent types, indexed by population name."""

    def __init__(self, config: dict[str, Any]):
        """This method creates the experiment object and fixes its parameters, but does not create any generators.
        This function should be called at the *end* of the inheriting class' ``__init__`` function.

        Args:
            config: A dictionary of configuration parameters for the experiment from a configuration file.
                The configuration file should only contain parameters that can not be fixed
                in the implementation of this class.
        """
        self.config = self._apply_config_defaults(config)
        self.agent_types_by_population_name = {}
        for population_name, agent_type in zip(self.population_names(), self.population_agent_types()):
            self.agent_types_by_population_name[population_name] = agent_type

    @abc.abstractmethod
    def get_evaluate(self, **kwargs) -> EvaluateProtocol:
        """Return an evaluation function which takes a list of agents and returns a list of results.
        Any parameters should be baked into the evaluation function using :func:`functools.partial` or similar methods.

        Returns:
            A function matching the :class:`.EvaluateProtocol` protocol.
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
    def _create_manager(self, generators: Sequence[BaseGenerator]) -> BaseEvolutionManager:
        """Create the evolution/coevolution manager for the experiment.

        Args:
            generators: A list of generators corresponding to each population in the experiment.
                Use this and do not call :meth:`._create_generators` yourself.
        Returns:
            A :class:`.BaseEvolutionManager` object built with the provided `generators`.
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
        updated_config = deep_copy_dictionary(config)
        if 'default' not in config:
            return updated_config

        if 'populations' not in config:
            updated_config['populations'] = {}
        for population in self.population_names():
            population_config: dict = deep_copy_dictionary(updated_config['default'])
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


    def create_experiment(self) -> BaseEvolutionManager:
        """Create and initialize the generators and manager for an experiment.

        Returns:
            A :class:`.BaseEvolutionManager` object initialized for the experiment.
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

    def create_archive_generators(self,
                                  genotypes: dict[str, Sequence[BaseObjectiveTracker]],
                                  original_ids: dict[str, dict[GenotypeID, int]]) -> dict[str, ArchiveGenerator]:
        """Create a list of :class:`.ArchiveGenerator` objects from a list of genotypes.
        This is used to load archived agents from a log to be evaluated during post-experiment analysis.

        Args:
            genotypes: A nested dictionary, mapping population names to genotypes.
                Generators will only be created for populations in this dictionary,
                not for all populations in the experiment (in case agents from multiple log files are being mixed).
            original_ids: A nested dictionary, mapping population names to mappings from current `GenotypeID` values to original logged IDs.

        Returns:
            A dictionary of :class:`.ArchiveGenerator` objects keyed by population name.
        """
        generators = {}
        for population_name in genotypes:
            if population_name not in self.agent_types_by_population_name:
                raise ValueError(f"Genotypes submitted for {population_name}, which does not exist in this experiment.")

            generators[population_name] = ArchiveGenerator(
                population_name=population_name,
                genotypes=genotypes[population_name],
                original_ids=original_ids[population_name],
                agent_class=self.agent_types_by_population_name[population_name],
                agent_parameters=self.config['populations'][population_name]['agent']
            )

        metrics = self._build_metrics()
        for population_index, population_name in enumerate(self.population_names()):
            if population_name not in generators:
                continue
            generator = generators[population_name]
            population_metrics = metrics[population_index]
            for metric_configuration, metric_function in population_metrics.metrics:
                generator.register_metric(metric_configuration, metric_function)

        return generators

    def evaluate_all(self, agent_groups: Sequence[Sequence[BaseAgent]], parallel: bool = False, exhibition: bool = False, evaluation_pool: multiprocessing.Pool = None) -> list[Sequence[dict[str, Any]]]:
        """Evaluate a list of agent groups in parallel using a multiprocessing pool and return the results.
        If ``self.parallel`` is False, this will instead evaluate the agents sequentially.

        Args:
            agent_groups: A list of agent groups to evaluate.
            parallel: Whether to evaluate the agents using a multiprocessing pool.
            exhibition: If true, sends an `exhibition = True` parameter to the evaluation function.
            evaluation_pool: The multiprocessing pool to use for evaluation if ``parallel`` is True.
                If None, a new pool will be created.

        Returns:
            A list of results from the evaluation function, in the order of the agent groups passed in.

        """
        evaluate = self.get_evaluate(exhibition=exhibition)

        if parallel:
            if evaluation_pool is None:
                evaluation_pool = parallelutils.create_pool()
            end_states = evaluation_pool.map(evaluate, agent_groups, chunksize=len(agent_groups) // (evaluation_pool._processes * 10))
        else:
            end_states = list()
            for agents in agent_groups:
                end_states.append(evaluate(agents))
        return end_states


    def exhibition(self,
                    populations: Sequence[BaseGenerator],
                    amount: int,
                    log_path: str,
                    parallel: bool = False,
                    evaluation_pool: multiprocessing.Pool = None) -> None:
        """Run exhibition evaluations between the best individuals of the current generation.

        Args:
            populations: The list of generators to pull individuals from.
            amount: The number of agents to evaluate from each population.
            log_path: The path to the current log folder.
            parallel: Whether to evaluate the agents using a multiprocessing pool.
            evaluation_pool: The multiprocessing pool to use for evaluation.
        """
        agent_ids = [generator.get_representatives_from_generation(-1, amount) for generator in populations]
        population_agents = [[population.build_agent_from_id(agent_id, True) for agent_id in agent_ids[population_index]] for population_index, population in enumerate(populations)]
        agents = [population_agents[population_index] for population_index in self.player_populations()]
        agent_names = [populations[population_index].population_name for population_index in self.player_populations()]

        agent_groups = list(itertools.product(*agents))
        # Don't evaluate games with duplicate agents.
        # The same pair of agents can still be evaluated multiple times in different player orders.
        agent_groups = [agent_group for agent_group in agent_groups if len(set(agent_group)) == len(agent_group)]
        agent_numbers = [range(amount) for _ in agents]
        agent_group_numbers = list(itertools.product(*agent_numbers))
        results = self.evaluate_all(agent_groups, parallel=False, exhibition=True, evaluation_pool=evaluation_pool)
        for agent_group, agent_numbers, result in zip(agent_groups, agent_group_numbers, results):
            self._process_exhibition_results(agent_group, agent_numbers, agent_names, result, log_path)

    def _process_exhibition_results(self, agent_group, agent_numbers, agent_names, result, log_path):
        number_string = '-'.join([str(number) for number in agent_numbers])
        statistics_filepath = f'{log_path}/exhibitionStats{number_string}.txt'
        with open(statistics_filepath, 'w+') as statistics_file:
            statistics_file.truncate(0)
            for player_index, agent in enumerate(agent_group):
                agent_name = agent_names[player_index]
                statistics_file.write(f'{agent_name} genotype:\n{agent.genotype}\n')
                for metric_name, metric_value in result[player_index].items():
                    statistics_file.write(f'{metric_name}:\n{metric_value}\n')
                statistics_file.write('\n')



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
                        repeat_mode: Literal['replace', 'average', 'min', 'max', 'sum'] = 'average',
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
                - ``'sum'``: Record the sum of all submitted values. Must be a numeric type.
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