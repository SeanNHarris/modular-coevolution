import abc
import copy
import itertools
import multiprocessing
import os
from functools import partial
from typing import Sequence, Any, Union, Literal, Callable, Protocol

from modularcoevolution.agents.baseevolutionaryagent import BaseEvolutionaryAgent
from modularcoevolution.generators.archivegenerator import ArchiveGenerator
from modularcoevolution.agents.baseagent import BaseAgent
from modularcoevolution.genotypes.baseobjectivetracker import MetricConfiguration, BaseObjectiveTracker
from modularcoevolution.generators.basegenerator import BaseGenerator
from modularcoevolution.generators.basegenerator import MetricFunction
from modularcoevolution.generators.randomgenotypegenerator import RandomGenotypeGenerator
from modularcoevolution.managers.coevolution import Coevolution
from modularcoevolution.utilities import parallelutils
from modularcoevolution.utilities.dictutils import deep_copy_dictionary, deep_update_dictionary
from modularcoevolution.utilities.specialtypes import GenotypeID
from modularcoevolution.managers.baseevolutionmanager import BaseEvolutionManager

try:
    import tqdm
except ImportError:
    tqdm = None


class EvaluateProtocol(Protocol):
    def __call__(self, agents: Sequence[BaseAgent], **kwargs) -> Sequence[dict[str, Any]]:
        """Evaluate the agents in the context of the experiment.

        Args:
            agents: An ordered list of agents, e.g. [attacker, defender]

        Returns:
            A list giving each agent a dictionary of results, e.g.
            `[{'score': 0.7, 'cost': 0.3}, {'score': 0.3, 'cost': 0.5}]`
            The nth entry in this list should contain everything needed to compute the nth agent's metrics.
            (This may result in duplicate values between agents).
            Optionally, an additional dictionary can be returned containing metrics for logging or visualization.
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
    def population_generator_types(self) -> Sequence[type]:
        """Return a list containing the generator type of each population.
        Used for the default implementation of :meth:`_create_generators`.
        """

    @abc.abstractmethod
    def _build_metrics(self) -> Sequence['PopulationMetrics']:
        """Create a :class:`.PopulationMetrics` object for each population to define the metrics for this experiment,
        and register any metrics that should be tracked using :meth:`.PopulationMetrics.register_metric`.
        These metrics will be automatically registered with the :class:`.BaseGenerator` objects in corresponding order.

        Returns:
            A list containing a :class:`.PopulationMetrics` object for each population in order.
        """
        pass

    def create_generator(self, index) -> BaseGenerator:
        """Create a new generator for a specific population in the experiment.

        The default implementation takes parameters from:
        `config['populations'][(population name)]['generator']`,
        `config['populations'][(population name)]['genotype']`, and
        `config['populations'][(population name)]['agent']`

        Args:
            index: The population index to create a generator for.

        Returns:
            A :class:`.BaseGenerator` subclass based on the `index`th entry in :meth:`.population_generator_types`.
            The generator will be configured with parameters from the config file for the associated population name.
        """
        population_name = self.population_names()[index]
        generator_type = self.population_generator_types()[index]
        agent_type = self.population_agent_types()[index]

        population_config = self.config['populations'][population_name]
        generator_parameters: dict = population_config['generator']
        genotype_parameters: dict = population_config['genotype']
        agent_parameters: dict = population_config['agent']

        generator = generator_type(
            agent_type,
            population_name,
            genotype_parameters=genotype_parameters,
            agent_parameters=agent_parameters,
            **generator_parameters
        )
        return generator

    def _create_manager(self, generators: Sequence[BaseGenerator]) -> BaseEvolutionManager:
        """Create the evolution/coevolution manager for the experiment.

        The default implementation uses the :class:`.Coevolution` manager, taking parameters from
        `config['manager']`

        Args:
            generators: A list of generators corresponding to each population in the experiment.
                Use this and do not call :meth:`._create_generators` yourself.
        Returns:
            A :class:`.BaseEvolutionManager` object built with the provided `generators`.
        """

        manager_parameters = self.config['manager']
        return Coevolution(generators, self.player_populations(), **manager_parameters)

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
                        deep_update_dictionary(population_config[sub_config], override_config[sub_config])
            updated_config['populations'][population] = population_config
        return updated_config


    def create_experiment(self) -> BaseEvolutionManager:
        """Create and initialize the generators and manager for an experiment.

        Returns:
            A :class:`.BaseEvolutionManager` object initialized for the experiment.
        """
        generators = [self.create_generator(index) for index in range(len(self.population_names()))]
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

    def create_random_generators(self, generate_size: int, reduce_size: int = -1) -> Sequence[RandomGenotypeGenerator]:
        """Create a list of :class:`.RandomGenotypeGenerator` objects corresponding to the populations in the experiment.

        Args:
            generate_size: The number of random genotypes to generate in each population.
                See :meth:`.RandomGenotypeGenerator.__init__`.
            reduce_size: See :attr:`.RandomGenotypeGenerator.reduce_size`.
                This method does not perform evaluation or reduction, it just passes this parameter to the generators.

        Returns:
            A list of :class:`.RandomGenotypeGenerator` objects, one for each population in the experiment.
            These generators will be initialized with random genotypes corresponding to the experiment configuration.
        """

        generators = []
        for population_name in self.population_names():
            agent_class = self.agent_types_by_population_name[population_name]
            agent_parameters = self.config['populations'][population_name]['agent']
            genotype_parameters = self.config['populations'][population_name]['genotype']
            generator = RandomGenotypeGenerator(
                agent_class=agent_class,
                population_name=population_name,
                generate_size=generate_size,
                reduce_size=reduce_size,
                agent_parameters=agent_parameters,
                genotype_parameters=genotype_parameters
            )
            generators.append(generator)
        return generators

    def evaluate_all(self, agent_groups: Sequence[Sequence[BaseAgent]], parallel: bool = False, **kwargs) -> list[Sequence[dict[str, Any]]]:
        """Evaluate a list of agent groups in parallel using a multiprocessing pool and return the results.
        If ``self.parallel`` is False, this will instead evaluate the agents sequentially.

        Args:
            agent_groups: A list of agent groups to evaluate.
            parallel: Whether to evaluate the agents using a multiprocessing pool.
            kwargs: Additional keyword arguments to pass to the evaluation function.

        Returns:
            A list of results from the evaluation function, in the order of the agent groups passed in.

        """
        evaluate = self.get_evaluate(**kwargs)

        if parallel:
            evaluation_pool = parallelutils.create_pool()
            chunks = parallelutils.cores_available() * 8
            chunksize = max(1, len(agent_groups) // chunks)
            result_iterator = evaluation_pool.map(evaluate, agent_groups, chunksize=chunksize)
        else:
            evaluation_pool = None
            result_iterator = map(evaluate, agent_groups)

        if tqdm is not None and len(agent_groups) > 1:
            result_iterator = tqdm.tqdm(result_iterator, total=len(agent_groups), desc="Running evaluations", unit="evals", smoothing=0.0)

        try:
            results = []
            for result in result_iterator:
                results.append(result)
            if parallel:
                evaluation_pool.shutdown()
            return results
        except KeyboardInterrupt as interrupt:
            # If the user stops execution during evaluations, terminate the pool to kill any remaining processes.
            if parallel:
                evaluation_pool.shutdown(wait=False, cancel_futures=True)
            raise interrupt


    def exhibition(
            self,
            populations: Sequence[BaseGenerator],
            amount: int,
            log_path: str,
            generation: int = -1,
            parallel: bool = False
    ) -> None:
        """Run exhibition evaluations between the best individuals of the current generation.

        Args:
            populations: The list of generators to pull individuals from.
            amount: The number of agents to evaluate from each population.
            log_path: The path to the current log folder.
            generation: The generation to pull individuals from. Defaults to the last completed generation.
            parallel: Whether to evaluate the agents using a multiprocessing pool.
            evaluation_pool: The multiprocessing pool to use for evaluation.
        """
        agent_ids = [generator.get_representatives_from_generation(generation, amount) for generator in populations]
        population_agents = [[population.build_agent_from_id(agent_id, True) for agent_id in agent_ids[population_index]] for population_index, population in enumerate(populations)]
        agents = [population_agents[population_index] for population_index in self.player_populations()]
        agent_names = [populations[population_index].population_name for population_index in self.player_populations()]
        self._run_exhibition_games(agents, agent_names, log_path, parallel=False)
        # Disable parallel exhibitions, because of issues with pickling the extended result dictionary.

    def _run_exhibition_games(
            self,
            agents: Sequence[Sequence[BaseAgent]],
            agent_names: Sequence[str],
            log_path: str | os.PathLike,
            parallel: bool = False
    ) -> None:

        agent_groups = list(itertools.product(*agents))
        agent_numbers = [range(len(agents[i])) for i in range(len(agents))]
        agent_group_numbers = list(itertools.product(*agent_numbers))
        results = self.evaluate_all(agent_groups, parallel=parallel, exhibition=True)
        for agent_group, agent_numbers, result in zip(agent_groups, agent_group_numbers, results):
            self._process_exhibition_results(agent_group, agent_numbers, agent_names, result, log_path)

    def _process_exhibition_results(self, agent_group, agent_numbers, agent_names, result, log_path):
        number_string = '-'.join([str(number) for number in agent_numbers])
        statistics_filepath = f'{log_path}/exhibitionStats{number_string}.txt'
        with open(statistics_filepath, 'w+') as statistics_file:
            statistics_file.truncate(0)
            for player_index, agent in enumerate(agent_group):
                agent_name = agent_names[player_index]

                if isinstance(agent, BaseEvolutionaryAgent):
                    statistics_file.write(f'{agent_name} genotype:\n{agent.genotype}\n')
                else:
                    statistics_file.write(f'{agent_name}:\nNo genotype\n')
                for metric_name, metric_value in result[player_index].items():
                    statistics_file.write(f'{metric_name}:\n{metric_value}\n')
                statistics_file.write('\n')

            if len(result) > len(agent_group):
                statistics_file.write('Additional metrics:\n')
                for metric_name, metric_value in result[-1].items():
                    statistics_file.write(f"{metric_name}:\n")
                    match metric_value:
                        case dict():
                            for key, value in metric_value.items():
                                statistics_file.write(f'{key}:\t{value}\n')
                        case list():
                            for value in metric_value:
                                statistics_file.write(f'{value}\n')
                        case _:
                            statistics_file.write(f'{metric_value}\n')
                    statistics_file.write('\n')

    @staticmethod
    def set_config_value(config: dict, keys: Sequence[str], value: Any, overwrite: bool = False, update: bool = False) -> None:
        current_dict = config
        for key in keys[:-1]:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]

        if keys[-1] in current_dict:
            if update:
                if not isinstance(current_dict[keys[-1]], dict):
                    raise ValueError(f"Key {'.'.join(keys)} is not a dictionary and cannot be updated.")
                current_dict[keys[-1]].update(value)
            elif overwrite:
                current_dict[keys[-1]] = value
            else:
                raise ValueError(f"Key {'.'.join(keys)} already exists, and overwriting was not set.")
        else:
            current_dict[keys[-1]] = value



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
                        log_history: bool = False,
                        add_fitness_modifier: bool = False) -> None:
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
            add_fitness_modifier: If true, the individual's :meth:`BaseObjectiveTracker.get_fitness_modifier`
                result will be added to this metric (e.g. for parsimony pressure).

        """
        metric_configuration: MetricConfiguration = {
            'name': metric_name,
            'is_objective': is_objective,
            'repeat_mode': repeat_mode,
            'log_history': log_history,
            'automatic': True,
            'add_fitness_modifier': add_fitness_modifier
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

        self.register_metric('fitness', fitness_function, is_objective=True, repeat_mode=repeat_mode, log_history=True, add_fitness_modifier=True)