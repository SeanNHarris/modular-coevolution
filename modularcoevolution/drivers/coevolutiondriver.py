from __future__ import annotations

import copy
import itertools
import tomllib
from collections.abc import Callable
from typing import Any, Sequence, TypedDict, Union

from modularcoevolution.generators.basegenerator import BaseGenerator
from modularcoevolution.managers.coevolution import EvolutionEndedException
from modularcoevolution.utilities.datacollector import DataCollector, StringDefaultJSONEncoder
from modularcoevolution.utilities.agenttyperegistry import AgentTypeRegistry

import json
import multiprocessing
import os

from modularcoevolution.experiments.baseexperiment import BaseExperiment


def _apply_args_and_kwargs(function, args, kwargs):
    return function(*args, **kwargs)


ResultsType = Union[dict[str, Any], list[dict[str, Any]]], tuple[list[dict[str, Any]], dict[str, Any]]
EvaluateType = Callable[[Sequence, Any, ...], ResultsType]
"""
    Args:
        agents: An ordered list of agents, e.g. [attacker, defender]
        **kwargs: Any number of keyword arguments that will be passed by ``world_kwargs``.
    Returns:
        One of three result formats:
        #. A dictionary of results, e.g. {'attacker_score': 0.5, 'defender_cost': 0.3}
        #. A list of dictionaries of results, one for each player in the order of ``agents``.
        #. Two values: a list of dictionaries of agent-specific results, and a dictionary of shared additional results.
        The alternative formats are useful when drawing multiple agents from the same population.
"""


class ParameterSchema(TypedDict):
    """A type defining the parameters of a config file."""
    experiment_name: str
    """The name of the experiment, which will be used as the name of the log folder."""
    generators: dict[str, tuple[type[BaseGenerator], dict[str, Any]]]
    """A dict of tuples of generator types and their parameter dictionary, keyed by the population name.
    For example, ``{'predator': (EvolutionGenerator, {'population_size': 100, ...}), 'prey':(NSGAIIGenerator, {'population_size': 50, ...})}``."""
    coevolution_type: type[Coevolution]
    """The type of coevolution to use, e.g ``Coevolution``."""
    coevolution_kwargs: dict[str, Any]
    """The non-generator arguments to pass to the coevolution type, e.g. ``{'num_generations': 100}``.
    The ``agent_generators`` argument should be excluded, as those will be generated based on the :attr:`generators` parameter."""
    world_kwargs: dict[str, Any]
    """The non-agent arguments to pass to the evaluation function as a dictionary."""


class AugmentedParameterSchema(ParameterSchema):
    """A type containing the parameters for a run, with additional fields added by the driver."""
    log_subfolder: str
    """Where to store log files within the logs folder. Includes the name of the experiment and the run number."""


def _flatten_list_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Flatten a list of per-agent results into a disambiguated results dictionary, for logging purposes.

    Args:
        results: A list of per-agent results from an evaluation function.

    Returns:
        A dictionary of results, where entries with duplicate names are disambiguated with the player number.
        For example, ``[{ 'score': 1, 'score': 2 }]`` will become ``{ 'score<0>': 1, 'score<1>': 2 }``.

    """
    flattened_results = {}
    for player, player_results in enumerate(results):
        for result_name, result_value in player_results.items():
            if result_name in flattened_results:
                result_name = f'{result_name}<{player}>'
            flattened_results[result_name] = result_value
    return flattened_results


def _flatten_results(results: ResultsType) -> dict[str, Any]:
    """Flatten the results from an evaluation function into a disambiguated results dictionary, for logging purposes.
    This works for any of the types of results that can be returned by an evaluation function.

    Args:
        results: The results from an evaluation function.

    Returns:
        A dictionary of results, where entries with duplicate names are disambiguated with the player number.
        See ``_flatten_list_results`` for an example.
    """
    if isinstance(results, dict):
        return results
    elif isinstance(results, list):
        return _flatten_list_results(results)
    elif isinstance(results, tuple):
        return _flatten_list_results(results[0]) | results[1]
    else:
        raise TypeError(f'Unexpected results type {type(results)}')


def _get_results_for_player(results: ResultsType, player: int) -> dict[str, Any]:
    """Get the results for a specific player from the results of an evaluation function.
    This works for any of the types of results that can be returned by an evaluation function.

    Args:
        results: The results from an evaluation function.
        player: The index of the player to get results for.

    Returns:
        A dictionary of results for the specified player.
    """
    if isinstance(results, dict):
        return results
    elif isinstance(results, list):
        return results[player]
    elif isinstance(results, tuple):
        return results[0][player] | results[1]
    else:
        raise TypeError(f'Unexpected results type {type(results)}')


class CoevolutionDriver:
    """A class for running coevolutionary experiments which manages a coevolution wrapper and performs evaluations.
    This class will run several coevolutionary runs in sequence, configured by the ``run_amount`` parameter.
    A configuration file must be provided, which specifies the parameters for coevolution."""
    experiment_type: type[BaseExperiment]
    """The :class:`.BaseExperiment` class defining the current experiment."""

    parallel: bool
    """Whether to run the evaluations in parallel using a multiprocessing pool. Disable this for debugging."""
    use_data_collector: bool
    """Whether to use a data collector to store results. This will result in a lot of logged data."""
    run_exhibition: bool
    """Whether to run and log exhibition evaluations between the best individuals of each generation."""

    parameters: list[dict]
    """For each run to be performed, the parameters for that run."""

    def __init__(self, experiment_type: type[BaseExperiment], config_filename: str, run_amount: int = 30, parallel: bool = True, use_data_collector: bool = True, run_exhibition: bool = True, merge_parameters: dict = None):
        """Create a new coevolution driver.

        Args:
            experiment_type: The :class:`.BaseExperiment` class defining the current experiment.
            config_filename: The filename of the configuration file to use.
            run_amount: The number of runs to perform.
            parallel: Whether to run the evaluations in parallel using a multiprocessing pool. Disable this for debugging.
            use_data_collector: Whether to use a data collector to store results. This will result in a lot of logged data.
            run_exhibition: Whether to run and log exhibition evaluations between the best individuals of each generation.
            merge_parameters: A dictionary of parameters to merge into the configuration file.
                Use this for parameters generated programmatically.

        """
        self.experiment_type = experiment_type
        
        self.parallel = parallel
        self.use_data_collector = use_data_collector
        if not self.use_data_collector:
            # TODO: Support disabling the data collector again
            raise Warning("Disabling the data collector is not currently supported.")
        self.run_exhibition = run_exhibition

        if merge_parameters is None:
            merge_parameters = {}
        self.parameters = self._parse_config(config_filename, run_amount, merge_parameters)

        if self.parallel:
            # TODO: Behave differently on Windows and Linux, as this only works on linux
            # TODO: Run this from a static function with a check, because it maybe breaks if you run it more than once (double-check first)
            # Allows data to be shared in global variables across processes with copy-on-write memory if you don't touch it
            try:
                multiprocessing.set_start_method('fork')
            except ValueError:
                print("Warning: this system does not support copy-on-write memory for global variables.")

    def _evaluate_parallel(self, evaluate, evaluation_pool, agent_groups, world_kwargs=None) -> list[dict[str, Any]]:
        """Evaluate a list of agent groups in parallel using a multiprocessing pool and return the results.
        If ``self.parallel`` is False, this will instead evaluate the agents sequentially.

        Args:
            evaluate: The evaluation function to use.
            evaluation_pool: The multiprocessing pool to use for evaluation.
            agent_groups: A list of agent groups to evaluate.
            world_kwargs: The keyword arguments to pass to the evaluation function.

        Returns:
            A list of results from the evaluation function, in the order of the agent groups passed in.

        """
        if world_kwargs is None:
            world_kwargs = {}

        parameters = [(evaluate, [agents], world_kwargs) for agents in agent_groups]

        if self.parallel:
            end_states = evaluation_pool.starmap(_apply_args_and_kwargs, parameters)
        else:
            end_states = list()
            for agents in agent_groups:
                end_states.append(evaluate(agents, **world_kwargs))
        return end_states

    def _exhibition(self, coevolution, amount, world_kwargs, log_path, evaluation_pool) -> None:
        """Run exhibition evaluations between the best individuals of the current generation.

        Args:
            coevolution: The coevolution wrapper to get agents from.
            amount: The number of agents to evaluate from each population.
            world_kwargs: The keyword arguments to pass to the evaluation function.
            log_path: The path to the current log folder.
            evaluation_pool: The multiprocessing pool to use for evaluation.
        """
        generation = coevolution.generation - 1
        generator_order = coevolution.get_generator_order()
        agent_ids = [generator.get_representatives_from_generation(generation, amount) for generator in generator_order]
        agents = [[generator.build_agent_from_id(agent_id, True) for agent_id in agent_ids[player_index]] for player_index, generator in enumerate(generator_order)]
        agent_names = [generator.population_name for generator in generator_order]

        agent_groups = list(itertools.product(*agents))
        agent_numbers = [range(amount) for _ in agents]
        agent_group_numbers = list(itertools.product(*agent_numbers))
        results = self._evaluate_parallel(evaluation_pool, agent_groups, world_kwargs)
        for agent_group, agent_numbers, result in zip(agent_groups, agent_group_numbers, results):
            flat_result = _flatten_results(result)
            number_string = '-'.join([str(number) for number in agent_numbers])
            statistics_filepath = f'{log_path}/exhibitionStats{number_string}.txt'
            with open(statistics_filepath, 'w+') as statistics_file:
                statistics_file.truncate(0)
                for agent, agent_name in zip(agent_group, agent_names):
                    statistics_file.write(f'{agent_name} genotype:\n{agent.genotype}\n')
                for result_name, result_value in flat_result.items():
                    statistics_file.write(f'{result_name}:\n{result_value}\n')

    def _parse_config(self, config_filename: str, run_count: int, merge_parameters: dict) -> list[dict[str, Any]]:
        """Parse a configuration file and return a dictionary of parameters for each run.

        Args:
            config_filename: The filename of the configuration file to parse.
            run_count: The number of runs to perform.
            merge_parameters: A dictionary of parameters to merge into the configuration file's parameters.

        Returns:


        """

        with open(config_filename, 'rb') as config_file:
            base_parameters = tomllib.load(config_file)

        parameters = []
        for i in range(run_count):
            run_parameters = copy.deepcopy(base_parameters)
            run_parameters['log_subfolder'] = f"{base_parameters['log_folder']}/Run {i}"
            for parameter_name, parameter in merge_parameters.items():
                if isinstance(parameter, dict):
                    run_parameters[parameter_name].update(parameter)
                else:
                    run_parameters[parameter_name] = parameter

            # TODO: Allow merge parameters to vary per run, e.g. a list indexed by i
            parameters.append(run_parameters)
        return parameters

    def _parse_config_old(self, config_filename: str, run_count: int, merge_parameters: dict) -> list[dict[str, Any]]:
        """Parse a configuration file and return a dictionary of parameters for each run.

        Args:
            config_filename: The filename of the configuration file to parse.
            run_count: The number of runs to perform.
            merge_parameters: A dictionary of parameters to merge into the configuration file's parameters.

        Returns:
            A list containing the parameters for each run.

        """

        with open(config_filename, 'r') as config_file:
            lines = config_file.read()
            extended_globals = globals()
            extended_globals.update(AgentTypeRegistry.name_lookup)
            config_locals = {}
            exec(lines, globals(), config_locals)  # TODO: Better way of parsing config files
        base_parameters = config_locals

        if 'predator_type' in base_parameters and 'generators' not in base_parameters:
            print("Old-style config file detected. Assuming two-population coevolution with predator and prey populations.")
            predator_kwargs = base_parameters['predator_kwargs']
            predator_kwargs['agent_class'], predator_kwargs['initial_size'], predator_kwargs['children_size'] = base_parameters['predator_args']
            prey_kwargs = base_parameters['prey_kwargs']
            prey_kwargs['agent_class'], prey_kwargs['initial_size'], prey_kwargs['children_size'] = base_parameters['prey_args']
            base_parameters['generators'] = {
                'predator': (base_parameters['predator_type'], base_parameters['predator_kwargs']),
                'prey': (base_parameters['prey_type'], base_parameters['prey_kwargs'])
            }
            base_parameters['coevolution_kwargs']['num_generations'] = base_parameters['coevolution_args'][2]
            base_parameters['coevolution_kwargs']['player_generators'] = (0, 1)

        parameters = []
        for i in range(run_count):
            run_parameters = copy.deepcopy(base_parameters)
            run_parameters['log_subfolder'] = f"{base_parameters['experiment_name']}/Run {i}"
            for parameter_name, parameter in merge_parameters.items():
                if isinstance(parameter, dict):
                    run_parameters[parameter_name].update(parameter)
                else:
                    run_parameters[parameter_name] = parameter

            # TODO: Allow merge parameters to vary per run, e.g. a list indexed by i
            parameters.append(run_parameters)
        return parameters

    def start(self) -> None:
        """Start the experiment and wait for all runs to complete."""
        for parameter_set in self.parameters:
            self._run_experiment(parameter_set)


    def _run_experiment(self, run_parameters: AugmentedParameterSchema) -> None:
        """Run a single experiment.

        Args:
            run_parameters: A dictionary of parameters for the experiment.

        Todo:
            * Store random seeds, propagate to threads.
        """
        log_subfolder = run_parameters['log_subfolder']

        if self.use_data_collector:
            data_collector = DataCollector()
        else:
            data_collector = None

        experiment = self.experiment_type(run_parameters)
        coevolution_manager: Coevolution = experiment.create_experiment()
        world_kwargs = {}  # TODO: Determine if this is still needed.
        
        # for population_name, generator_parameters in generators.items():
        #     generator_type, generator_kwargs = generator_parameters
        #     generator_kwargs['population_name'] = population_name
        #     generator_kwargs['data_collector'] = data_collector
        #     agent_generators.append(generator_type(**generator_kwargs))
        #     if population_name in population_names:
        #         raise ValueError("Error: Multiple identical population names will cause logging conflicts.")
        #     population_names.add(population_name)
        # 
        # coevolution_kwargs['agent_generators'] = agent_generators
        # coevolution_kwargs['data_collector'] = data_collector
        # coevolution_kwargs['log_subfolder'] = log_subfolder
        # coevolution_manager = coevolution_type(**coevolution_kwargs)
        # 
        # for player_index, generator in enumerate(coevolution_manager.get_generator_order()):
        #     for metric_configuration, metric_function in self.metrics[player_index].values():
        #         generator.register_metric(metric_configuration, metric_function)

        if log_subfolder != '' and not log_subfolder.startswith('/'):
            log_path = f'Logs/{log_subfolder}'
        else:
            log_path = f'Logs{log_subfolder}'

        os.makedirs(log_path, exist_ok=True)
        data_collector.set_experiment_parameters(run_parameters)
        with open(f'{log_path}/parameters.txt', 'a+') as parameter_file:
            parameter_file.truncate(0)
            json.dump(data_collector.data, parameter_file, cls=StringDefaultJSONEncoder)

        if self.parallel:
            try:
                num_processes = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            except KeyError:
                print("Not a Slurm job, using all CPU cores.")
                num_processes = multiprocessing.cpu_count()
            print(f"Running with {num_processes} processes.")
            evaluation_pool = multiprocessing.Pool(num_processes)
        else:
            evaluation_pool = None

        while True:
            try:
                while len(coevolution_manager.get_remaining_evaluations()) > 0:
                    evaluations = coevolution_manager.get_remaining_evaluations()
                    agent_groups = [coevolution_manager.build_agent_group(evaluation) for evaluation in evaluations]
                    agent_args = [(*pair,) for pair in agent_groups]

                    end_states = self._evaluate_parallel(experiment.evaluate, evaluation_pool, agent_args, world_kwargs)

                    for i, results in enumerate(end_states):
                        evaluation = evaluations[i]

                        agent_ids = coevolution_manager.evaluation_table[evaluation]
                        results_per_agent = {agent_id: _get_results_for_player(results, index) for index, agent_id in enumerate(agent_ids)}

                        coevolution_manager.submit_evaluation(evaluation, results_per_agent)

                coevolution_manager.next_generation()
                log_filename = f'{log_path}/data/data{coevolution_manager.generation}'
                data_collector.save_to_file(log_filename, True)
                if self.run_exhibition and coevolution_manager.generation % 1 == 0:
                    self._exhibition(coevolution_manager, 3, world_kwargs, log_path, evaluation_pool)

            except EvolutionEndedException:
                if self.run_exhibition:
                    self._exhibition(coevolution_manager, 5, world_kwargs, log_path, evaluation_pool)
                break