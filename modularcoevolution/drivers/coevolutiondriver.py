from __future__ import annotations

import argparse
import copy
import itertools
import tomllib
from collections.abc import Callable
from typing import Any, Sequence, TypedDict, Union

from modularcoevolution.generators.basegenerator import BaseGenerator
from modularcoevolution.managers.coevolution import EvolutionEndedException, Coevolution
from modularcoevolution.utilities.datacollector import DataCollector, StringDefaultJSONEncoder
from modularcoevolution.utilities.agenttyperegistry import AgentTypeRegistry

import json
import multiprocessing
import os

from modularcoevolution.experiments.baseexperiment import BaseExperiment
from modularcoevolution.utilities.dictutils import deep_copy_dictionary


def _apply_args_and_kwargs(function, args, kwargs):
    return function(*args, **kwargs)

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


class CoevolutionDriver:
    """A class for running coevolutionary experiments which manages a coevolution wrapper and performs evaluations.
    This class will run several coevolutionary runs in sequence, configured by the ``run_amount`` parameter.
    A configuration file must be provided, which specifies the parameters for coevolution."""
    experiment_type: type[BaseExperiment]
    """The :class:`.BaseExperiment` class defining the current experiment."""

    parallel: bool
    """Whether to run the evaluations in parallel using a multiprocessing pool. Disable this for debugging."""
    run_exhibition: bool
    """Whether to run and log exhibition evaluations between the best individuals of each generation."""
    exhibition_rate: int
    """The rate at which to run exhibition evaluations, e.g. every 5 generations."""

    use_data_collector: bool
    """Whether to use a data collector to store results. This will result in a lot of logged data."""
    data_collector_split_generations: bool
    """If True, each generation will be stored in a separate file. See :attr:`.DataCollector.split_generations`."""
    data_collector_compress: bool
    """If True, the data collector will compress the saved files with gzip. See :attr:`.DataCollector.compress`."""
    data_collector_evaluation_logging: bool
    """If True, the data collector will log the results of each evaluation. See :attr:`.DataCollector.evaluation_logging`."""

    parameters: list[dict]
    """For each run to be performed, the parameters for that run."""

    def __init__(self,
                 experiment_type: type[BaseExperiment],
                 config_filename: str,
                 run_amount: int = 30,
                 parallel: bool = True,
                 use_data_collector: bool = True,
                 data_collector_split_generations: bool = True,
                 data_collector_compress: bool = True,
                 data_collector_evaluation_logging: bool = True,
                 run_exhibition: bool = True,
                 exhibition_rate: int = 1,
                 merge_parameters: dict = None):
        """Create a new coevolution driver.

        Args:
            experiment_type: The :class:`.BaseExperiment` class defining the current experiment.
            config_filename: The filename of the configuration file to use.
            run_amount: The number of runs to perform.
            parallel: Whether to run the evaluations in parallel using a multiprocessing pool. Disable this for debugging.
            use_data_collector: Whether to use a data collector to store results. This will result in a lot of logged data.
            data_collector_split_generations: If True, each generation will be stored in a separate file. See :attr:`.DataCollector.split_generations`.
            data_collector_compress: If True, the data collector will compress the saved files with gzip. See :attr:`.DataCollector.compress`.
            data_collector_evaluation_logging: If True, the data collector will log the results of each evaluation. See :attr:`.DataCollector.evaluation_logging`.
            run_exhibition: Whether to run and log exhibition evaluations between the best individuals of each generation.
            exhibition_rate: The rate at which to run exhibition evaluations, e.g. every 5 generations.
            merge_parameters: A dictionary of parameters to merge into the configuration file.
                Use this for parameters generated programmatically.

        """
        self.experiment_type = experiment_type
        
        self.parallel = parallel
        self.use_data_collector = use_data_collector
        self.data_collector_split_generations = data_collector_split_generations
        self.data_collector_compress = data_collector_compress
        self.data_collector_evaluation_logging = data_collector_evaluation_logging
        if not self.use_data_collector:
            # TODO: Support disabling the data collector again
            raise Warning("Disabling the data collector is not currently supported.")
        self.run_exhibition = run_exhibition
        self.exhibition_rate = exhibition_rate

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

        base_parameters['experiment']['experiment_type'] = self.experiment_type.__name__

        parameters = []
        for i in range(run_count):
            run_parameters = deep_copy_dictionary(base_parameters)
            run_parameters['log_subfolder'] = f"{base_parameters['log_folder']}/Run {i}"
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
            data_collector = DataCollector(self.data_collector_split_generations, self.data_collector_compress, self.data_collector_evaluation_logging)
        else:
            data_collector = None

        config = deep_copy_dictionary(run_parameters)
        config['manager']['data_collector'] = data_collector
        config['default']['generator']['data_collector'] = data_collector
        experiment = self.experiment_type(config)
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
        with open(f'{log_path}/parameters.json', 'a+') as parameter_file:
            parameter_file.truncate(0)
            json.dump(run_parameters, parameter_file)

        if self.parallel:
            if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
                num_processes = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
            else:
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

                    end_states = experiment.evaluate_all(agent_groups, parallel=self.parallel, evaluation_pool=evaluation_pool)

                    for i, results in enumerate(end_states):
                        evaluation = evaluations[i]

                        agent_ids = coevolution_manager.evaluation_table[evaluation]
                        results_per_agent = {agent_id: results[index] for index, agent_id in enumerate(agent_ids)}

                        coevolution_manager.submit_evaluation(evaluation, results_per_agent)

                coevolution_manager.next_generation()
                if self.use_data_collector and self.data_collector_split_generations:
                    log_filename = f'{log_path}/data/data'
                    data_collector.save_to_file(log_filename)
                if self.run_exhibition and coevolution_manager.generation % self.exhibition_rate == (self.exhibition_rate - 1):
                    experiment.exhibition(coevolution_manager.agent_generators, 3, log_path, self.parallel, evaluation_pool)

            except EvolutionEndedException:
                print("Run complete.")
                if self.run_exhibition:
                    experiment.exhibition(coevolution_manager.agent_generators, 5, log_path, self.parallel, evaluation_pool)
                if self.use_data_collector and not self.data_collector_split_generations:
                    log_filename = f'{log_path}/data/data'
                    data_collector.save_to_file(log_filename)
                break
            except KeyboardInterrupt:
                print("Keyboard interrupt received. Ending run.")
                if evaluation_pool is not None:
                    evaluation_pool.terminate()
                break
        if evaluation_pool is not None:
            evaluation_pool.close()

    @staticmethod
    def create_argument_parser() -> argparse.ArgumentParser:
        """Create a command-line argument parser for the coevolution driver.
        Additional arguments can be added to the returned parser if needed.
        The resulting arguments can be sent to :meth:`__init__` as keyword arguments.

        Returns:
            An argument parser for the coevolution driver.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('config_filename')
        parser.add_argument('-r', '--runs', dest='run_amount', type=int, default=30)
        parser.add_argument('-np', '--no-parallel', dest='parallel', action='store_false')
        parser.add_argument('-nd', '--no-data-collector', dest='use_data_collector', action='store_false')
        parser.add_argument('--no-split-generations', dest='data_collector_split_generations', action='store_false')
        parser.add_argument('--no-compress', dest='data_collector_compress', action='store_false')
        parser.add_argument('--no-evaluation-logs', dest='data_collector_evaluation_logging', action='store_false')
        parser.add_argument('-ne', '--no-exhibition', dest='run_exhibition', action='store_false')
        parser.add_argument('--exhibition-rate', dest='exhibition_rate', type=int, default=1)
        return parser
