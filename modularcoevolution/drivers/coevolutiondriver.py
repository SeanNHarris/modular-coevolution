#  Copyright 2026 BONSAI Lab at Auburn University
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

import argparse
import functools
import logging
import sys
import warnings
from typing import Any, Sequence, TypedDict, Union

from modularcoevolution.managers.coevolution import EvolutionEndedException, Coevolution
from modularcoevolution.postprocessing import postprocessingutils
from modularcoevolution.utilities.datacollector import DataCollector

import json
import multiprocessing
import os

from modularcoevolution.experiments.baseexperiment import BaseExperiment
from modularcoevolution.utilities import dictutils, fileutils, configutils, parallelutils, loggingutils

try:
    import tqdm
except ImportError:
    tqdm = None


def _apply_args_and_kwargs(function, args, kwargs):
    return function(*args, **kwargs)


class CoevolutionDriver:
    """A class for running coevolutionary experiments which manages a coevolution wrapper and performs evaluations.
    This class will run several coevolutionary runs in sequence, configured by the ``run_amount`` parameter.
    A configuration file must be provided, which specifies the parameters for coevolution."""
    experiment_type: type[BaseExperiment]
    """The :class:`.BaseExperiment` class defining the current experiment."""

    parallel: bool
    """Whether to run the evaluations in parallel using a multiprocessing pool. Disable this for debugging."""
    parallel_runs: int
    """If greater than one, parallelization will be across runs rather than across evaluations.
    Requires `parallel` to be True.
    This is often much more memory-intensive than parallelization per-eval, and may not use every CPU core."""
    run_exhibition: bool
    """Whether to run and log exhibition evaluations between the best individuals of each generation."""
    exhibition_rate: int
    """The rate at which to run exhibition evaluations, e.g. every 5 generations."""

    use_data_collector: bool
    """Whether to use a data collector to store results. This will result in a lot of logged data."""

    parameters: list[dict]
    """For each run to be performed, the parameters for that run."""

    def __init__(self,
                 experiment_type: type[BaseExperiment] | None,
                 config_filename: str,
                 run_amount: int = 30,
                 run_start: int = 0,
                 parallel: bool = True,
                 parallel_runs: bool = 1,
                 use_data_collector: bool = True,
                 run_exhibition: bool = True,
                 exhibition_rate: int = 1,
                 merge_parameters: dict = None):
        """Create a new coevolution driver.

        Args:
            experiment_type: The :class:`.BaseExperiment` class defining the current experiment.
                Can be set to None if the experiment type is defined in the configuration file (as `experiment_type`),
                with a value giving the full import path to the experiment class.
            config_filename: The filename of the configuration file to use.
            run_amount: The number of runs to perform.
            run_start: The run number to start at. Runs will end at the number specified by the ``run_amount`` argument.
                Used for resuming experiments.
            parallel: Whether to run the evaluations in parallel using a multiprocessing pool. Disable this for debugging.
            parallel_runs: If greater than 1, parallelization will be across runs rather than across evaluations.
                Requires `parallel` to be True.
                This is often much more memory-intensive than parallelization per-eval, and may not use every CPU core.
            use_data_collector: Whether to use a data collector to store results. This will result in a lot of logged data.
            run_exhibition: Whether to run and log exhibition evaluations between the best individuals of each generation.
            exhibition_rate: The rate at which to run exhibition evaluations, e.g. every 5 generations.
            merge_parameters: A dictionary of parameters to merge into the configuration file.
                Use this for parameters generated programmatically.

        """
        self.experiment_type = experiment_type
        
        self.parallel = parallel
        self.parallel_runs = parallel_runs
        self.use_data_collector = use_data_collector
        if not self.use_data_collector:
            # TODO: Support disabling the data collector again
            raise Warning("Disabling the data collector is not currently supported.")
        self.run_exhibition = run_exhibition
        self.exhibition_rate = exhibition_rate

        if merge_parameters is None:
            merge_parameters = {}

        # Note: we don't pass in merge_parameters here, because we want to merge them separately per-run.
        raw_config = configutils.parse_config(config_filename, experiment_type=self.experiment_type)
        with open(fileutils.get_logs_path() / raw_config['log_folder'] / 'parameters.json', 'w') as parameter_file:
            json.dump(raw_config, parameter_file)

        self.parameters = configutils.generate_run_parameters(raw_config, run_amount, run_start, merge_parameters)

        if self.parallel:
            # TODO: Behave differently on Windows and Linux, as this only works on linux
            # Allows data to be shared in global variables across processes with copy-on-write memory if you don't touch it
            try:
                start_method = multiprocessing.get_start_method(allow_none=True)
                if start_method is None:  # `set_start_method` can not be called twice.
                    multiprocessing.set_start_method('forkserver')
            except ValueError:
                warnings.warn("Warning: this system does not support copy-on-write memory for global variables.")

    def start(self) -> None:
        """Start the experiment and wait for all runs to complete."""
        run_experiment = functools.partial(
            _run_experiment,
            experiment_type=self.experiment_type,
            use_data_collector=self.use_data_collector,
            run_exhibition=self.run_exhibition,
            exhibition_rate=self.exhibition_rate
        )

        try:
            if self.parallel and self.parallel_runs > 1:
                with parallelutils.create_pool(self.parallel_runs) as evaluation_pool:
                    run_experiment_parallel = functools.partial(run_experiment, parallel=False, redirect_output=True)
                    result_iterator = evaluation_pool.map(run_experiment_parallel, self.parameters)

                    if tqdm is not None and len(self.parameters) > 1:
                        result_iterator = tqdm.tqdm(result_iterator, total=len(self.parameters), desc="Running experiment", unit="runs", smoothing=0.0)
                    for result in result_iterator:
                        result.result()  # Make the iterator track completed results.
            else:
                for parameter_set in self.parameters:
                    run_experiment(parameter_set, parallel=self.parallel)

        except KeyboardInterrupt:
            print("Keyboard interrupt received. Ending all runs.")
            raise KeyboardInterrupt

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
        parser.add_argument('-r', '--runs', dest='run_amount', type=int, default=1,
                            help='The number of runs to perform.')
        parser.add_argument('--run-start', dest='run_start', type=int, default=0,
                            help='The run number to start at. Runs will end at the number specified by the --runs argument. Used for resuming experiments.')
        parser.add_argument('-np', '--no-parallel', dest='parallel', action='store_false',
                            help='Disable parallel evaluations.')
        parser.add_argument('-pr', '--parallel-runs', dest='parallel_runs', type=int, default=1,
                            help='If greater than 1, parallelize over runs instead of evaluations. Sets the number of runs to execute in parallel.')
        parser.add_argument('-nd', '--no-data-collector', dest='use_data_collector', action='store_false',
                            help='Disable the data collector.')
        parser.add_argument('-ne', '--no-exhibition', dest='run_exhibition', action='store_false',
                            help='Disable exhibition evaluations.')
        parser.add_argument('--exhibition-rate', dest='exhibition_rate', type=int, default=1,
                            help='The rate at which to run exhibition evaluations, e.g. every 5 generations.')
        return parser


def _run_experiment(
        run_parameters: dict[str, Any],
        experiment_type: type,
        parallel: bool = False,
        use_data_collector: bool = True,
        run_exhibition: bool = True,
        exhibition_rate: int = 1,
        redirect_output: bool = False) -> None:
    """Run a single experiment.

    Args:
        run_parameters: A dictionary of parameters for the experiment.

    Todo:
        * Store random seeds, propagate to threads.
    """
    log_subfolder = run_parameters['log_subfolder']
    if 'logging' in run_parameters:
        logging_parameters = run_parameters['logging']
    else:
        logging_parameters = {}

    if use_data_collector:
        data_collector = DataCollector(**logging_parameters)
    else:
        data_collector = None

    logs_path = fileutils.get_logs_path(can_create=True)
    log_path = logs_path / log_subfolder
    os.makedirs(log_path, exist_ok=True)

    console_path = log_path / "console.txt"
    loggingutils.initialize_logger(log_path=console_path, console_output=not redirect_output, debug=True)
    logger = logging.getLogger(__name__)

    config = dictutils.deep_copy_dictionary(run_parameters)

    run_experiment_type = experiment_type
    if 'experiment_type' in config:
        config_experiment_type = postprocessingutils.get_experiment_type(config)
        if experiment_type is None:
            run_experiment_type = config_experiment_type
        else:
            if experiment_type != config_experiment_type:
                if '__main__' in run_experiment_type.__module__ and run_experiment_type.__name__ == config_experiment_type.__name__:
                    extra_warning = "\nThis is probably because you're running the experiment file instead of running the driver directly."
                else:
                    extra_warning = ""
                warnings.warn(f"The experiment type specified in the configuration file ({config_experiment_type}) does not match the experiment type given to the driver ({run_experiment_type}.{extra_warning}")
    if run_experiment_type is None:
        raise ValueError("No experiment type specified as a parameter or in the configuration file.")

    experiment = run_experiment_type(config)
    coevolution_manager: Coevolution = experiment.create_experiment(data_collector=data_collector)

    with open(f'{log_path}/parameters.json', 'a+') as parameter_file:
        parameter_file.truncate(0)
        json.dump(experiment.config, parameter_file)

    # Do a test evaluation to visualize what the scenario looks like
    # experiment.exhibition(coevolution_manager.agent_generators, 1, log_path, generation=0, parallel=parallel)

    while True:
        try:
            while len(coevolution_manager.get_remaining_evaluations()) > 0:
                evaluations = coevolution_manager.get_remaining_evaluations()
                agent_groups = [coevolution_manager.build_agent_group(evaluation) for evaluation in evaluations]

                end_states = experiment.evaluate_all(agent_groups, parallel=parallel)

                for i, results in enumerate(end_states):
                    evaluation = evaluations[i]

                    agent_ids = coevolution_manager.evaluation_table[evaluation]
                    results_per_agent = {agent_id: results[index] for index, agent_id in enumerate(agent_ids)}

                    coevolution_manager.submit_evaluation(evaluation, results_per_agent)

            coevolution_manager.next_generation()
            if use_data_collector and data_collector.split_generations:
                log_filename = f'{log_path}/data/data'
                data_collector.save_to_file(log_filename)
            if run_exhibition and coevolution_manager.generation % exhibition_rate == (exhibition_rate - 1):
                experiment.exhibition(coevolution_manager.agent_generators, 2, log_path, parallel=parallel)

        except EvolutionEndedException:
            logger.info("Run complete.")
            if run_exhibition:
                experiment.exhibition(coevolution_manager.agent_generators, 3, log_path, parallel=parallel)
            if use_data_collector and not data_collector.split_generations:
                log_filename = f'{log_path}/data/data'
                data_collector.save_to_file(log_filename)
            break
        except KeyboardInterrupt as error:
            raise error


if __name__ == '__main__':
    parser = CoevolutionDriver.create_argument_parser()
    args = parser.parse_args()

    driver = CoevolutionDriver(None, **vars(args))
    driver.start()