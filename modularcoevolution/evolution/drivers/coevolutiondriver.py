from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, Sequence

from modularcoevolution.evolution.wrappers.coevolution import EvolutionEndedException
from modularcoevolution.evolution.datacollector import DataCollector, StringDefaultJSONEncoder
from modularcoevolution.evolution.agenttyperegistry import AgentTypeRegistry

from modularcoevolution.evolution.generators import *
from modularcoevolution.evolution.wrappers import *

import numpy

import json
import multiprocessing
import os


def _apply_args_and_kwargs(function, args, kwargs):
    return function(*args, **kwargs)


EvaluateType = Callable[[Sequence, ...], tuple]
'''
    Args:
        agents: An ordered list of agents, e.g. [attacker, defender]
        world_kwargs: A dictionary of world parameters mostly from the config file
'''

class CoevolutionDriver:
    evaluate: EvaluateType
    metrics: dict[str, Callable]

    events: dict[str, Callable]

    parallel: bool
    use_data_collector: bool
    run_exhibition: bool

    parameters: list[dict]

    def __init__(self, evaluate: EvaluateType, config_filename: str, run_amount: int = 30, parallel: bool = True, use_data_collector: bool = True, run_exhibition: bool = True, merge_parameters: dict = None):
        self.evaluate = evaluate
        self.metrics = dict()

        self.parallel = parallel
        self.use_data_collector = use_data_collector
        if not self.use_data_collector:
            # TODO: Support disabling the data collector again
            raise Warning('Disabling the data collector is not currently supported.')
        self.run_exhibition = run_exhibition

        if merge_parameters is None:
            merge_parameters = dict()
        self.parameters = self.parse_config(config_filename, run_amount, merge_parameters)

        if self.parallel:
            # TODO: Behave differently on Windows and Linux, as this only works on linux
            # TODO: Run this from a static function with a check, because it maybe breaks if you run it more than once (double-check first)
            # Allows data to be shared in global variables across processes with copy-on-write memory if you don't touch it
            try:
                multiprocessing.set_start_method('fork')
            except ValueError:
                print("Warning: this system does not support copy-on-write memory for global variables.")

    def add_metric(self, name: str, function: Callable):
        self.metrics[name] = function

    def evaluate_parallel(self, evaluation_pool, agent_pairs, world_kwargs):
        numpy.seterr(all='ignore')
        parameters = [(self.evaluate, [agents], world_kwargs) for agents in agent_pairs]

        if self.parallel:
            end_states = evaluation_pool.starmap(_apply_args_and_kwargs, parameters)
        else:
            end_states = list()
            for predator, prey in agent_pairs:
                end_states.append(self.evaluate((predator, prey), **world_kwargs))
        return end_states

    def exhibition(self, coevolution, amount, world_kwargs, log_path='Logs'):
        generation = coevolution.generation - 1
        predator_ids = coevolution.attacker_generator.get_representatives_from_generation(generation, amount)
        prey_ids = coevolution.defender_generator.get_representatives_from_generation(generation, amount)
        predators = [coevolution.attacker_generator.build_agent_from_id(id, True) for id in predator_ids]
        preys = [coevolution.defender_generator.build_agent_from_id(id, True) for id in prey_ids]
        for x, predator in enumerate(predators):
            for y, prey in enumerate(preys):
                player_objectives, logs = self.evaluate((predator, prey), **world_kwargs)
                statistics_filepath = f'{log_path}/exhibitionStats{x}-{y}.txt'
                with open(statistics_filepath, 'w+') as statistics_file:
                    statistics_file.truncate(0)
                    statistics_file.write(f'{predator.agent_type_name} genotype:\n{predator.genotype}\n')
                    statistics_file.write(f'{prey.agent_type_name} genotype:\n{prey.genotype}\n')
                    statistics_file.write(f'{predator.agent_type_name} objectives:\n{player_objectives[0]}\n')
                    statistics_file.write(f'{prey.agent_type_name} objectives:\n{player_objectives[1]}\n')
                    for log_type in logs:
                        statistics_file.write(f'{log_type}:\n{logs[log_type]}\n')


    def parse_config(self, config_filename: str, run_count: int, merge_parameters: dict):
        with open(config_filename, 'r') as config_file:
            lines = config_file.read()
            extended_globals = globals()
            extended_globals.update(AgentTypeRegistry.name_lookup)
            config_locals = dict()
            exec(lines, globals(), config_locals)  # TODO: Better way of parsing config files
        base_parameters = config_locals

        parameters = list()
        for i in range(run_count):
            run_parameters = copy.deepcopy(base_parameters)
            run_parameters['log_subfolder'] = '{0}/Run {1}'.format(base_parameters['experiment_name'], i)
            for parameter_name in merge_parameters:
                parameter = merge_parameters[parameter_name]
                if isinstance(parameter, dict):
                    run_parameters[parameter_name].update(parameter)
                else:
                    run_parameters[parameter_name] = parameter

            # TODO: Allow merge parameters to vary per run, e.g. a list indexed by i
            parameters.append(run_parameters)
        return parameters

    def start(self):
        for parameter_set in self.parameters:
            self.run_experiment(**parameter_set)

    # TODO: Store random seeds, propagate to threads
    def run_experiment(self, predator_type, predator_args, predator_kwargs, prey_type, prey_args, prey_kwargs, coevolution_type,
                       coevolution_args, coevolution_kwargs, world_kwargs, excluded_predator_objectives, excluded_prey_objectives, log_subfolder, **kwargs):
        data_collector = DataCollector()
        predator_kwargs['data_collector'] = data_collector
        prey_kwargs['data_collector'] = data_collector

        predator_generator = predator_type(*predator_args, **predator_kwargs)
        prey_generator = prey_type(*prey_args, **prey_kwargs)
        coevolution_args[0] = predator_generator
        coevolution_args[1] = prey_generator
        coevolution_kwargs['data_collector'] = data_collector
        coevolution_kwargs['log_subfolder'] = log_subfolder

        coevolution = coevolution_type(*coevolution_args, **coevolution_kwargs)

        if log_subfolder != '' and not log_subfolder.startswith('/'):
            log_subfolder = '/' + log_subfolder
        log_path = 'Logs' + log_subfolder

        parameters = {'predator_type': predator_type, 'predator_args': predator_args, 'predator_kwargs': predator_kwargs, 'prey_type': prey_type, 'prey_args': prey_args, 'prey_kwargs': prey_kwargs,
                               'coevolution_type': coevolution_type, 'coevolution_args': coevolution_args, 'coevolution_kwargs': coevolution_kwargs, 'world_kwargs': world_kwargs}
        data_collector.set_experiment_parameters(parameters)
        with open(log_path + '/parameters.txt', 'a+') as parameter_file:
            parameter_file.truncate(0)
            json.dump(data_collector.data, parameter_file, cls=StringDefaultJSONEncoder)

        if self.parallel:
            try:
                num_processes = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            except KeyError:
                print('Not a Slurm job, using all CPU cores.')
                num_processes = multiprocessing.cpu_count()
            print(f'Running with {num_processes} processes.')
            evaluation_pool = multiprocessing.Pool(num_processes)
        else:
            evaluation_pool = None

        while True:
            try:
                while len(coevolution.get_remaining_evaluations()) > 0:
                    evaluations = coevolution.get_remaining_evaluations()
                    controller_pairs = [coevolution.build_agent_pair(evaluation) for evaluation in evaluations]
                    agent_args = [(*pair,) for pair in controller_pairs]

                    end_states = self.evaluate_parallel(evaluation_pool, agent_args, world_kwargs)

                    for i, results in enumerate(end_states):
                        player_objectives = results[0]
                        predator_objectives = player_objectives[0]
                        for objective in excluded_predator_objectives:
                            if objective in predator_objectives:
                                del predator_objectives[objective]
                        prey_objectives = player_objectives[1]
                        for objective in excluded_prey_objectives:
                            if objective in prey_objectives:
                                del prey_objectives[objective]
                        coevolution.send_objectives(evaluations[i], predator_objectives, prey_objectives)
                coevolution.next_generation()
                log_filename = f'{log_path}/data/data{coevolution.generation}'
                data_collector.save_to_file(log_filename, True)
                if self.run_exhibition and coevolution.generation % 1 == 0:
                    self.exhibition(coevolution, 3, world_kwargs, log_path)

            except EvolutionEndedException:
                if self.run_exhibition:
                    self.exhibition(coevolution, 5, world_kwargs, log_path)
                break