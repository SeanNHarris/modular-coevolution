import itertools
from functools import partial
from typing import Type

from evolution import datacollector
from evolution.baseevolutionaryagent import BaseEvolutionaryAgent
from evolution.drivers import coevolutiondriver
from evolution.drivers.coevolutiondriver import EvaluateType, ResultsType
from modularcoevolution.utilities.datacollector import DataCollector

import multiprocessing
import os
import random

TRUNCATE = True
world_kwargs = {}


def load_run_data(run_folder, last_generation = False):
    data_path = f"{run_folder}/data"
    print(f"Reading experiment run data from {data_path}")
    data_collector = DataCollector()
    if last_generation:
        data_collector.load_last_generation(data_path)
    else:
        data_collector.load_directory(data_path)
    return data_collector.data


def load_experiment_data(experiment_folder, last_generation = False, parallel = True):
    run_folders = [folder.path for folder in os.scandir(f"Logs/{experiment_folder}") if folder.is_dir()]

    if parallel:
        try:
            num_processes = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        except KeyError:
            print("Not a Slurm job, using all CPU cores.")
            num_processes = multiprocessing.cpu_count()
        print(f"Running with {num_processes} processes.")
        pool = multiprocessing.Pool(num_processes)
        load_run_data_partial = partial(load_run_data, last_generation=last_generation)
        data_files = pool.map(load_run_data_partial, run_folders)
        pool.close()
        pool.join()
        run_data_files = {_get_run_name(run_folder): data_file for run_folder, data_file in
                          zip(run_folders, data_files)}
    else:
        run_data_files = {_get_run_name(run_folder): load_run_data(run_folder, last_generation) for run_folder in run_folders}
    return run_data_files

def _get_run_name(run_folder):
    return str(run_folder).split("/")[-1]


def create_best_individuals(run_data: datacollector.DataSchema,
                            agent_type_dictionary: dict[str, Type[BaseEvolutionaryAgent]],
                            limit_populations = None,
                            representative_size= -1) -> dict[str, list[BaseEvolutionaryAgent]]:
    experiment_agents = {}
    if limit_populations:
        populations = limit_populations
    else:
        populations = run_data["generations"].keys()
    for population in populations:
        print(f"Processing population \"{population}\"...")
        if population not in experiment_agents:
            experiment_agents[population] = {}

        parameters: coevolutiondriver.ParameterSchema = run_data['experiment']['parameters']
        run_agent_parameters = parameters['generators'][population][1]['agent_parameters']
        run_genotype_parameters = parameters['generators'][population][1]['genotype_parameters']

        last_generation = max(
            int(key) for key in run_data["generations"][population].keys()
        )

        experiment_agents[population] = {}
        population_size = len(run_data["generations"][population][str(last_generation)]["individual_ids"])
        if representative_size < 0:
            population_representative_size = population_size
        else:
            population_representative_size = min(representative_size, population_size)

        elites = []
        if "front members" in run_data["generations"][population][str(last_generation)]["metric_statistics"]:
            # Multiobjective
            front = 0

            while len(elites) < population_representative_size:
                front_elites = \
                run_data["generations"][population][str(last_generation)]["metric_statistics"]["front members"][front]
                amount_needed = population_representative_size - len(elites)
                if amount_needed >= len(front_elites):
                    elites.extend(front_elites)
                else:
                    elites.extend(random.sample(front_elites, amount_needed))
                front += 1
        else:
            # Single objective
            # The member list is sorted the same way it's sorted in evolution's next generation function, which is more or less fitness
            generation_members = run_data["generations"][population][str(last_generation)]["individual_ids"]
            elites.extend(generation_members[:min(population_representative_size, len(generation_members))])

        population_agents = []
        for individual_id in elites:
            agent_type = agent_type_dictionary[population]
            data_genotype_parameters = run_data["individuals"][population][str(individual_id)]["genotype"]
            if "genotype" in data_genotype_parameters:
                data_genotype_parameters.update(data_genotype_parameters["genotype"])
            if "nodeType" in data_genotype_parameters:
                del data_genotype_parameters["nodeType"]
            genotype_parameters = agent_type.genotype_default_parameters(run_agent_parameters)
            deep_update_dictionary(genotype_parameters, run_genotype_parameters)
            deep_update_dictionary(genotype_parameters, data_genotype_parameters)
            agent = agent_type(run_agent_parameters, genotype=agent_type.genotype_class()(genotype_parameters))
            population_agents.append(agent)
        experiment_agents[population] = population_agents
    return experiment_agents

def deep_update_dictionary(dictionary: dict, update: dict) -> None:
    for key, value in update.items():
        if isinstance(value, dict):
            if key not in dictionary:
                dictionary[key] = {}
            deep_update_dictionary(dictionary[key], value)
        else:
            dictionary[key] = value

def compare_populations(representatives: list[list[BaseEvolutionaryAgent]], player_sources: list[int], evaluate: EvaluateType, world_kwargs: dict) -> list[float]:
    """
    Compute the relative fitness between population representatives through round-robin evaluations.
    Args:
        representatives: A list of lists of agents, where each list of agents are the representatives from a population.
        player_sources: Which population to draw each player in the game from.
        evaluate: The evaluation function to be used.
        world_kwargs: The keyword arguments to be passed to the evaluation function.

    Returns:
        The average fitness for each population, as a list.
    """
    players = [representatives[source] for source in player_sources]
    agent_groups = list(itertools.product(*representatives))
    for agent_group in agent_groups:
        results: ResultsType = evaluate(agent_group, **world_kwargs)



def process_experiment(experiment, limit_types=None, all_generations=False, parallel=True, representative_size=0):
    experiment_agents = dict()
    run_folders = [folder.path for folder in os.scandir(f"Logs/{experiment}") if folder.is_dir()]

    if parallel:
        try:
            num_processes = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        except KeyError:
            print("Not a Slurm job, using all CPU cores.")
            num_processes = multiprocessing.cpu_count()
        print(f"Running with {num_processes} processes.")
        pool = multiprocessing.Pool(num_processes)
        data_files = pool.map(load_run_data, run_folders)
        pool.close()
        pool.join()
        run_data_files = {_get_run_name(run_folder): data_file for run_folder, data_file in zip(run_folders, data_files)}
    else:
        run_data_files = dict()
        for run_folder in run_folders:
            run_data_files[_get_run_name(run_folder)] = (load_run_data(run_folder))

    # Main loop
    for run_folder in run_folders:
        run_name = _get_run_name(run_folder)
        print(f"Processing {run_name}...")
        data = run_data_files[run_name]

        if limit_types:
            populations = limit_types
        else:
            populations = data["generations"]
        for population in populations:
            print(f"Processing population \"{population}\"...")
            if population not in experiment_agents:
                experiment_agents[population] = dict()

            last_generation = max([int(key) for key in data["generations"][population].keys()])
            if all_generations:
                generations = range(last_generation + 1)
            else:
                generations = [last_generation]

            experiment_agents[population][run_name] = dict()
            for generation in generations:
                population_size = len(data["generations"][population][str(generation)]["individual_ids"])
                if representative_size == 0:
                    population_representative_size = population_size
                else:
                    population_representative_size = min(representative_size, population_size)

                elites = list()
                if "front members" in data["generations"][population][str(generation)]["metric_statistics"]:
                    # Multiobjective
                    front = 0

                    while len(elites) < population_representative_size:
                        front_elites = data["generations"][population][str(generation)]["metric_statistics"]["front members"][front]
                        amount_needed = population_representative_size - len(elites)
                        if amount_needed >= len(front_elites):
                            elites.extend(front_elites)
                        else:
                            elites.extend(random.sample(front_elites, amount_needed))
                        front += 1
                else:
                    # Single objective
                    # The member list is sorted the same way it's sorted in evolution's next generation function, which is more or less fitness
                    generation_members = data["generations"][population][str(generation)]["individual_ids"]
                    elites.extend(generation_members[:min(population_representative_size, len(generation_members))])

                run_agents = list()
                for individual_id in elites:
                    agent_type = agent_type_dictionary[population]
                    data_parameters = data["individuals"][population][str(individual_id)]["genotype"]
                    if "genotype" in data_parameters:
                        data_parameters.update(data_parameters["genotype"])
                    if "nodeType" in data_parameters:
                        del data_parameters["nodeType"]
                    parameters = agent_type.genotype_default_parameters()
                    parameters.update(data_parameters)
                    agent = agent_type(genotype=agent_type.genotype_class()(parameters))
                    run_agents.append(agent)
                experiment_agents[population][run_name][generation] = run_agents
    return experiment_agents, run_data_files
