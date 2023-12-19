import itertools
import json
from functools import partial
from typing import Type

from modularcoevolution.agents.baseevolutionaryagent import BaseEvolutionaryAgent
from modularcoevolution.drivers import coevolutiondriver
from modularcoevolution.drivers.coevolutiondriver import EvaluateType, ResultsType
from modularcoevolution.experiments.baseexperiment import BaseExperiment
from modularcoevolution.generators.archivegenerator import ArchiveGenerator
from modularcoevolution.utilities import parallelutils
from modularcoevolution.utilities.datacollector import DataCollector, DataSchema

import multiprocessing
import os
import random

TRUNCATE = True
world_kwargs = {}


def load_run_experiment_definition(run_folder: str, experiment_type: Type[BaseExperiment]) -> BaseExperiment:
    '''
    Initialize an experiment definition object based on the parameters logged for a given run.

    Args:
        run_folder: The path to the folder for the run.
        experiment_type: The type of experiment used in the run.
            This should match the type logged in the parameters.json file.

    Returns:
        BaseExperiment: The experiment definition object.

    Raises:
        FileNotFoundError: If the configuration file is not found.

    '''
    config_path = f'{run_folder}/parameters.json'
    with open(config_path) as config_file:
        config = json.load(config_file)

    if 'experiment_type' in config['experiment'] and config['experiment']['experiment_type'] != experiment_type.__name__:
        raise ValueError(f'Experiment type in parameters.json ({config['experiment']['experiment_type']}) does not '
                         f'match specified experiment_type ({experiment_type.__name__})')

    experiment = experiment_type(config)
    return experiment


def load_run_data(run_folder, last_generation=False):
    data_path = f'{run_folder}/data'
    print(f'Reading experiment run data from {data_path}')
    data_collector = DataCollector()
    if last_generation:
        data_collector.load_last_generation(data_path)
    else:
        data_collector.load_directory(data_path)
    return data_collector.data


def load_experiment_data(experiment_folder, last_generation=False, parallel=True) -> dict[str, DataSchema]:
    run_folders = [folder.path for folder in os.scandir(f'Logs/{experiment_folder}') if folder.is_dir()]

    if parallel:
        pool = parallelutils.create_pool()
        load_run_data_partial = partial(load_run_data, last_generation=last_generation)
        data_files = pool.map(load_run_data_partial, run_folders)
        pool.close()
        pool.join()
        run_data_files = {_get_run_name(run_folder): data_file for run_folder, data_file in
                          zip(run_folders, data_files)}
    else:
        run_data_files = {_get_run_name(run_folder): load_run_data(run_folder, last_generation) for run_folder in run_folders}
    return run_data_files


def load_experiment_definition(experiment_folder: str, experiment_type: type[BaseExperiment]) -> BaseExperiment:
    run_folders = [folder.path for folder in os.scandir(f'Logs/{experiment_folder}') if folder.is_dir()]
    return load_run_experiment_definition(run_folders[0], experiment_type)


def _get_run_name(run_folder):
    return str(run_folder).split('/')[-1]


def load_best_run_individuals(
    run_data: DataSchema,
    experiment_definition: BaseExperiment,
    limit_populations=None,
    representative_size=-1
) -> dict[str, ArchiveGenerator]:
    """
    Load the best individuals from a run into :class:`.ArchiveGenerator` objects.

    Parameters:
    - run_data: The run data containing information about the generations.
    - experiment_definition: The experiment definition used for this run.
    - limit_populations: If provided, only the specified populations will be loaded.
    - representative_size: How many of the top individuals to load. If -1, all individuals will be loaded.

    Returns:
    - dict[str, ArchiveGenerator]: A dictionary mapping population names to archive generators.

    Raises:
    - ValueError: If a population name is not found in the experiment definition.
    """

    if limit_populations:
        populations_to_load = limit_populations
    else:
        populations_to_load = run_data['generations'].keys()

    experiment_genotypes = {}
    for population_name in populations_to_load:
        if population_name not in experiment_definition.population_names():
            raise ValueError(f'Population name {population_name} not found in experiment definition.')
        agent_type = experiment_definition.agent_types_by_population_name[population_name]

        print(f'Processing population \'{population_name}\'...')

        #parameters: coevolutiondriver.ParameterSchema = run_data['experiment']['parameters']
        #run_agent_parameters = parameters['generators'][population_name][1]['agent_parameters']
        #run_genotype_parameters = parameters['generators'][population_name][1]['genotype_parameters']

        run_agent_parameters = experiment_definition.config['populations'][population_name]['agent']
        run_genotype_parameters = experiment_definition.config['populations'][population_name]['genotype']

        last_generation = max(
            int(key) for key in run_data['generations'][population_name].keys()
        )

        experiment_genotypes[population_name] = {}
        population_size = len(run_data['generations'][population_name][str(last_generation)]['individual_ids'])
        if representative_size < 0:
            population_representative_size = population_size
        else:
            population_representative_size = min(representative_size, population_size)

        elites = []
        if 'front members' in run_data['generations'][population_name][str(last_generation)]['metric_statistics']:
            # Multiobjective
            front = 0

            while len(elites) < population_representative_size:
                front_elites = run_data['generations'][population_name][str(last_generation)]['metric_statistics']['front members'][front]
                amount_needed = population_representative_size - len(elites)
                if amount_needed >= len(front_elites):
                    elites.extend(front_elites)
                else:
                    elites.extend(random.sample(front_elites, amount_needed))
                front += 1
        else:
            # Single objective
            # The member list is sorted the same way it's sorted in evolution's next generation function, which is more or less fitness
            generation_members = run_data['generations'][population_name][str(last_generation)]['individual_ids']
            elites.extend(generation_members[:min(population_representative_size, len(generation_members))])

        population_genotypes = {}
        for individual_id in elites:
            genotype_parameters = agent_type.genotype_default_parameters(run_agent_parameters)
            deep_update_dictionary(genotype_parameters, run_genotype_parameters)
            genotype = agent_type.genotype_class()(genotype_parameters)
            population_genotypes[individual_id] = genotype
        experiment_genotypes[population_name] = population_genotypes
    archive_generators = experiment_definition.create_archive_generators(experiment_genotypes)
    return archive_generators


def deep_update_dictionary(dictionary: dict, update: dict) -> None:
    for key, value in update.items():
        if isinstance(value, dict):
            if key not in dictionary:
                dictionary[key] = {}
            deep_update_dictionary(dictionary[key], value)
        else:
            dictionary[key] = value


def round_robin_evaluation(
    populations: list[ArchiveGenerator],
    experiment_definition: BaseExperiment,
    repeat_evaluations: int = 1,
    parallel: bool = False
):
    """
    Evaluate the populations through round-robin evaluations.
    The resulting objective scores are managed within the archive generators.
    Args:
        populations: A list containing the :class:`.ArchiveGenerator` for each population.
        experiment_definition: The experiment definition to use for evaluation.
        repeat_evaluations: The number of times to repeat each pairing.
        parallel: Whether to evaluate the populations in parallel.
    """
    individuals = [population.get_individuals_to_test() for population in populations]
    agent_groups = []
    for player_ids in itertools.product(*individuals):
        for _ in range(repeat_evaluations):
            agents = []
            for player_id, population in zip(player_ids, populations):
                agents.append(population.build_agent_from_id(player_id))
            agent_groups.append(agents)
    results = experiment_definition.evaluate_all(agent_groups, parallel)
    for agents, result in zip(agent_groups, results):
        for agent, generator in zip(agents, populations):
            generator.submit_evaluation(agent.genotype.id, result)


def compare_populations(populations: dict[str, ArchiveGenerator], experiment_definition: BaseExperiment) -> list[float]:
    """
    Compute the relative fitness between population representatives through round-robin evaluations.
    Args:
        populations: A dictionary mapping population names to :class:`.ArchiveGenerator` objects for each population.
        experiment_definition: The experiment definition to use for comparison.

    Returns:
        The average fitness for each population, as a list.
    """
    pass