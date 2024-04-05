import functools
import itertools
import json
import re
import warnings
from functools import partial
from typing import Type, Sequence

from modularcoevolution.agents.baseevolutionaryagent import BaseEvolutionaryAgent
from modularcoevolution.drivers import coevolutiondriver
from modularcoevolution.experiments.baseexperiment import BaseExperiment
from modularcoevolution.generators.archivegenerator import ArchiveGenerator
from modularcoevolution.generators.basegenerator import BaseGenerator
from modularcoevolution.genotypes.basegenotype import BaseGenotype
from modularcoevolution.genotypes.baseobjectivetracker import BaseObjectiveTracker
from modularcoevolution.utilities import parallelutils
from modularcoevolution.utilities.datacollector import DataCollector, DataSchema, IndividualData

import multiprocessing
import os
import random

from modularcoevolution.utilities.dictutils import deep_update_dictionary
from modularcoevolution.utilities.specialtypes import GenotypeID

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
        warnings.warn(f'Experiment type in parameters.json ({config["experiment"]["experiment_type"]}) does not '
                      f'match specified experiment_type ({experiment_type.__name__})')

    experiment = experiment_type(config)
    return experiment


def load_run_data(run_folder, last_generation=False, load_only: Sequence[str] = None) -> DataSchema:
    data_path = f'{run_folder}/data'
    print(f'Reading experiment run data from {data_path}')
    data_collector = DataCollector()
    if last_generation:
        data_collector.load_last_generation(data_path, load_only=load_only)
    else:
        data_collector.load_directory(data_path, load_only=load_only)
    return data_collector.data


def load_experiment_data(experiment_folder, last_generation=False, load_only: Sequence[str] = None, parallel=True, run_numbers=None) -> dict[str, DataSchema]:
    run_folders = [folder.path for folder in os.scandir(f'Logs/{experiment_folder}') if folder.is_dir()]
    # Get folders of the form 'Run #', sorted by their number
    run_folders = [folder for folder in run_folders if re.match(r'.*Run \d+', folder)]
    run_folders.sort(key=lambda folder: int(folder.split(' ')[-1]))

    if run_numbers is not None:
        run_folders = [run_folders[i] for i in run_numbers]

    if parallel:
        pool = parallelutils.create_pool()
        load_run_data_partial = partial(load_run_data, last_generation=last_generation, load_only=load_only)
        data_files = pool.map(load_run_data_partial, run_folders)
        pool.close()
        pool.join()
        run_data_files = {_get_run_name(run_folder): data_file for run_folder, data_file in
                          zip(run_folders, data_files)}
    else:
        run_data_files = {_get_run_name(run_folder): load_run_data(run_folder, last_generation, load_only) for run_folder in run_folders}
    return run_data_files


def load_experiment_definition(experiment_folder: str, experiment_type: type[BaseExperiment]) -> BaseExperiment:
    run_folders = [folder.path for folder in os.scandir(f'Logs/{experiment_folder}') if folder.is_dir()]
    #return load_run_experiment_definition(run_folders[0], experiment_type)
    run_folders = [folder for folder in run_folders if re.match(r'.*Run \d+', folder)]
    return load_run_experiment_definition(run_folders[0], experiment_type)


def _get_run_name(run_folder):
    return str(run_folder).split('/')[-1]


def _get_generation_list(run_data: DataSchema):
    population_name = run_data['generations'].keys().__iter__().__next__()
    return [int(generation) for generation in run_data['generations'][population_name].keys()]


def load_best_run_individuals(
    run_data: DataSchema,
    experiment_definition: BaseExperiment,
    limit_populations: Sequence[str] = None,
    representative_size: int = -1,
    generation: int = -1,
    load_metrics: bool = False
) -> dict[str, ArchiveGenerator]:
    """
    Load the best individuals from a run into :class:`.ArchiveGenerator` objects.

    Parameters:
        run_data: The run data containing information about the generations.
        experiment_definition: The experiment definition used for this run.
        limit_populations: If provided, only the specified populations will be loaded.
        representative_size: How many of the top individuals to load. If -1, all individuals will be loaded.
        generation: The generation to load the individuals from. If -1, the last generation will be used.
        load_metrics: Whether to populate individuals' objective trackers with their metric data from the logs.

    Returns:
        dict[str, ArchiveGenerator]: A dictionary mapping population names to archive generators.

    Raises:
        ValueError: If a population name is not found in the experiment definition.
    """

    if limit_populations:
        populations_to_load = limit_populations
    else:
        populations_to_load = list(run_data['generations'].keys())

    if generation < 0:
        generation = max(_get_generation_list(run_data))

    experiment_genotypes = {}
    original_ids = {}
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

        experiment_genotypes[population_name] = {}
        population_size = len(run_data['generations'][population_name][str(generation)]['individual_ids'])
        if representative_size < 0:
            population_representative_size = population_size
        else:
            population_representative_size = min(representative_size, population_size)

        elites = []
        if 'front members' in run_data['generations'][population_name][str(generation)]['metric_statistics']:
            # Multiobjective
            front = 0

            while len(elites) < population_representative_size:
                front_elites = run_data['generations'][population_name][str(generation)]['metric_statistics']['front members'][front]
                amount_needed = population_representative_size - len(elites)
                if amount_needed >= len(front_elites):
                    elites.extend(front_elites)
                else:
                    elites.extend(random.sample(front_elites, amount_needed))
                front += 1
        else:
            # Single objective
            # The member list is sorted the same way it's sorted in evolution's next generation function, which is more or less fitness
            generation_members = run_data['generations'][population_name][str(generation)]['individual_ids']
            elites.extend(generation_members[:min(population_representative_size, len(generation_members))])

        population_genotypes = []
        population_original_ids = {}
        for individual_id in elites:
            individual_data: IndividualData = run_data['individuals'][population_name][str(individual_id)]
            genotype_parameters = individual_data['genotype']
            default_genotype_parameters = agent_type.genotype_default_parameters(run_agent_parameters)
            deep_update_dictionary(genotype_parameters, default_genotype_parameters)
            deep_update_dictionary(genotype_parameters, run_genotype_parameters)
            genotype: BaseGenotype = agent_type.genotype_class()(genotype_parameters)
            population_genotypes.append(genotype)
            population_original_ids[genotype.id] = individual_id

            if load_metrics:
                genotype.metrics = individual_data['metrics']
                genotype.metric_statistics = individual_data['metric_statistics']
                genotype.metric_histories = individual_data['metric_histories']

        experiment_genotypes[population_name] = population_genotypes
        original_ids[population_name] = population_original_ids
    archive_generators = experiment_definition.create_archive_generators(experiment_genotypes, original_ids)
    return archive_generators


def load_generational_representatives(
    run_data: DataSchema,
    experiment_definition: BaseExperiment,
    limit_populations: Sequence[str] = None,
    representative_size: int = -1
) -> list[dict[str, ArchiveGenerator]]:
    """
    Load the representatives from each generation of a run into :class:`.ArchiveGenerator` objects.

    Parameters:
        run_data: The run data containing information about the generations.
        experiment_definition: The experiment definition used for this run.
        limit_populations: If provided, only the specified populations will be loaded.
        representative_size: How many of the top individuals to load. If -1, all individuals will be loaded.

    Returns:
        For each generation, a dictionary mapping population names to archive generators.
    """
    generations = _get_generation_list(run_data)
    result = []
    for generation in generations:
        result.append(load_best_run_individuals(run_data, experiment_definition, limit_populations, representative_size, generation))

    return result


def round_robin_evaluation(
    populations: Sequence[BaseGenerator],
    experiment_definition: BaseExperiment,
    repeat_evaluations: int = 1,
    parallel: bool = False,
    evaluation_pool = None,
    **kwargs
):
    """
    Evaluate the populations through round-robin evaluations.
    The resulting objective scores are managed within the archive generators.
    Args:
        populations: A list containing the :class:`.ArchiveGenerator` for each population.
        experiment_definition: The experiment definition to use for evaluation.
        repeat_evaluations: The number of times to repeat each pairing.
        parallel: Whether to evaluate the populations in parallel.
        kwargs: Additional keyword arguments to pass to the evaluation function.
    """
    player_populations = experiment_definition.player_populations()
    individual_ids = [population.get_individuals_to_test() for population in populations]
    # TODO: Build a new agent for each evaluation in case the agent has state.
    population_agents = [[population.build_agent_from_id(individual_id, True)
                          for individual_id in individual_ids[population_index]]
                         for population_index, population in enumerate(populations)]
    agents = [population_agents[population_index] for population_index in player_populations]

    agent_groups = list(itertools.product(*agents))
    # Don't evaluate games with duplicate agents.
    # The same pair of agents can still be evaluated multiple times in different player orders.
    # agent_groups = [agent_group for agent_group in agent_groups if len(set(agent_group)) == len(agent_group)]
    agent_groups = agent_groups * repeat_evaluations
    results = experiment_definition.evaluate_all(agent_groups, parallel=parallel, evaluation_pool=evaluation_pool, **kwargs)
    for agents, result in zip(agent_groups, results):
        for player_index, agent in enumerate(agents):
            generator = populations[player_populations[player_index]]
            opponents = [opponent.genotype.id for opponent in agents if opponent != agent]
            generator.submit_evaluation(agent.genotype.id, result[player_index], opponents)


def round_robin_homogenous_evaluation(
    populations: Sequence[BaseGenerator],
    experiment_definition: BaseExperiment,
    repeat_evaluations: int = 1,
    parallel: bool = False,
    evaluation_pool = None,
    **kwargs
):
    """
    Evaluate the populations through round-robin evaluations,
    using the same individual for all players from the same population.
    This results in `O(len(populations)**2)` evaluations, less than :func:`round_robin_evaluation`,
    but won't consider heterogeneous teams.
    The resulting objective scores are managed within the archive generators.
    Args:
        populations: A list containing the :class:`.ArchiveGenerator` for each population.
        experiment_definition: The experiment definition to use for evaluation.
        repeat_evaluations: The number of times to repeat each pairing.
        parallel: Whether to evaluate the populations in parallel.
        kwargs: Additional keyword arguments to pass to the evaluation function.
    """
    player_populations = experiment_definition.player_populations()
    individual_ids = [population.get_individuals_to_test() for population in populations]
    # TODO: Build a new agent for each evaluation in case the agent has state.
    population_agents = [[population.build_agent_from_id(individual_id, True)
                          for individual_id in individual_ids[population_index]]
                         for population_index, population in enumerate(populations)]
    population_groups = list(itertools.product(*population_agents))
    agent_groups = []
    for group in population_groups:
        agent_group = [group[player_index] for player_index in player_populations]
        agent_groups.extend([agent_group] * repeat_evaluations)

    results = experiment_definition.evaluate_all(agent_groups, parallel=parallel, evaluation_pool=evaluation_pool, **kwargs)
    for agents, result in zip(agent_groups, results):
        for player_index, agent in enumerate(agents):
            generator = populations[player_populations[player_index]]
            opponents = [opponent.genotype.id for opponent in agents if opponent != agent]
            generator.submit_evaluation(agent.genotype.id, result[player_index], opponents)


def round_robin_team_evaluation(
    populations: Sequence[BaseGenerator],
    population_teams: list[list[list[GenotypeID]]],
    experiment_definition: BaseExperiment,
    repeat_evaluations: int = 1,
    parallel: bool = False,
    evaluation_pool = None,
    **kwargs
):
    """
    Evaluate the populations through round-robin evaluations, using fixed teams.
    The resulting objective scores are managed within the archive generators.
    Args:
        populations: A list containing the :class:`.ArchiveGenerator` for each population.
        population_teams: For each generator, a list of teams, where each team is a list of genotype IDs.
            A team is a list of individuals from a population that are always evaluated together.
        experiment_definition: The experiment definition to use for evaluation.
        repeat_evaluations: The number of times to repeat each pairing.
        parallel: Whether to evaluate the populations in parallel.
        kwargs: Additional keyword arguments to pass to the evaluation function.
    """
    player_populations = experiment_definition.player_populations()
    agent_table = {}
    for population in populations:
        for individual in population.get_individuals_to_test():
            agent_table[individual] = population.build_agent_from_id(individual, True)

    groups = list(itertools.product(*population_teams))
    for index, group in enumerate(groups):
        groups[index] = functools.reduce(list.__add__, group)
    groups = groups * repeat_evaluations
    agent_groups = [[agent_table[individual] for individual in group] for group in groups]
    results = experiment_definition.evaluate_all(agent_groups, parallel=parallel, evaluation_pool=evaluation_pool, **kwargs)
    for agents, result in zip(agent_groups, results):
        for player_index, agent in enumerate(agents):
            generator = populations[player_populations[player_index]]
            opponents = [opponent.genotype.id for opponent in agents if opponent != agent]
            generator.submit_evaluation(agent.genotype.id, result[player_index], opponents)


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


def compare_experiments_symmetric(experiment_1_populations: list[ArchiveGenerator],
                                  experiment_2_populations: list[ArchiveGenerator],
                                  experiment_definition: BaseExperiment,
                                  repeat_evaluations: int = 1,
                                  parallel: bool = False) -> list[list[float]]:
    """
    Compare two experiments for a symmetric game by playing the representatives from all their runs against each other
    in a round-robin tournament.

    Args:
        experiment_1_populations: The populations from the first experiment.
        experiment_2_populations: The populations from the second experiment.
        experiment_definition: The experiment definition to use for comparison.
        repeat_evaluations: The number of times to repeat each pairing.
        parallel: Whether to evaluate the populations in parallel.

    Returns:
        A tuple containing the fitnesses per run for each experiment.
    """

    if len(experiment_definition.population_names()) > 1:
        raise ValueError('This function only supports experiments configured with a single population.')
    combined_archive = ArchiveGenerator.merge_archives(experiment_1_populations + experiment_2_populations)
    round_robin_evaluation([combined_archive], experiment_definition, repeat_evaluations, parallel=parallel)

    experiment_metrics = []
    for experiment in [experiment_1_populations, experiment_2_populations]:
        run_metrics = []
        for run in experiment:
            metrics = run.aggregate_metrics()
            run_metrics.append(metrics)
        experiment_metrics.append(run_metrics)
    return experiment_metrics
