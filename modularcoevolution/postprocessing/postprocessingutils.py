import functools
import importlib
import itertools
import json
import re
import warnings
from functools import partial
from typing import Type, Sequence, TypeVar

from modularcoevolution.experiments.baseexperiment import BaseExperiment
from modularcoevolution.generators.archivegenerator import ArchiveGenerator
from modularcoevolution.generators.basegenerator import BaseGenerator
from modularcoevolution.genotypes.basegenotype import BaseGenotype
from modularcoevolution.utilities import parallelutils
from modularcoevolution.utilities.datacollector import DataCollector, DataSchema, IndividualData

import os

from modularcoevolution.utilities.dictutils import deep_update_dictionary
from modularcoevolution.utilities.fileutils import get_run_paths, resolve_experiment_path
from modularcoevolution.utilities.specialtypes import GenotypeID

TRUNCATE = True
world_kwargs = {}


def load_run_config(run_folder: str, override_parameters: dict = None) -> dict:
    """
    Load the configuration parameters logged for a run from the parameters.json file.

    Args:
        run_folder: The path to the run folder.
        override_parameters: Optional parameters to override the ones loaded from the file.

    Returns:
        A dictionary containing the configuration parameters, as if loaded from the original config file.
    """
    config_path = f'{run_folder}/parameters.json'
    with open(config_path) as config_file:
        config = json.load(config_file)

    if override_parameters:
        deep_update_dictionary(config, override_parameters)

    return config


class UnspecifiedExperimentError(Exception):
    ...


def get_experiment_type(
        config: dict,
        experiment_type_string: str = None,
        default_module_location: str = 'modularcoevolution.experiments'
) -> Type[BaseExperiment]:
    """
    Infer and import the experiment type specified in the given configuration file.
    Includes fallback options for old-style config files that omit this information.

    Args:
        config: The loaded configuration dictionary.
        experiment_type_string: A string specifying the experiment type to use if it is not specified in the config.
        default_module_location: A default module location to use if the experiment type string is not fully qualified.
            This is intended for use with manual input; the config file should always specify the full module path.

    Returns:
        A subclass of :class:`BaseExperiment` corresponding to the config file or the provided string.

    Raises:
        UnspecifiedExperimentError: If the experiment type is not specified in the config or as a parameter.
    """
    if 'experiment_type' in config:
        config_experiment_type = config['experiment_type']
        if experiment_type_string is not None and config_experiment_type != experiment_type_string:
            warnings.warn(f"Using the experiment type specified in the config file ({config_experiment_type}), which differs from the type specified as a parameter ({experiment_type_string}).")
        experiment_type_string = config_experiment_type

    if experiment_type_string is not None:
        if '.' not in experiment_type_string:
            experiment_type_string = f"{default_module_location}.{experiment_type_string.lower()}.{experiment_type_string}"
        experiment_type_module, experiment_type_class = experiment_type_string.rsplit('.', 1)
        try:
            experiment_type = getattr(importlib.import_module(experiment_type_module), experiment_type_class)
        except (ModuleNotFoundError, AttributeError) as error:
            raise UnspecifiedExperimentError(f"Experiment type {experiment_type_string} could not be imported.\n{error}")
    else:
        raise UnspecifiedExperimentError("The config file did not specify an experiment type (and is probably outdated). Please specify an experiment type manually as a parameter to this function.")
    return experiment_type


def load_run_data(
        run_folder: str,
        last_generation: bool = False,
        generations: Sequence[int] = None,
        load_only: Sequence[str] = None
) -> DataSchema:
    """
    Load the logged :class:`DataCollector` data for a specific run.

    Args:
        run_folder: The path to the run folder.
        last_generation: If true, only the last generation will be loaded.
        generations: A list of specific generations to load. Cannot be used with `last_generation`.
        load_only: A list of specific sub-dictionaries to load from the logged data. If None, everything will be loaded.

    Returns:
        A dictionary of loaded data in the format used by the :class:`DataCollector`.
    """
    if generations is not None and last_generation:
        raise ValueError("Can't set both \"last_generation\" and \"generations\" parameters.")

    data_path = os.path.join(run_folder, 'data')
    print(f'Reading experiment run data from {data_path}')
    data_collector = DataCollector()
    if last_generation:
        data_collector.load_last_generation(data_path, load_only=load_only)
    elif generations:
        data_collector.load_generations(data_path, generations, load_only=load_only)
    else:
        data_collector.load_directory(data_path, load_only=load_only)
    return data_collector.data


def load_experiment_data(
        experiment_folder: str,
        last_generation: bool = False,
        generations: Sequence[int] = None,
        load_only: Sequence[str] = None,
        parallel: bool = True,
        run_numbers: Sequence[int] = None
) -> dict[str, DataSchema]:
    """
    Load the logged :class:`DataCollector` data for all runs within a given experiment folder.

    Args:
        experiment_folder: The path to the experiment folder within the logs directory.
        last_generation: If true, only the last generation will be loaded.
        generations: A list of specific generations to load. Cannot be used with `last_generation`.
        load_only: A list of specific sub-dictionaries to load from the logged data. If None, everything will be loaded.
        parallel: If true, each run will be loaded in parallel.
        run_numbers: A list of specific run numbers to load. If None, all runs will be loaded.

    Returns:
        A dictionary mapping run names to loaded data in the format used by the :class:`DataCollector`.
    """
    run_folders = get_run_paths(experiment_folder)

    if run_numbers is not None:
        run_folders = [run_folders[i] for i in run_numbers]

    load_run_data_partial = partial(load_run_data, last_generation=last_generation, generations=generations, load_only=load_only)

    if parallel:
        pool = parallelutils.create_pool()
        data_files = pool.map(load_run_data_partial, run_folders)
        pool.shutdown()
        run_data_files = {_get_run_name(run_folder): data_file for run_folder, data_file in
                          zip(run_folders, data_files)}
    else:
        run_data_files = {_get_run_name(run_folder): load_run_data_partial(run_folder) for run_folder in run_folders}
    return run_data_files


EXPERIMENT = TypeVar('EXPERIMENT', bound=BaseExperiment)


def load_run_experiment_definition(
        run_folder: str,
        experiment_type: Type[EXPERIMENT] = None,
        override_parameters: dict = None
) -> EXPERIMENT:
    """
    Initialize a :class:`BaseExperiment` subclass based on the parameters logged for a given run.

    Args:
        run_folder: The path to the folder for the run.
        experiment_type: The type of experiment used in the run.
            This should match the type logged in the parameters.json file.
            If None, the experiment type will be inferred from the parameters.json file.
        override_parameters: Optional parameters to override the ones loaded from the file.

    Returns:
        The initialized experiment object.

    Raises:
        FileNotFoundError: If the run folder or its `parameters.json` file could not be found.
        UnspecifiedExperimentError: If the experiment type is not specified in the loaded config file or as a parameter.
    """
    config = load_run_config(run_folder, override_parameters)

    if experiment_type is None:
        experiment_type = get_experiment_type(config)

    if 'experiment_type' in config['experiment'] and config['experiment']['experiment_type'] != experiment_type.__name__:
        warnings.warn(f'Experiment type in parameters.json ({config["experiment"]["experiment_type"]}) does not '
                      f'match specified experiment_type ({experiment_type.__name__})')

    experiment = experiment_type(config)
    return experiment


def load_experiment_definition(
        experiment_folder: str,
        experiment_type: type[EXPERIMENT] = None,
        override_parameters: dict = None
) -> EXPERIMENT:
    """
    Initialize a :class:`BaseExperiment` subclass based on the parameters logged for a given experiment.
    Uses the parameters from the first run in the experiment folder.

    Args:
        experiment_folder: The path to the experiment folder within the logs directory.
        experiment_type: The type of experiment used in the run.
            This should match the type logged in the parameters.json file.
            If None, the experiment type will be inferred from the parameters.json file.
        override_parameters: Optional parameters to override the ones loaded from the file.

    Returns:
        The initialized experiment object.

    Raises:
        FileNotFoundError: If the experiment folder or the first run's `parameters.json` file could not be found.
        UnspecifiedExperimentError: If the experiment type is not specified in the loaded config file or as a parameter.
    """
    run_folders = get_run_paths(experiment_folder)
    return load_run_experiment_definition(run_folders[0], experiment_type, override_parameters)


def _get_run_name(run_folder: str) -> str:
    return str(run_folder).split('/')[-1]


def _get_generation_list(run_data: DataSchema) -> list[int]:
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
        limit_populations: If provided, only the specified population names will be loaded.
        representative_size: How many of the top individuals to load. If -1, all individuals will be loaded.
        generation: The generation to load the individuals from. If -1, the last generation will be used.
        load_metrics: Whether to populate individuals' objective trackers with their metric data from the logs.

    Returns:
        A dictionary mapping population names to archive generators containing the loaded individuals.

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

        # Making sure to copy before sorting
        individual_ids = list(list(run_data['generations'][population_name].values())[generation]['individual_ids'])
        metrics = {individual_id: run_data['individuals'][population_name][str(individual_id)]['metrics'] for individual_id in individual_ids}
        metric_list = list(metrics.values())[0].keys()

        elites = []
        if 'front' in metric_list:
            # Multiobjective
            def front_comparator(id_1, id_2):
                front_difference = metrics[id_1]['front'] - metrics[id_2]['front']
                if front_difference != 0:
                    return front_difference
                else:
                    return -(metrics[id_1]['crowding'] - metrics[id_2]['crowding'])

            individual_ids.sort(key=functools.cmp_to_key(front_comparator))
        else:
            # Single objective
            # The member list is sorted the same way it's sorted in evolution's next generation function, which is more or less fitness
            pass
        elites.extend(individual_ids[:min(population_representative_size, len(individual_ids))])

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
    representative_size: int = -1,
    last_generation: bool = False,
    generations: Sequence[int] = None
) -> dict[int, dict[str, ArchiveGenerator]]:
    """
    Load the representatives from each generation of a run into :class:`.ArchiveGenerator` objects.

    Parameters:
        run_data: The run data containing information about the generations.
        experiment_definition: The experiment definition used for this run.
        limit_populations: If provided, only the specified population names will be loaded.
        representative_size: How many of the top individuals to load. If -1, all individuals will be loaded.
        last_generation: If true, only the last generation will be loaded.
        generations: The generations to load representatives from. If None, all generations will be used.
            Cannot be used with `last_generation`.

    Returns:
        For each generation number, a dictionary mapping population names to archive generators containing the loaded individuals.
    """
    if generations is None:
        generations = _get_generation_list(run_data)
    if last_generation:
        generations = [max(generations)]

    result = {}
    for generation in generations:
        result[generation] = load_best_run_individuals(run_data, experiment_definition, limit_populations, representative_size, generation)

    return result


def easy_load_experiment_results(
        experiment_folder: str,
        run_numbers: Sequence[int] = None,
        limit_populations: Sequence[str] = None,
        representative_size: int = -1,
        last_generation: bool = False,
        generations: Sequence[int] = None,
        override_parameters: dict = None
) -> tuple[BaseExperiment, dict[str, DataSchema], dict[str, dict[int, dict[str, ArchiveGenerator]]]]:
    """
    Load the experiment definition, logged data, and population archives from a given experiment folder.

    Args:
        experiment_folder: The path to the experiment folder within the logs directory.
        run_numbers: A list of specific run numbers to load. If None, all runs will be loaded.
        limit_populations: If provided, only the specified population names will be loaded.
        representative_size: How many of the top individuals to load. If -1, all individuals will be loaded.
        last_generation: If true, only the last generation will be loaded.
        generations: A list of specific generations to load. If None, all generations will be used.
            Cannot be used with `last_generation`.
        override_parameters: Optional parameters to override the ones loaded from the file.

    Returns:
        A tuple containing:
        - The initialized experiment object.
        - A dictionary mapping run names to loaded data in the format used by the :class:`DataCollector`.
        - For each run name, for each generation number, a dictionary mapping population names to archive generators containing the loaded individuals.
    """
    experiment = load_experiment_definition(experiment_folder, override_parameters=override_parameters)
    experiment_data = load_experiment_data(
        experiment_folder, run_numbers=run_numbers, last_generation=last_generation,
        generations=generations, parallel=True
    )
    representatives = {run_name: load_generational_representatives(
        run_data, experiment, limit_populations=limit_populations, representative_size=representative_size,
        last_generation=last_generation, generations=generations
    ) for run_name, run_data in experiment_data.items()}

    return experiment, experiment_data, representatives


def round_robin_evaluation(
    populations: Sequence[BaseGenerator],
    experiment_definition: BaseExperiment,
    repeat_evaluations: int = 1,
    parallel: bool = False,
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
    results = experiment_definition.evaluate_all(agent_groups, parallel=parallel, **kwargs)
    for agents, result in zip(agent_groups, results):
        for player_index, agent in enumerate(agents):
            generator = populations[player_populations[player_index]]
            opponents = [opponent.id for opponent in agents if opponent != agent]
            generator.submit_evaluation(agent.id, result[player_index], opponents)
    return results


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

    results = experiment_definition.evaluate_all(agent_groups, parallel=parallel, **kwargs)
    for agents, result in zip(agent_groups, results):
        for player_index, agent in enumerate(agents):
            generator = populations[player_populations[player_index]]
            opponents = [opponent.id for opponent in agents if opponent != agent]
            generator.submit_evaluation(agent.id, result[player_index], opponents)


def round_robin_team_evaluation(
    populations: Sequence[BaseGenerator],
    population_teams: list[list[list[GenotypeID]]],
    experiment_definition: BaseExperiment,
    repeat_evaluations: int = 1,
    parallel: bool = False,
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
    results = experiment_definition.evaluate_all(agent_groups, parallel=parallel, **kwargs)
    for agents, result in zip(agent_groups, results):
        for player_index, agent in enumerate(agents):
            generator = populations[player_populations[player_index]]
            opponents = [opponent.id for opponent in agents if opponent != agent]
            generator.submit_evaluation(agent.id, result[player_index], opponents)


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


def compare_experiments(
        experiment_1_populations: list[dict[str, ArchiveGenerator]],
        experiment_2_populations: list[dict[str, ArchiveGenerator]],
        experiment_definition: BaseExperiment,
        repeat_evaluations: int = 1,
        parallel: bool = False
) -> list[list[dict[str, float]]]:
    """
    Compare two experiments for a symmetric game by playing the representatives from all their runs against each other
    in a round-robin tournament.

    Args:
        experiment_1_populations: Each run's populations from the first experiment.
            A list of outputs from :func:`load_best_run_individuals` per run.
        experiment_2_populations: Each run's populations from the second experiment.
            A list of outputs from :func:`load_best_run_individuals` per run.
        experiment_definition: The experiment definition to use for comparison.
        repeat_evaluations: The number of times to repeat each pairing.
        parallel: Whether to evaluate the populations in parallel.

    Returns:
        A tuple containing the fitnesses per run for each experiment.
    """

    population_names = experiment_definition.population_names()
    experiment_combined_archives = []
    for populations in [experiment_1_populations, experiment_2_populations]:
        population_archives = {population_name: [] for population_name in population_names}
        for run_populations in populations:
            for population_name in population_names:
                population_archives[population_name].append(run_populations[population_name])
        combined_archives = {population_name: ArchiveGenerator.merge_archives(archives)
                             for population_name, archives in population_archives.items()}
        experiment_combined_archives.append(combined_archives)

    if len(experiment_definition.population_names()) > 2:
        raise NotImplementedError("This function currently only supports experiments configured with up to two populations.")

    round_robin_1 = [experiment_combined_archives[0][population_names[0]], experiment_combined_archives[1][population_names[1]]]
    round_robin_2 = [experiment_combined_archives[1][population_names[0]], experiment_combined_archives[0][population_names[1]]]
    round_robin_evaluation(round_robin_1, experiment_definition, repeat_evaluations, parallel=parallel)
    round_robin_evaluation(round_robin_2, experiment_definition, repeat_evaluations, parallel=parallel)

    experiment_metrics = []
    for experiment in [experiment_1_populations, experiment_2_populations]:
        run_metrics = []
        for run in experiment:
            population_metrics = {}
            for population_name in population_names:
                archive = run[population_name]
                metrics = archive.aggregate_metrics()
                population_metrics[population_name] = metrics
            run_metrics.append(population_metrics)
        experiment_metrics.append(run_metrics)
    return experiment_metrics


def identify_last_generation(pathname: str) -> int:
    """
    Identify the last generation from a path to a run's log directory.

    Args:
        pathname: The path to the run's log directory.

    Returns:
        int: The last generation number.
    """
    data_path = os.path.join(pathname, 'data')
    files = [file for file in os.scandir(data_path) if file.is_file()]
    if len(files) == 1:
        raise ValueError("This function is not supported for run data saved to a single file (i.e. with split_generations=False).")
    return len(files) - 1
