import random
from typing import Sequence

from matplotlib import pyplot

from modularcoevolution.experiments.baseexperiment import BaseExperiment
from modularcoevolution.generators.archivegenerator import ArchiveGenerator
from modularcoevolution.generators.randomgenotypegenerator import RandomGenotypeGenerator
from modularcoevolution.postprocessing import postprocessingutils
from modularcoevolution.utilities import parallelutils
from modularcoevolution.utilities.datacollector import DataSchema


def generate_random(experiment: BaseExperiment, generate_size: int, reduce_size: int = -1):
    generators = experiment.create_random_generators(generate_size, reduce_size)
    return generators


def reduce_random_populations(experiment: BaseExperiment, random_generators: Sequence[RandomGenotypeGenerator], team_multiplicity: int = 1, parallel: bool = False):
    # Construct teams for each generator
    player_populations = experiment.player_populations()
    population_teams = []
    for population_index, generator in enumerate(random_generators):
        team_size = player_populations.count(population_index)
        population_teams.append(build_teams(generator, team_size, team_multiplicity))
    # Evaluate the teams
    postprocessingutils.round_robin_team_evaluation(random_generators, population_teams, experiment, parallel=parallel, analysis=True)
    for generator in random_generators:
        generator.reduce_population()


def build_teams(generator: RandomGenotypeGenerator, team_size: int, multiplicity: int = 1):
    teams = []
    player_positions = [generator.get_individuals_to_test() for _ in range(team_size)]
    for _ in range(multiplicity):
        for id_list in player_positions:
            random.shuffle(id_list)
        offset = team_size - 1
        for start_index in range(len(player_positions[0])):
            team = []
            for position in range(len(player_positions)):
                team.append(player_positions[position][(start_index + offset * position) % len(player_positions[position])])
            teams.append(team)
    return teams


def evaluate_generation(
        experiment: BaseExperiment,
        archive_generators: Sequence[ArchiveGenerator],
        random_generators: Sequence[RandomGenotypeGenerator],
        repeat_evaluations: int = 1,
        parallel: bool = False,
) -> None:
    for population_index, archive_population in enumerate(archive_generators):
        # Copy the random generator list and replace one with the archive generator
        generators = list(random_generators)
        generators[population_index] = archive_population
        postprocessingutils.round_robin_evaluation(generators, experiment, repeat_evaluations=repeat_evaluations, parallel=parallel, analysis=True)


def random_opponent_analysis(
        run_data: DataSchema,
        experiment: BaseExperiment,
        representative_size: int = 1,
        random_generate_size: int = 1000,
        random_reduce_size: int = -1,
        repeat_evaluations: int = 1,
        parallel: bool = False,
) -> dict[str, list[ArchiveGenerator]]:
    archive_generators = postprocessingutils.load_generational_representatives(run_data, experiment, representative_size=representative_size)
    random_generators = generate_random(experiment, random_generate_size, random_reduce_size)
    if random_reduce_size > 0 and random_reduce_size < random_generate_size:
        print("Reducing to best random individuals...")
        reduce_random_populations(experiment, random_generators, team_multiplicity=1, parallel=parallel)
    result_archives = {population_name: [] for population_name in experiment.population_names()}
    for generation, archive_dictionary in archive_generators.items():
        print(f"Evaluating generation {generation}...")
        archive_populations = [archive_dictionary[population_name] for population_name in experiment.population_names()]
        evaluate_generation(experiment, archive_populations, random_generators, repeat_evaluations=repeat_evaluations, parallel=parallel)
        for population_name, archive_population in archive_dictionary.items():
            result_archives[population_name].append(archive_population)
    return result_archives


def plot_fitness(archive_generators: list[ArchiveGenerator]):
    fitnesses = []
    for archive in archive_generators:
        fitnesses.append(sum(individual.fitness for individual in archive.population) / archive.population_size)
    pyplot.plot(fitnesses)
    pyplot.show()
