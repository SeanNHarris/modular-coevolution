from typing import Sequence

from matplotlib import pyplot

from modularcoevolution.experiments.baseexperiment import BaseExperiment
from modularcoevolution.generators.archivegenerator import ArchiveGenerator
from modularcoevolution.generators.randomgenotypegenerator import RandomGenotypeGenerator
from modularcoevolution.postprocessing import postprocessingutils
from modularcoevolution.utilities.datacollector import DataSchema


def generate_random(experiment: BaseExperiment, generate_size: int, reduce_size: int = -1):
    generators = experiment.create_random_generators(generate_size, reduce_size)
    return generators


def evaluate_generation(
        experiment: BaseExperiment,
        archive_generators: Sequence[ArchiveGenerator],
        random_generators: Sequence[RandomGenotypeGenerator],
        repeat_evaluations: int = 1,
        parallel: bool = False
) -> None:
    for population_index, archive_population in enumerate(archive_generators):
        # Copy the random generator list and replace one with the archive generator
        generators = list(random_generators)
        generators[population_index] = archive_population
        postprocessingutils.round_robin_evaluation(generators, experiment, repeat_evaluations=repeat_evaluations, parallel=parallel)


def random_opponent_analysis(
        run_data: DataSchema,
        experiment: BaseExperiment,
        representative_size: int = 1,
        random_generate_size: int = 1000,
        random_reduce_size: int = -1,
        repeat_evaluations: int = 1,
        parallel: bool = False
) -> dict[str, list[ArchiveGenerator]]:
    archive_generators = postprocessingutils.load_generational_representatives(run_data, experiment, representative_size=representative_size)
    random_generators = generate_random(experiment, random_generate_size, random_reduce_size)
    result_archives = {population_name: [] for population_name in experiment.population_names()}
    for generation, archive_dictionary in enumerate(archive_generators):
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
