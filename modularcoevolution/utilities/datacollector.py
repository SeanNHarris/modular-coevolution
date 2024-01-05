import gzip
import json
import os
from typing import Sequence, Any, TypedDict

import numpy

from modularcoevolution.utilities.specialtypes import GenotypeID, EvaluationID


class ExperimentData(TypedDict):
    parameters: dict[str, Any]
    master_tournament_objectives: dict[str, Any]


class GenerationData(TypedDict):
    individual_ids: Sequence[GenotypeID]
    objective_statistics: dict[str, Any]
    population_metrics: dict[str, Any]


class IndividualData(TypedDict):
    genotype: Any
    evaluation_ids: Sequence[EvaluationID]
    metrics: dict[str, Any]
    metric_statistics: dict[str, Any]
    metric_histories: dict[str, Any]
    parent_ids: Sequence[GenotypeID]
    creation_information: str


class EvaluationData(TypedDict):
    agent_ids: Sequence[GenotypeID]
    results: dict[str, Any]


class DataSchema(TypedDict):
    experiment: ExperimentData
    generations: dict[str, dict[int, GenerationData]]
    individuals: dict[str, dict[GenotypeID, IndividualData]]
    evaluations: dict[EvaluationID, EvaluationData]


class DataCollector:
    """
    A class for logging data from an experiment and storing it as a compressed JSON file.

    Todo:
        * Document the exact schema of the data dictionary.
    """

    data: DataSchema
    """A dictionary storing the data collected by this object. Consists of four sub-dictionaries:
    ``experiment``, ``generations``, ``individuals``, and ``evaluations``."""

    split_generations: bool
    """If True, each generation will be stored in a separate file.
    This is useful to limit RAM usage, by storing each generation's data and clearing the logs in RAM."""
    compress: bool
    """If True, the saved JSON file will be compressed with gzip.
    Otherwise, it will be saved as a plaintext JSON file."""
    evaluation_logging: bool
    """If True, evaluation data from :meth:`set_evaluation_data` will be stored in the log.
    Otherwise, it will be discarded. Disable this to save disk space if you don't need evaluation data."""

    def __init__(self, split_generations: bool = True, compress: bool = True, evaluation_logging: bool = True):
        self.split_generations = split_generations
        self.compress = compress
        self.evaluation_logging = evaluation_logging

        self.data = {
            "experiment": {
                "parameters": {},
                "master_tournament_objectives": {},
            },
            "generations": {},
            "individuals": {},
            "evaluations": {},
        }

    def update_experiment(self):
        """
        Called after any change to :attr:`data`. Currently does nothing.
        """
        pass

    def set_experiment_parameters(self, parameters: dict):
        """
        Log the parameters for the experiment.

        Args:
            parameters: A dictionary of parameters for the experiment. No specific format is specified.
        """
        self.data["experiment"]["parameters"] = parameters
        self.update_experiment()

    def set_experiment_master_tournament_objective(self, objective: str, matrix: Any):
        """
        Update the master tournament matrix for a given objective.

        Args:
            objective: The string key of the objective.
            matrix: The master tournament matrix for the objective.
        """
        self.data["experiment"]["master_tournament_objectives"][objective] = matrix
        self.update_experiment()
    
    def set_generation_data(self, population_name: str, generation: int, individual_ids: Sequence[GenotypeID], metric_statistics: dict, population_metrics: dict) -> None:
        """
        Log the data for a completed generation, for a given agent generator.
        Args:
            population_name: The population name associated with this generation.
            generation: The generation number.
            individual_ids: The genotype ids of the individuals in this generation.
            metric_statistics: A dictionary of statistics about the metrics of the individuals in this generation.
            population_metrics: A dictionary of metrics about the entire population which aren't directly derived from individual metrics.
        """
        if population_name not in self.data["generations"]:
            self.data["generations"][population_name] = {}

        generation_data: GenerationData = {
            "individual_ids": individual_ids,
            "metric_statistics": metric_statistics,
            "population_metrics": population_metrics
        }
        self.data["generations"][population_name][generation] = generation_data
        self.update_experiment()

    def set_individual_data(self, population_name: str, id: GenotypeID, genotype: Any, evaluation_ids: Sequence[EvaluationID], metrics: dict, metric_statistics: dict, metric_histories: dict,
                            parent_ids: Sequence[GenotypeID], creation_information: str) -> None:
        """
        Log the data for an individual.
        Args:
            population_name: The population name associated with this individual.
            id: The genotype id of the individual.
            genotype: The raw genotype data for the individual. Given this and the parameters of the experiment, it should be possible to reconstruct the individual.
            evaluation_ids: A collection of evaluation ids this individual participated in.
            metrics: This individual's metrics, from its objective tracker.
            metric_statistics: This individual's metric statistics, from its objective tracker.
            metric_histories: This individual's metric histories, from its objective tracker.
            parent_ids: The genotype ids of the parents of this individual, if any.
            creation_information: A string describing how this individual was created.
        """
        if population_name not in self.data["individuals"]:
            self.data["individuals"][population_name] = {}
        individual_data: IndividualData = {
            "genotype": genotype,
            "evaluation_ids": evaluation_ids,
            "metrics": metrics,
            "metric_statistics": metric_statistics,
            "metric_histories": metric_histories,
            "parent_ids": parent_ids,
            "creation_information": creation_information,
        }
        self.data["individuals"][population_name][id] = individual_data
        self.update_experiment()

    def set_evaluation_data(self, evaluation_id: EvaluationID, agent_ids: Sequence[GenotypeID], results: dict) -> None:
        """
        Log the data for an evaluation.
        Args:
            evaluation_id: The evaluation id.
            agent_ids: The genotype ids of the agents in this evaluation.
            results: The dictionary of results returned by the evaluation function.
        """
        if not self.evaluation_logging:
            return
        self.data["evaluations"][evaluation_id] = {"agent_ids": agent_ids, "results": results}
        self.update_experiment()

    def save_to_file(self, filename) -> None:
        """
        Save the stored data to a JSON file based on :attr:`split_generations` and :attr:`compress`.
        Args:
            filename: The path to the file to save to.
        """
        self._save_to_file(filename, clear_memory=self.split_generations, compress=self.compress)

    def _save_to_file(self, filename, clear_memory=False, compress=True) -> None:
        """
        Save the stored data to a compressed JSON file.
        Args:
            filename: The path to the file to save to. If `clear_memory` is True,
            a generation number will be appended to the filename.
            clear_memory: If True, the individual and evaluation data will be cleared from memory after saving.
            This is useful to limit RAM usage, by storing each generation's data to a separate file.
            compress: If True, compress the JSON file with gzip. Otherwise, save as a plaintext JSON file.
        """
        if clear_memory:
            last_generation = max(int(generation) for population in self.data["generations"].values() for generation in population)
            filename = filename + str(last_generation)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if compress:
            with gzip.open(filename, 'wt+', encoding='UTF-8') as log_file:
                json.dump(self.data, log_file, cls=StringDefaultJSONEncoder)
        else:
            with open(filename, 'wt+', encoding='UTF-8') as log_file:
                json.dump(self.data, log_file, cls=StringDefaultJSONEncoder)

        if clear_memory:
            self.data["individuals"] = {}
            self.data["evaluations"] = {}

    def load_from_file(self, filename) -> None:
        """
        Load data from a previously logged experiment to the current data collector.
        To load data from multiple files (saved using `clear_memory = True`), use :meth:`load_directory` instead.
        Args:
            filename: The path of the file to load from.
        """
        try:
            with gzip.open(filename, 'rt', encoding='UTF-8') as log_file:
                self._load_from_file(log_file)
        except OSError:
            with open(filename, 'rt', encoding='UTF-8') as log_file:
                self._load_from_file(log_file)

    def _load_from_file(self, log_file) -> None:
        new_data: DataSchema = json.load(log_file)
        for table in new_data:
            if table in ("generations", "individuals"):
                for population_name in new_data[table]:
                    if population_name not in self.data[table]:
                        self.data[table][population_name] = {}
                    self.data[table][population_name].update(new_data[table][population_name])
            else:
                self.data[table].update(new_data[table])

    def load_directory(self, pathname) -> None:
        """
        Load multiple files of data from a previously logged experiment to the current data collector.
        These files must end in a number, and will be loaded in order of that number.
        (This is already the default saving behavior of :class:`.CoevolutionDriver`.)
        Args:
            pathname: The path of the directory to load from.
        """
        files = [file for file in os.scandir(pathname) if file.is_file()]
        files.sort(key=lambda file: int("".join(filter(str.isdigit, file.name))))
        for file in files:
            self.load_from_file(file.path)

    def load_last_generation(self, pathname) -> None:
        """
        Load the last generation of data from a previously logged experiment to the current data collector.
        This is useful if you only need to use the end results of an experiment.
        Args:
            pathname: The path of the directory to load from.
        """
        files = [file for file in os.scandir(pathname) if file.is_file()]
        if len(files) == 1:
            # Handles the case where the experiment was saved to a single file.
            self.load_from_file(files[0].path)
            return
        files.sort(key=lambda file: int("".join(filter(str.isdigit, file.name))))
        self.load_from_file(files[-1].path)

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)


class StringDefaultJSONEncoder(json.JSONEncoder):
    """
    A JSON encoder which converts non-JSON-serializable objects to strings.
    """
    def default(self, o):
        try:
            return super().default(o)
        except TypeError:
            if isinstance(o, type):
                return o.__name__
            elif isinstance(o, numpy.ndarray):
                return o.tolist()
            else:
                return str(o)
