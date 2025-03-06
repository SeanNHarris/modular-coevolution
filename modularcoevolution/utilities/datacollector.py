import gzip
import json
import os
from typing import Sequence, Any, TypedDict

import numpy

from modularcoevolution.utilities.specialtypes import GenotypeID, EvaluationID


class ExperimentData(TypedDict):
    """A dictionary storing global data for the experiment."""
    parameters: dict[str, Any]
    """A nested dictionary of parameters for the experiment. No specific format is specified."""
    # master_tournament_objectives: dict[str, Any]


class GenerationData(TypedDict):
    """A dictionary storing the data for a single generation in a generator."""
    individual_ids: Sequence[GenotypeID]
    """An ordered list of the genotype IDs of the individuals in this generation. Order is up to the generator."""
    metric_statistics: dict[str, Any]
    """Aggregate statistics about the metrics of the individuals in this generation. Keyed by metric name."""
    population_metrics: dict[str, Any]
    """Additional metrics about the entire population which aren't directly derived from individual metrics."""


class IndividualData(TypedDict):
    """A dictionary storing the data for a single individual, including its :class:`BaseObjectiveTracker` data."""
    genotype: Any
    """A JSON-serializable representation of the genotype of this individual, which can be used to reconstruct it."""
    evaluation_ids: Sequence[EvaluationID]
    """A list of evaluations this individual participated in."""
    metrics: dict[str, Any]
    """The metrics of this individual, from its objective tracker."""
    metric_statistics: dict[str, Any]
    """The metric statistics of this individual, from its objective tracker."""
    metric_histories: dict[str, Any]
    """The metric histories of this individual, from its objective tracker."""
    parent_ids: Sequence[GenotypeID]
    """The genotype IDs of the parents of this individual, if any."""
    creation_information: str
    """A string describing how this individual was created (e.g. mutation, recombination)."""


class EvaluationData(TypedDict):
    """A dictionary storing the data for a single evaluation."""
    agent_ids: Sequence[GenotypeID]
    """The genotype IDs of the agents in this evaluation, in player order."""
    results: Sequence[dict[str, Any]]
    """A sequence of results for each agent in the evaluation. The format of each result is not specified."""


class DataSchema(TypedDict):
    """Top-level schema for the data stored by a :class:`DataCollector` object."""
    experiment: ExperimentData
    """Stores overall data for the experiment."""
    generations: dict[str, dict[str, GenerationData]]
    """A nested dictionary of generation data, indexed by population name and generation number.
    Note that the generation number is a string, not an integer (but can be converted to an integer)."""
    individuals: dict[str, dict[GenotypeID, IndividualData]]  # Indexed by population name
    """A nested dictionary of individual data, indexed by population name and genotype ID."""
    evaluations: dict[EvaluationID, EvaluationData]
    """A dictionary of all recorded evaluations, indexed by evaluation ID."""


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
    metric_history_logging: bool
    """If True, metric history data from :class:BaseObjectiveTracker will be stored in the log.
    Otherwise, it will be discarded. Disable this to save disk space if you don't need evaluation data."""

    def __init__(
            self,
            split_generations: bool = True,
            compress: bool = True,
            evaluation_logging: bool = True,
            metric_history_logging: bool = True
    ):
        self.split_generations = split_generations
        self.compress = compress
        self.evaluation_logging = evaluation_logging
        self.metric_history_logging = metric_history_logging

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
        if not self.metric_history_logging:
            metric_histories = {}
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

    def load_from_file(self, filename, load_only: Sequence[str] = None) -> None:
        """
        Load data from a previously logged experiment to the current data collector.
        To load data from multiple files (saved using `clear_memory = True`), use :meth:`load_directory` instead.
        Args:
            filename: The path of the file to load from.
            load_only: A list of tables to load from the file. If None, all tables will be loaded.
        """
        try:
            with gzip.open(filename, 'rt', encoding='UTF-8') as log_file:
                self._load_from_file(log_file, load_only=load_only)
        except OSError:
            with open(filename, 'rt', encoding='UTF-8') as log_file:
                self._load_from_file(log_file, load_only=load_only)

    def _load_from_file(self, log_file, load_only: Sequence[str] = None) -> None:
        new_data: DataSchema = json.load(log_file)
        for table in new_data:
            if load_only is not None and table not in load_only:
                continue
            if table in ("generations", "individuals"):
                for population_name in new_data[table]:
                    if population_name not in self.data[table]:
                        self.data[table][population_name] = {}
                    self.data[table][population_name].update(new_data[table][population_name])
            else:
                self.data[table].update(new_data[table])

    def load_directory(self, pathname, load_only: Sequence[str] = None) -> None:
        """
        Load multiple files of data from a previously logged experiment to the current data collector.
        These files must end in a number, and will be loaded in order of that number.
        (This is already the default saving behavior of :class:`.CoevolutionDriver`.)
        Args:
            pathname: The path of the directory to load from.
            load_only: A list of tables to load from the file. If None, all tables will be loaded.
        """
        files = [file for file in os.scandir(pathname) if file.is_file()]
        files.sort(key=lambda file: int("".join(filter(str.isdigit, file.name))))
        for file in files:
            print(f"Loading {file.name}")
            self.load_from_file(file.path, load_only=load_only)

    def load_last_generation(self, pathname, load_only: Sequence[str] = None) -> None:
        """
        Load the last generation of data from a previously logged experiment to the current data collector.
        This is useful if you only need to use the end results of an experiment.
        Args:
            pathname: The path of the directory to load from.
            load_only: A list of tables to load from the file. If None, all tables will be loaded.
        """
        files = [file for file in os.scandir(pathname) if file.is_file()]
        if len(files) == 1:
            # Handles the case where the experiment was saved to a single file.
            self.load_from_file(files[0].path, load_only=load_only)
            return
        files.sort(key=lambda file: int("".join(filter(str.isdigit, file.name))))
        self.load_from_file(files[-1].path, load_only=load_only)

    def load_generation(self, pathname, generation: int, load_only: Sequence[str] = None) -> None:
        """
        Load a specific generation of data from a previously logged experiment to the current data collector.
        Args:
            pathname: The path of the directory to load from.
            generation: The generation number to load.
            load_only: A list of tables to load from the file. If None, all tables will be loaded.
        """
        files = [file for file in os.scandir(pathname) if file.is_file()]
        if len(files) == 1 and generation != 0:
            raise ValueError("Can not load a specific generation from a single-file log.")
        files.sort(key=lambda file: int("".join(filter(str.isdigit, file.name))))
        self.load_from_file(files[generation].path, load_only=load_only)

    def load_generations(self, pathname, generations: Sequence[int], load_only: Sequence[str] = None) -> None:
        """
        Load specific generations of data from a previously logged experiment to the current data collector.
        Args:
            pathname: The path of the directory to load from.
            generations: A list of generation numbers to load.
            load_only: A list of tables to load from the file. If None, all tables will be loaded.
        """
        for generation in generations:
            self.load_generation(pathname, generation, load_only=load_only)

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
