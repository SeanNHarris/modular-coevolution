#  Copyright 2025 BONSAI Lab at Auburn University
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

import statistics
from typing import Any, Sequence

from modularcoevolution.genotypes.baseobjectivetracker import BaseObjectiveTracker
from modularcoevolution.generators.basegenerator import BaseGenerator, AgentType
from modularcoevolution.utilities.specialtypes import GenotypeID


class ArchiveGenerator(BaseGenerator):
    """A generator that loads archived agents from a log to be evaluated during post-experiment analysis."""

    population: list[BaseObjectiveTracker]
    """A list of genotypes to be used as the population"""
    agent_class: type
    """The class of `BaseAgent` associated with the genotypes."""
    agent_parameters: dict[str, Any]
    """A dictionary of parameters to be passed to the agent class when creating agents."""

    current_to_original_ids: dict[GenotypeID, int]
    """A dictionary mapping the genotype ID assigned during analysis to the original genotype ID in the log data.
    This is necessary because genotype IDs are not unique across different runs."""
    original_to_current_ids: dict[int, GenotypeID]
    """The reverse of :attr:`current_to_original_ids`."""
    genotypes_by_id: dict[GenotypeID, BaseObjectiveTracker]
    """A mapping from *current* genotype ID to a genotype with that :attr:`.BaseGenotype.id`."""

    def __init__(self,
                 population_name: str,
                 genotypes: Sequence[BaseObjectiveTracker],
                 original_ids: dict[GenotypeID, int],
                 agent_class: type,
                 agent_parameters: dict[str, Any]):
        super().__init__(population_name)
        # TODO: Move the ID to the BaseObjectiveTracker class to ensure that it always exists.
        self.population = list(genotypes)
        self.genotypes_by_id = {genotype.id: genotype for genotype in genotypes}
        self.current_to_original_ids = original_ids
        self.original_to_current_ids = {original_id: current_id for current_id, original_id in original_ids.items()}
        self.agent_class = agent_class
        self.agent_parameters = agent_parameters

    @property
    def population_size(self) -> int:
        return len(self.population)

    def get_genotype_with_id(self, agent_id: GenotypeID) -> BaseObjectiveTracker:
        return self.genotypes_by_id[agent_id]

    def build_agent_from_id(self, agent_id: GenotypeID, active: bool) -> AgentType:
        if agent_id not in self.genotypes_by_id:
            raise ValueError(f"The agent ID {agent_id} is not present in this generator."
                             f"Ensure the correct generator is being queried.")

        agent = self.agent_class(genotype=self.genotypes_by_id[agent_id], active=active, parameters=self.agent_parameters)
        return agent

    def get_individuals_to_test(self) -> list[GenotypeID]:
        return list(self.genotypes_by_id.keys())

    def get_representatives_from_generation(self, generation: int, amount: int, force: bool = False) -> list[GenotypeID]:
        if generation <= 0:
            return [individual.id for individual in self.population[:amount]]
        else:
            raise IndexError("ArchiveGenerator is not generational, and only has a generation 0.")

    def end_generation(self) -> None:
        """Empty, since this is not a generational generator."""
        pass

    def next_generation(self) -> None:
        """Empty, since this is not a generational generator."""
        pass

    def aggregate_metrics(self):
        metric_statistics = dict()
        for metric, configuration in self.metric_configurations.items():
            sample_metric = self.population[0].metrics[metric]
            if isinstance(sample_metric, (int, float)):
                population_metrics = [individual.metrics[metric] for individual in self.population]
                metric_mean = statistics.mean(population_metrics)
                if len(population_metrics) > 1:
                    metric_standard_deviation = statistics.stdev(population_metrics)
                else:
                    metric_standard_deviation = float('nan')
                metric_minimum = min(population_metrics)
                metric_maximum = max(population_metrics)
                metric_statistics[metric] = {
                    "mean": metric_mean,
                    "standard_deviation": metric_standard_deviation,
                    "minimum": metric_minimum,
                    "maximum": metric_maximum
                }
        return metric_statistics

    def copy(self, clear_metrics: bool = False) -> 'ArchiveGenerator':
        population_copy = [genotype.clone() for genotype in self.population]
        if clear_metrics:
            for genotype in population_copy:
                genotype.reset_objective_tracker()
        copy_current_to_original_ids = {}
        for genotype, clone in zip(self.population, population_copy):
            copy_current_to_original_ids[clone.id] = self.current_to_original_ids[genotype.id]
        generator = ArchiveGenerator(self.population_name, population_copy, copy_current_to_original_ids, self.agent_class, self.agent_parameters)
        generator.metric_configurations = self.metric_configurations
        generator.metric_functions = self.metric_functions
        return generator

    def make_subset(self, subset_ids: Sequence[GenotypeID], clear_metrics: bool = False) -> 'ArchiveGenerator':
        subset_population = [self.genotypes_by_id[genotype_id] for genotype_id in subset_ids]
        subset_copy = [genotype.clone() for genotype in subset_population]
        if clear_metrics:
            for genotype in subset_copy:
                genotype.reset_objective_tracker()
        copy_current_to_original_ids = {}
        for genotype, clone in zip(subset_population, subset_copy):
            copy_current_to_original_ids[clone.id] = self.current_to_original_ids[genotype.id]
        generator = ArchiveGenerator(self.population_name, subset_copy, copy_current_to_original_ids, self.agent_class, self.agent_parameters)
        generator.metric_configurations = self.metric_configurations
        generator.metric_functions = self.metric_functions
        return generator

    @staticmethod
    def merge_archives(archives: Sequence['ArchiveGenerator']) -> 'ArchiveGenerator':
        population_name = archives[0].population_name
        population = []
        original_ids = {}
        agent_class = archives[0].agent_class
        agent_parameters = archives[0].agent_parameters
        for archive in archives:
            population.extend(archive.population)
            original_ids.update(archive.current_to_original_ids)
        merged_archive = ArchiveGenerator(population_name, population, original_ids, agent_class, agent_parameters)
        merged_archive.metric_configurations = archives[0].metric_configurations
        merged_archive.metric_functions = archives[0].metric_functions
        return merged_archive

