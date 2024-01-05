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

    original_ids: dict[GenotypeID, int]
    """A dictionary mapping the genotype ID assigned during analysis to the original genotype ID in the log data.
    This is necessary because genotype IDs are not unique across different runs."""
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
        self.original_ids = original_ids
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
        raise NotImplementedError("ArchiveGenerator does not support get_representatives_from_generation.")

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

    @staticmethod
    def merge_archives(archives: Sequence['ArchiveGenerator']) -> 'ArchiveGenerator':
        population_name = archives[0].population_name
        population = []
        original_ids = {}
        agent_class = archives[0].agent_class
        agent_parameters = archives[0].agent_parameters
        for archive in archives:
            population.extend(archive.population)
            original_ids.update(archive.original_ids)
        merged_archive = ArchiveGenerator(population_name, population, original_ids, agent_class, agent_parameters)
        merged_archive.metric_configurations = archives[0].metric_configurations
        merged_archive.metric_functions = archives[0].metric_functions
        return merged_archive

