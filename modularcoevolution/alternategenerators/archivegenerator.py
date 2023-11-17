from typing import Any

from modularcoevolution.evolution.baseobjectivetracker import BaseObjectiveTracker
from modularcoevolution.evolution.generators.basegenerator import BaseGenerator, AgentType
from modularcoevolution.evolution.specialtypes import GenotypeID


class ArchiveGenerator(BaseGenerator):
    """A generator that loads archived agents from a log to be evaluated during post-experiment analysis."""

    population: dict[GenotypeID, BaseObjectiveTracker]
    """A dictionary of genotypes to be used as the population."""
    agent_class: type
    """The class of `BaseAgent` associated with the genotypes."""
    agent_parameters: dict[str, Any]
    """A dictionary of parameters to be passed to the agent class when creating agents."""

    def __init__(self,
                 population_name: str,
                 population: dict[GenotypeID, BaseObjectiveTracker],
                 agent_class: type,
                 agent_parameters: dict[str, Any]):
        super().__init__(population_name)
        self.population = population
        for genotype_id, genotype in self.population.values():
            genotype.id = genotype_id
        self.agent_class = agent_class
        self.agent_parameters = agent_parameters

    @property
    def population_size(self) -> int:
        return len(self.population)

    def get_genotype_with_id(self, agent_id: GenotypeID) -> BaseObjectiveTracker:
        return self.population[agent_id]

    def build_agent_from_id(self, agent_id: GenotypeID, active: bool) -> AgentType:
        if agent_id not in self.population:
            raise ValueError(f"The agent ID {agent_id} is not present in this generator."
                             f"Ensure the correct generator is being queried.")

        agent = self.agent_class(genotype=self.population[agent_id], active=active, parameters=self.agent_parameters)
        return agent

    def get_individuals_to_test(self) -> list[GenotypeID]:
        return list(self.population.keys())

    def get_representatives_from_generation(self, generation: int, amount: int, force: bool = False) -> list[GenotypeID]:
        raise NotImplementedError("ArchiveGenerator does not support get_representatives_from_generation.")

    def end_generation(self) -> None:
        """Empty, since this is not a generational generator."""
        pass

    def next_generation(self) -> None:
        """Empty, since this is not a generational generator."""
        pass