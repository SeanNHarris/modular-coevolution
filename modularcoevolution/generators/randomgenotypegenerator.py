from typing import Type, Any

from modularcoevolution.generators.basegenerator import BaseGenerator, AgentType
from modularcoevolution.genotypes.basegenotype import BaseGenotype
from modularcoevolution.genotypes.baseobjectivetracker import BaseObjectiveTracker
from modularcoevolution.utilities.specialtypes import GenotypeID


class RandomGenotypeGenerator(BaseGenerator):
    """A generator that creates a population of randomly-generated genotypes."""

    population: list[BaseObjectiveTracker]
    """The population of randomly-generated individuals."""
    agent_class: Type[AgentType]
    """The class to instantiate agents with."""
    genotype_class: Type[BaseGenotype]
    """The class to instantiate genotypes with, determined by :attr:`agent_class`."""
    agent_parameters: dict[str, Any]
    """The parameters to be sent to the ``__init__`` function of :attr:`agent_class`, other than genotype."""
    genotype_parameters: dict[str, Any]
    """The parameters to be sent to the ``__init__`` function of the :attr:`genotype_class`, in addition to the default
    parameters from :meth:`.BaseEvolutionaryAgent.genotype_default_parameters`. Overwrites any default parameters."""
    reduce_size: int
    """If non-negative, the population will be reduced to this size when (todo) is called."""

    @property
    def population_size(self) -> int:
        return len(self.population)

    def __init__(
            self,
            agent_class: Type[AgentType],
            population_name: str,
            generate_size: int,
            reduce_size: int = -1,
            agent_parameters: dict[str, Any] = None,
            genotype_parameters: dict[str, Any] = None,
    ):
        super().__init__(population_name=population_name)
        self.agent_class = agent_class
        self.genotype_class = agent_class.genotype_class()

        self.reduce_size = reduce_size

        if agent_parameters is None:
            agent_parameters = {}
        self.agent_parameters = agent_parameters
        if genotype_parameters is None:
            genotype_parameters = {}
        self.genotype_parameters = genotype_parameters

        self.population = []
        self.genotypes_by_id = {}

        population_set = set()
        for _ in range(generate_size):
            default_parameters = self.agent_class.genotype_default_parameters(agent_parameters)
            default_parameters.update(self.genotype_parameters)
            unique = False
            individual = None
            while not unique:
                individual = self.genotype_class(default_parameters.copy())
                if hash(individual) not in population_set:
                    unique = True
            self.population.append(individual)
            population_set.add(hash(individual))
            self.genotypes_by_id[individual.id] = individual

    def get_genotype_with_id(self, agent_id: GenotypeID) -> BaseObjectiveTracker:
        return self.genotypes_by_id[agent_id]

    def build_agent_from_id(self, agent_id: GenotypeID, active: bool) -> AgentType:
        if agent_id not in self.genotypes_by_id:
            raise ValueError(f"The agent ID {agent_id} is not present in this generator."
                             f"Ensure the correct generator is being queried.")
        agent = self.agent_class(genotype=self.genotypes_by_id[agent_id], active=active,
                                 parameters=self.agent_parameters)
        return agent

    def get_individuals_to_test(self) -> list[GenotypeID]:
        return [genotype.id for genotype in self.population]

    def get_representatives_from_generation(self, generation: int, amount: int, force: bool = False) -> list[GenotypeID]:
        if generation <= 0:
            return [individual.id for individual in self.population[:amount]]
        else:
            raise IndexError("RandomGenotypeGenerator is not generational, and only has a generation 0.")

    def end_generation(self) -> None:
        pass

    def next_generation(self) -> None:
        pass

    def reduce_population(self) -> None:
        if self.reduce_size >= 0:
            self.population = self.population[:self.reduce_size]
            self.genotypes_by_id = {genotype.id: genotype for genotype in self.population}
