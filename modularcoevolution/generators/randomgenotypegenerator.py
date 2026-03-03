#  Copyright 2026 BONSAI Lab at Auburn University
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

from typing import Type, Any

from modularcoevolution.agents.baseevolutionaryagent import BaseEvolutionaryAgent
from modularcoevolution.generators.basegenotypegenerator import BaseGenotypeGenerator
from modularcoevolution.genotypes.basegenotype import BaseGenotype
from modularcoevolution.genotypes.baseobjectivetracker import BaseObjectiveTracker
from modularcoevolution.utilities.specialtypes import GenotypeID


class RandomGenotypeGenerator[AgentType: BaseEvolutionaryAgent, GenotypeType: BaseGenotype](
    BaseGenotypeGenerator[AgentType, GenotypeType]):
    """A generator that creates a population of randomly-generated genotypes."""

    population: list[BaseObjectiveTracker]
    """The population of randomly-generated individuals."""
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
            **kwargs,
    ):
        super().__init__(agent_class=agent_class, population_name=population_name, **kwargs)

        self.reduce_size = reduce_size
        self.population = []

        population_set = set()
        for _ in range(generate_size):
            default_parameters = self.agent_class.genotype_default_parameters(self.agent_parameters)
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

    def get_individuals_to_test(self) -> list[GenotypeID]:
        return [genotype.id for genotype in self.population]

    def get_representatives_from_generation(self, generation: int, amount: int, force: bool = False) -> list[GenotypeID]:
        if generation <= 0:
            return [individual.id for individual in self.population[:amount]]
        else:
            raise IndexError("RandomGenotypeGenerator is not generational, and only has a generation 0.")

    def end_generation(self) -> None:
        super().end_generation()

    def next_generation(self) -> None:
        super().next_generation()

    def reduce_population(self) -> None:
        if self.reduce_size >= 0:
            self.population = self.population[:self.reduce_size]
            self.genotypes_by_id = {genotype.id: genotype for genotype in self.population}
