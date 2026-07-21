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
__copyright__ = 'Copyright 2026, BONSAI Lab at Auburn University'
__license__ = 'Apache-2.0'

from enum import Enum
from functools import lru_cache
from typing import Any

from modularcoevolution.generators.basegenerator import BaseGenerator, AgentType
from modularcoevolution.generators.fixedgenerator import FixedGenerator
from modularcoevolution.genotypes.baseobjectivetracker import BaseObjectiveTracker
from modularcoevolution.utilities.specialtypes import GenotypeID, EvaluationID


class MultipleGeneratorRole(Enum):
    DEFAULT = 'DEFAULT'
    MANDATORY = 'MANDATORY'


class MultipleGenerator(BaseGenerator[AgentType]):
    """Combines multiple generators into a single generator."""

    generator_roles: dict[BaseGenerator[AgentType], MultipleGeneratorRole]

    @property
    def population_size(self) -> int:
        return sum(generator.population_size for generator in self.generators)

    @property
    def generators(self) -> list[BaseGenerator[AgentType]]:
        return list(self.generator_roles.keys())

    def __init__(self, population_name: str):
        super().__init__(population_name)
        self.generator_roles = {}

    def add_generator(
            self,
            generator: BaseGenerator[AgentType],
            role: MultipleGeneratorRole = MultipleGeneratorRole.DEFAULT
    ) -> None:
        self.generator_roles[generator] = role
        self.metric_configurations.update(generator.metric_configurations)
        self.metric_functions.update(generator.metric_functions)

    def get_tracker_with_id(self, agent_id: GenotypeID) -> BaseObjectiveTracker:
        generator = self.get_generator_with_id(agent_id)
        return generator.get_tracker_with_id(agent_id)

    def _build_agent_from_id(self, agent_id: GenotypeID, active: bool) -> AgentType:
        generator = self.get_generator_with_id(agent_id)
        return generator.build_agent_from_id(agent_id, active)

    def get_individuals_to_test(self) -> list[GenotypeID]:
        return sum([generator.get_individuals_to_test() for generator in self._default_generators()], start=[])

    def get_representatives_from_generation(self, generation: int, amount: int, force: bool = False) -> list[
        GenotypeID]:
        # TODO: This doesn't really make sense if there are multiple default generators without a standard comparison.
        default_generators = self._default_generators()
        representatives = []
        for index, generator in enumerate(default_generators):
            if index == len(default_generators) - 1:
                sub_amount = amount - len(representatives)
            else:
                sub_amount = amount // len(default_generators)
            representatives.extend(generator.get_representatives_from_generation(generation, sub_amount, force))
        return representatives

    def end_generation(self) -> None:
        for generator in self.generators:
            generator.end_generation()

    def next_generation(self) -> None:
        for generator in self.generators:
            generator.next_generation()

    def set_objectives(self, agent_id: GenotypeID, objectives: dict[str, float], opponent: GenotypeID = None,
                       evaluation_id: EvaluationID = None) -> None:
        generator = self.get_generator_with_id(agent_id)
        generator.set_objectives(agent_id, objectives, opponent, evaluation_id)

    def get_mandatory_opponents(self) -> list[GenotypeID]:
        mandatory_opponents = []
        for generator in self.generators:
            mandatory_opponents.extend(generator.get_mandatory_opponents())
            if self.generator_roles[generator] == MultipleGeneratorRole.MANDATORY:
                mandatory_opponents.extend(generator.get_individuals_to_test())
        return mandatory_opponents

    def _default_generators(self) -> list[BaseGenerator[AgentType]]:
        return [generator for generator, role in self.generator_roles.items() if role == MultipleGeneratorRole.DEFAULT]

    def _mandatory_generators(self) -> list[BaseGenerator[AgentType]]:
        return [generator for generator, role in self.generator_roles.items() if role == MultipleGeneratorRole.MANDATORY]

    @lru_cache
    def get_generator_with_id(self, agent_id: GenotypeID) -> BaseGenerator[AgentType]:
        for generator in self.generators:
            try:
                generator.get_tracker_with_id(agent_id)
                return generator
            except ValueError:
                pass

        raise ValueError(f"ID {agent_id} was not found in any sub-generator.")

    @staticmethod
    def from_generator_and_fixed(
            generator: BaseGenerator[AgentType],
            fixed_agent_type: type,
            fixed_agent_parameters: dict[str, Any] = None,
            reuse_fixed_agent: bool = False,
            fixed_count: int = 1
    ) -> 'MultipleGenerator[AgentType]':
        population_name = generator.population_name
        multiple_generator = MultipleGenerator(population_name)
        multiple_generator.add_generator(generator, MultipleGeneratorRole.DEFAULT)

        fixed_generator = FixedGenerator(fixed_agent_type, generator.population_name, fixed_agent_parameters, reuse_fixed_agent, fixed_count)
        multiple_generator.add_generator(fixed_generator, MultipleGeneratorRole.MANDATORY)
        return multiple_generator
