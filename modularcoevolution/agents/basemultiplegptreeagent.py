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

import abc
from typing import Type, Any

from modularcoevolution.agents.baseevolutionaryagent import BaseEvolutionaryAgent
from modularcoevolution.genotypes.geneticprogramming.gptree import GPTree, GPTreeParameters
from modularcoevolution.genotypes.multiplegenotype import MultipleGenotype
from modularcoevolution.utilities.dictutils import deep_update_dictionary


class BaseMultipleGPTreeAgent(BaseEvolutionaryAgent, metaclass=abc.ABCMeta):
    genotype: MultipleGenotype[GPTree]

    @classmethod
    @abc.abstractmethod
    def subgenotype_names(cls) -> list[str]:
        """An abstract method allowing the implementer to name the subgenotypes that this agent uses.
        Returns:
            A list of the names of the subgenotypes in the MultipleGenotype, in order.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def subgenotype_parameters(cls) -> dict[str, GPTreeParameters]:
        """An abstract method allowing the implementer to specify the default parameters for the subgenotypes.
        Returns:
            A dictionary mapping the names of the subgenotypes to their default parameters.
        """
        pass
    
    @classmethod
    def genotype_class(cls) -> Type[MultipleGenotype[GPTree]]:
        return MultipleGenotype[GPTree]

    @classmethod
    def genotype_default_parameters(cls, agent_parameters: dict[str, Any] = None) -> dict[str, Any]:
        subgenotypes = {name: GPTree for name in cls.subgenotype_names()}
        subparameters = cls.subgenotype_parameters()
        return {'subgenotypes': subgenotypes, 'subparameters': subparameters}

    def __init__(self, parameters=None, genotype=None, **kwargs):
        super().__init__(parameters=parameters, genotype=genotype, **kwargs)

    def get_parameters(self) -> dict[str, Any]:
        parameters = self.genotype_default_parameters()
        deep_update_dictionary(parameters, self.genotype.get_raw_genotype())
        return parameters

    def apply_parameters(self, parameters: dict[str, Any]) -> None:
        if self.genotype is None:
            genotype_parameters = self.genotype_default_parameters()
            deep_update_dictionary(genotype_parameters, parameters)
            self.genotype = self.genotype_class()(genotype_parameters)
