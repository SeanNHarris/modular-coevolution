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

import abc
from typing import Type, Any

from modularcoevolution.agents.baseevolutionaryagent import BaseEvolutionaryAgent
from modularcoevolution.genotypes.geneticprogramming.gptree import GPTree


class BaseGPTreeAgent(BaseEvolutionaryAgent, metaclass=abc.ABCMeta):
    genotype: GPTree

    @classmethod
    def genotype_class(cls) -> Type[GPTree]:
        return GPTree

    def __init__(self, parameters=None, genotype=None, **kwargs):
        super().__init__(parameters=parameters, genotype=genotype, **kwargs)

    def parameter_string(self) -> str:
        return str(self.genotype.get_node_id_list())

    def get_parameters(self) -> dict[str, Any]:
        parameters = self.genotype_default_parameters()
        parameters.update({
            "id_list": self.genotype.get_node_id_list(),
            # TODO: Maybe fixed context should be a class property?
            "fixed_context": self.genotype.fixed_context
        })
        return parameters

    def apply_parameters(self, parameters: dict[str, Any]) -> None:
        if self.genotype is None:
            genotype_parameters = self.genotype_default_parameters(parameters)
            if "id_list" in parameters:
                genotype_parameters["id_list"] = parameters["id_list"].copy()
            if "fixed_context" in parameters:
                genotype_parameters["fixed_context"] = parameters["fixed_context"].copy()
            self.genotype = self.genotype_class()(genotype_parameters)