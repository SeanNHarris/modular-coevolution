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
            genotype_parameters = self.genotype_default_parameters()
            if "id_list" in parameters:
                genotype_parameters["id_list"] = parameters["id_list"].copy()
            if "fixed_context" in parameters:
                genotype_parameters["fixed_context"] = parameters["fixed_context"].copy()
            self.genotype = self.genotype_class()(genotype_parameters)