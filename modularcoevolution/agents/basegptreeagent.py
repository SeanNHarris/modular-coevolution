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
        return str(self.genotype.getNodeIDList())

    def get_parameters(self) -> dict[str, Any]:
        parameters = self.genotype_default_parameters()
        parameters.update({
            "idList": self.genotype.getNodeIDList(),
            # TODO: Maybe fixed context should be a class property?
            "fixedContext": self.genotype.fixed_context
        })
        return parameters

    def apply_parameters(self, parameters: dict[str, Any]) -> None:
        if self.genotype is None:
            genotype_parameters = self.genotype_default_parameters()
            if "idList" in parameters:
                genotype_parameters["idList"] = parameters["idList"].copy()
            if "fixedContext" in parameters:
                genotype_parameters["fixedContext"] = parameters["fixedContext"].copy()
            self.genotype = self.genotype_class()(genotype_parameters)