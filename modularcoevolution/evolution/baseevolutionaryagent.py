from modularcoevolution.evolution.baseagent import *

import abc


# TODO: Correct variable case for this and all subclasses (including other repositories)
class BaseEvolutionaryAgent(BaseAgent, metaclass=abc.ABCMeta):

    @classmethod
    @abc.abstractmethod
    def genotype_class(cls):
        pass

    @classmethod
    @abc.abstractmethod
    def genotype_default_parameters(cls):
        pass

    def __init__(self, parameters=None, genotype=None, *args, **kwargs):
        super().__init__(parameters, *args, **kwargs)
        assert genotype is None or isinstance(genotype, self.genotype_class())
        if genotype is not None:
            self.genotype = genotype
        if parameters is None and genotype is None:
            self.genotype = self.genotype_class()(self.genotype_default_parameters())
