from Evolution.BaseAgent import *

import abc


# TODO: Correct variable case for this and all subclasses (including other repositories)
class BaseEvolutionaryAgent(BaseAgent, metaclass=abc.ABCMeta):

    @classmethod
    @abc.abstractmethod
    def genotypeClass(cls):
        pass

    @classmethod
    @abc.abstractmethod
    def genotypeDefaultParameters(cls):
        pass

    def __init__(self, parameters=None, genotype=None, *args, **kwargs):
        super().__init__(parameters, *args, **kwargs)
        assert genotype is None or isinstance(genotype, self.genotypeClass())
        if genotype is not None:
            self.genotype = genotype
        if parameters is None and genotype is None:
            self.genotype = self.genotypeClass()(self.genotypeDefaultParameters())
