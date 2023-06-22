from modularcoevolution.evolution.generators.baseevolutionarygenerator import BaseEvolutionaryGenerator

import cma


class CMAGenerator(BaseEvolutionaryGenerator):
    def __init__(self, *args, parameter_length=None, **kwargs):
        super().__init__(*args, **kwargs)

        if parameter_length is None:
            raise ValueError("parameter_length must be provided.")

        self.es = cma.CMAEvolutionStrategy(parameter_length)
