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

from modularcoevolution.genotypes.lineargenotype import LinearGenotype

import random

MUTATION_RATE = 0.1


class BinaryGenotype(LinearGenotype):
    def __init__(self, parameters):
        if "initial_rate" in parameters:
            self.initial_rate = parameters["initial_rate"]
        else:
            self.initial_rate = 0.5

        super().__init__(parameters)

    def random_gene(self, index):
        return random.random() < self.initial_rate
    
    def mutate(self):
        for i in range(len(self.genes)):
            if random.random() < MUTATION_RATE:
                self.genes[i] = not self.genes[i]
        self.creation_method = "Mutation"
