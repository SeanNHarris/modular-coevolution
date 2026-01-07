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

__author__ = 'Sean N. Harris'
__copyright__ = 'Copyright 2026, BONSAI Lab at Auburn University'
__license__ = 'Apache-2.0'

from typing import Sequence, Any

import numpy

from modularcoevolution.genotypes.lineargenotype import (
    LinearGenotype,
    LinearGenotypeParameters,
    GENE_MUTATION_SIZE_DEFAULT,
    GENE_MUTATION_RATE_DEFAULT,
    MIN_VALUE_DEFAULT,
    MAX_VALUE_DEFAULT)


class NumpyLinearGenotype(LinearGenotype):
    genes: numpy.ndarray
    min_value: numpy.ndarray
    max_value: numpy.ndarray
    loop_genes: numpy.ndarray
    round_genes: numpy.ndarray

    def __init__(self, parameters: LinearGenotypeParameters):
        # Skip parent class, call grandparent __init__
        super(LinearGenotype, self).__init__(parameters)
        if 'gene_mutation_rate' in parameters:
            self.gene_mutation_rate = parameters['gene_mutation_rate']
        else:
            self.gene_mutation_rate = GENE_MUTATION_RATE_DEFAULT

        if 'gene_mutation_standard_deviation' in parameters:
            self.gene_mutation_standard_deviation = parameters['gene_mutation_standard_deviation']
        else:
            self.gene_mutation_standard_deviation = GENE_MUTATION_SIZE_DEFAULT

        must_generate = False
        self.genes = None
        if 'values' in parameters:
            if isinstance(parameters['values'], numpy.ndarray):
                self.genes = parameters['values'].copy()
            elif isinstance(parameters['values'], bytes):
                self.genes = numpy.frombuffer(parameters['values'], dtype=numpy.float16)
            elif isinstance(parameters['values'], Sequence):
                self.genes = numpy.array(parameters['values'], dtype=numpy.float16)
            self.length = len(self.genes)
        elif 'length' in parameters:
            must_generate = True
            self.length = parameters['length']
        if self.genes is None and must_generate == False:
            raise TypeError('If \'values\' is not provided, a \'length\' must be.')

        # TODO: Clean this up, since these are all basically the same per parameter
        def apply_gene_parameter(parameter_name, default):
            if parameter_name in parameters:
                if isinstance(parameters[parameter_name], numpy.ndarray):
                    value = parameters[parameter_name]  # Immutable, don't need to copy
                elif isinstance(parameters[parameter_name], Sequence):
                    value_list = list()
                    for i in range(self.length):
                        value_list.append(parameters[parameter_name][i % len(parameters[parameter_name])])
                    value = numpy.array(value_list)
                else:
                    value = numpy.full(self.length, parameters[parameter_name])
            else:
                value = numpy.full(self.length, default)
            self.__dict__[parameter_name] = value

        apply_gene_parameter('min_value', MIN_VALUE_DEFAULT)
        apply_gene_parameter('max_value', MAX_VALUE_DEFAULT)
        apply_gene_parameter('loop_genes', False)
        apply_gene_parameter('round_genes', False)

        if must_generate:
            self.genes = numpy.random.uniform(self.min_value, self.max_value).astype(numpy.float16)
            if self.round_genes.any():
                self.genes[self.round_genes] = numpy.round(self.genes[self.round_genes])

    def get_raw_genotype(self):
        return {'values': self.genes.tobytes()}

