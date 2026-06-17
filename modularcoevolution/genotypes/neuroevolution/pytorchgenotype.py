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

import io
from typing import Any, TypedDict

from modularcoevolution.genotypes.basegenotype import BaseGenotype


try:
    import torch
except ImportError as error:
    print('PyTorch is required for neuroevolution modules.')
    raise error


class PyTorchGenotypeParameters(TypedDict, total=False):
    network_parameters: dict[str, torch.Tensor | bytes]
    """Optional, explicitly sets the network parameters."""

    parameter_sizes: dict[str, torch.Size]
    """The size of each set of network parameters."""

    mutation_std: float
    """The standard deviation of the mutation factor."""


_MUTATION_STD_DEFAULT = 0.01


class PyTorchGenotype(BaseGenotype):
    network_parameters: dict[str, torch.Tensor]
    parameter_sizes: dict[str, torch.Size]

    mutation_std: float

    def __init__(self, parameters: PyTorchGenotypeParameters):
        super().__init__(parameters)

        if 'network_parameters' in parameters:
            self.network_parameters = {}
            for parameter_name, parameter_tensor in parameters['network_parameters'].items():
                if isinstance(parameter_tensor, torch.Tensor):
                    self.network_parameters[parameter_name] = parameter_tensor
                elif isinstance(parameter_tensor, bytes):
                    self.network_parameters[parameter_name] = _bytes_to_tensor(parameter_tensor)
            self.parameter_sizes = {}
            for parameter_name, parameter_tensor in self.network_parameters.items():
                self.parameter_sizes[parameter_name] = parameter_tensor.size()
        elif 'parameter_sizes' in parameters:
            self.parameter_sizes = parameters['parameter_sizes']
            self.network_parameters = {}
            for parameter_name, parameter_size in self.parameter_sizes.items():
                self.network_parameters[parameter_name] = torch.randn(parameter_size)
        else:
            raise ValueError("If 'network_parameters' is not provided, 'parameter_sizes' must be.")

        if 'mutation_std' in parameters:
            self.mutation_std = parameters['mutation_std']
        else:
            self.mutation_std = _MUTATION_STD_DEFAULT

    def mutate(self):
        for parameter_name, parameters in self.network_parameters.items():
            means = torch.ones_like(parameters)
            stds = torch.full_like(parameters, self.mutation_std)
            self.network_parameters[parameter_name] = parameters * torch.normal(means, stds)

    def recombine(self, donor: "PyTorchGenotype"):
        # TODO: Revisit this after testing
        for parameter_name, parameters in self.network_parameters.items():
            donor_parameters = donor.network_parameters[parameter_name]
            uniform_crossover = torch.rand_like(parameters) < 0.5
            self.network_parameters[parameter_name] = torch.where(uniform_crossover, parameters, donor_parameters)

    def clone(self) -> "PyTorchGenotype":
        cloned_parameters: PyTorchGenotypeParameters = {
            'network_parameters': {name: tensor.clone() for name, tensor in self.network_parameters.items()},
            'mutation_std': self.mutation_std
        }
        return PyTorchGenotype(cloned_parameters)

    def __hash__(self) -> int:
        parameter_hashes = tuple(torch.hash_tensor(parameters) for parameters in self.network_parameters.values())
        return hash(parameter_hashes)

    def get_raw_genotype(self) -> dict[str, Any]:
        parameters: PyTorchGenotypeParameters = {
            'network_parameters': {name: _tensor_to_bytes(tensor) for name, tensor in self.network_parameters.items()},
            'mutation_std': self.mutation_std
        }
        return parameters

    def diversity_function(self, population, reference=None, samples=None):
        pass


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def _bytes_to_tensor(bytes_tensor: bytes) -> torch.Tensor:
    buffer = io.BytesIO(bytes_tensor)
    return torch.load(buffer)