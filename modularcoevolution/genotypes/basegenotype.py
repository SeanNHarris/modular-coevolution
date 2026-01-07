"""
Todo:
    * For all genotypes, accept a parameter list as keywords rather than strings.
"""
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

from modularcoevolution.genotypes.baseobjectivetracker import BaseObjectiveTracker
from modularcoevolution.utilities.specialtypes import claim_genotype_id, GenotypeID

from typing import Any, TypeVar, TypedDict

import abc


Parameters = TypeVar("Parameters", bound=dict[str, Any])


class BaseGenotype(BaseObjectiveTracker, metaclass=abc.ABCMeta):
    """The base class of all genotypes, as used by instances of :class:`.BaseEvolutionaryGenerator`.

    """
    parameters: TypedDict
    """The parameters used to create this genotype. Used for serialization."""

    parent_ids: list["GenotypeID"]
    """A list of genotypes used as parents for this one. Could be empty for random genotypes, or have a size
    of 1 for genotypes produced through mutation only."""
    creation_method: str
    """A string describing what method was used to create this genotype, such as "mutation", for logging purposes."""

    def __init__(self, parameters, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = parameters

        self.parent_ids = list()
        self.creation_method = "Parthenogenesis"

    @abc.abstractmethod
    def mutate(self):
        """An evolutionary mutation operator which modifies the current individual. Clone the parent first for standard
        evolutionary mutation.

        """
        pass

    @abc.abstractmethod
    def recombine(self, donor: "BaseGenotype"):
        """An evolutionary recombinaton operator which combines elements of this genotype and the
        ``donor`` genotype. Clone the donor parent first for standard
        evolutionary recombination.

        Args:
            donor: Another genotype of the same class which should be combined into this one. In asymmetric
                recombination operators, this individual is treated as the secondary parent.

        """
        pass

    @abc.abstractmethod
    def clone(self) -> "BaseGenotype":
        """Return a new individual with an identical genotype. The new individual will have a new ID and metrics.

        Returns:
            A new individual with an identical genotype.

        """
        pass

    @abc.abstractmethod
    def __hash__(self) -> int:
        pass

    @abc.abstractmethod
    def get_raw_genotype(self) -> dict[str, Any]:
        """Return a dictionary of parameters sufficient to recreate this genotype,
        assuming the same experiment and configuration file is used.
        Parameters that will always be implied by these can be omitted.

        This is what is stored in :class:`.DataCollector` logs.

        Returns:
            A dictionary of parameters sufficient to recreate this genotype when passed to :meth:`__init__`
            in combination with the experiment configuration.
        """
        pass

    @abc.abstractmethod
    def diversity_function(self, population, reference=None, samples=None):
        pass
    #
    # def __getstate__(self):
    #     """
    #     NOTE: CURRENTLY BROKEN
    #       Causes strings instead of types to be pickled for subgenotypes in MultipleGenotype.
    #
    #     Serializes the :class:`BaseGenotype` for pickling from its parameters and raw genotype,
    #     rather than by directly pickling it.
    #
    #     Note:
    #         Pickling will lose any temporary values stored in the genotype,
    #         such as saved values from :meth:`GPTree.execute` with `save_values=True`.
    #
    #     Returns:
    #         A dictionary containing the state of the genotype,
    #         including parameters sufficient to recreate this genotype when passed to :meth:`__init__`.
    #     """
    #     parameters = dict(self.parameters)
    #     parameters.update(self.get_raw_genotype())
    #
    #     local_dict = {
    #         'parent_ids': self.parent_ids,
    #         'creation_method': self.creation_method,
    #     }
    #
    #     base_state = super().__getstate__()
    #
    #     return {
    #         'parameters': parameters,
    #         **local_dict,
    #         **base_state,
    #     }
    #
    # def __setstate__(self, state):
    #     self.__init__(state['parameters'])
    #     self.__dict__.update(state)
