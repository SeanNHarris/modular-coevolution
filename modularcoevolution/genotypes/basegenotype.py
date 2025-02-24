"""
Todo:
    * For all genotypes, accept a parameter list as keywords rather than strings.
"""
from modularcoevolution.genotypes.baseobjectivetracker import BaseObjectiveTracker
from modularcoevolution.utilities.specialtypes import claim_genotype_id, GenotypeID

from typing import Any, TypeVar

import abc


Parameters = TypeVar("Parameters", bound=dict[str, Any])


class BaseGenotype(BaseObjectiveTracker, metaclass=abc.ABCMeta):
    """The base class of all genotypes, as used by instances of :class:`.BaseEvolutionaryGenerator`.

    """

    parent_ids: list["GenotypeID"]
    """A list of genotypes used as parents for this one. Could be empty for random genotypes, or have a size
    of 1 for genotypes produced through mutation only."""
    creation_method: str
    """A string describing what method was used to create this genotype, such as "mutation", for logging purposes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
    def get_raw_genotype(self):
        pass

    @abc.abstractmethod
    def diversity_function(self, population, reference=None, samples=None):
        pass
