"""
Todo:
    * For all genotypes, accept a parameter list as keywords rather than strings.
"""
from modularcoevolution.evolution.baseobjectivetracker import BaseObjectiveTracker
from modularcoevolution.evolution.specialtypes import claim_genotype_id, GenotypeID

from typing import Any, ClassVar, Generic, TypeVar

import abc


Parameters = TypeVar("Parameters", bound=dict[str, Any])


class BaseGenotype(BaseObjectiveTracker, metaclass=abc.ABCMeta):
    """The base class of all genotypes, as used by instances of :class:`.BaseEvolutionaryGenerator`.

    """

    id: GenotypeID
    """The ID associated with this genotype. ID values are unique across all genotypes (assuming no transfer between
    multiple python processes)."""

    parent_ids: list["GenotypeID"]
    """A list of genotypes used as parents for this one. Could be empty for random genotypes, or have a size
    of 1 for genotypes produced through mutation only."""
    creation_method: str
    """A string describing what method was used to create this genotype, such as "mutation", for logging purposes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = claim_genotype_id()

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
    def clone(self, copy_objectives: bool = False) -> "BaseGenotype":
        """Return a deep copy of the current individual.

        Args:
            copy_objectives: If True, objective values will be copied to the clone.
                If False, objective values will be reset for the clone.

        Returns:
            A deep copy of the current individual.

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
