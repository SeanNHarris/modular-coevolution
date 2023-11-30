"""A module storing specially named types to ensure special values like ID numbers are being used correctly.

Stored separately to prevent circular import issues.

"""
from typing import NewType, TypedDict

GenotypeID = NewType("GenotypeID", int)
"""Refers to an ``int`` used as an ID to uniquely reference :class:`.BaseGenotype` instances
"""
_next_genotype_id: GenotypeID = GenotypeID(0)


def claim_genotype_id() -> GenotypeID:
    """Each call to this function will return a unique :class:`GenotypeID` number.

    Returns:
        The next available :class:`GenotypeID` number.
    """
    global _next_genotype_id
    claimed_id = _next_genotype_id
    _next_genotype_id += 1
    return claimed_id


EvaluationID = NewType("EvaluationID", int)
"""Refers to an ``int`` used as an ID to uniquely reference evaluations as part of a :class:`.BaseEvolutionManager`
"""
_next_evaluation_id: EvaluationID = EvaluationID(0)


def claim_evaluation_id() -> EvaluationID:
    """Each call to this function will return a unique :class:`EvaluationID` number.

    Returns:
        The next available :class:`EvaluationID` number.
    """
    global _next_evaluation_id
    claimed_id = _next_evaluation_id
    _next_evaluation_id += 1
    return claimed_id
