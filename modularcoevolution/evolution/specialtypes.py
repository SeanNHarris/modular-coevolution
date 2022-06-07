"""A module storing specially named types to ensure special values like ID numbers are being used correctly.

Stored separately to prevent circular import issues.

"""
from typing import NewType, TypedDict

GenotypeID = NewType("GenotypeID", int)
"""Refers to an ``int`` used as an ID to reference :class:`.BaseGenotype` instances
"""
_next_genotype_id: GenotypeID = GenotypeID(0)


def claim_genotype_id() -> GenotypeID:
    global _next_genotype_id
    claimed_id = _next_genotype_id
    _next_genotype_id += 1
    return claimed_id


EvaluationID = NewType("EvaluationID", int)
"""Refers to an ``int`` used as an ID to reference evaluations as part of a :class:`.BaseEvolutionWrapper`
"""
_next_evaluation_id: EvaluationID = EvaluationID(0)


def claim_evaluation_id() -> EvaluationID:
    global _next_evaluation_id
    claimed_id = _next_evaluation_id
    _next_evaluation_id += 1
    return claimed_id


class ObjectiveStatistics(TypedDict):
    mean: float
    std_dev_intermediate: float
    standard_deviation: float
    minimum: float
    maximum: float
