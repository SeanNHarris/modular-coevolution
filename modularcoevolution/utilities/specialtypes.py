"""A module storing specially named types to ensure special values like ID numbers are being used correctly.

Stored separately to prevent circular import issues.

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

from typing import NewType, TypedDict

GenotypeID = NewType("GenotypeID", int)
"""A unique integer ID assigned to each :class:`.BaseGenotype` instance as a primary key."""
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
"""A unique integer ID assigned to each :class:`.BaseEvolutionManager` evaluation as a primary key.
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
