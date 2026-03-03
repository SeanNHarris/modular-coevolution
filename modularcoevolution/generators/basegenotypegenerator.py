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

import abc
from typing import Type, Any

from modularcoevolution.agents.baseevolutionaryagent import BaseEvolutionaryAgent
from modularcoevolution.generators.basegenerator import BaseGenerator
from modularcoevolution.genotypes.basegenotype import BaseGenotype
from modularcoevolution.genotypes.baseobjectivetracker import BaseObjectiveTracker
from modularcoevolution.utilities.specialtypes import GenotypeID


class BaseGenotypeGenerator[AgentType: BaseEvolutionaryAgent, GenotypeType: BaseGenotype](
    BaseGenerator[AgentType], metaclass=abc.ABCMeta):
    agent_class: Type[AgentType]
    """The class to instantiate agents with."""
    genotype_class: Type[GenotypeType]
    """The class to instantiate genotypes with, determined by :attr:`agent_class`."""
    agent_parameters: dict[str, Any]
    """The parameters to be sent to the ``__init__`` function of :attr:`agent_class`, other than genotype."""
    genotype_parameters: dict[str, Any]
    """The parameters to be sent to the ``__init__`` function of the :attr:`genotype_class`, in addition to the default
    parameters from :meth:`.BaseEvolutionaryAgent.genotype_default_parameters`. Overwrites any default parameters."""

    genotypes_by_id: dict[GenotypeID, GenotypeType]
    """A mapping from an ID to a genotype with that :attr:`.BaseGenotype.id`."""

    def __init__(
            self,
            population_name: str,
            agent_class: Type[AgentType],
            agent_parameters: dict[str, Any] = None,
            genotype_parameters: dict[str, Any] = None,
            **kwargs
    ):
        super().__init__(population_name, **kwargs)
        self.agent_class = agent_class
        self.agent_parameters = agent_parameters
        if self.agent_parameters is None:
            self.agent_parameters = {}
        self.genotype_parameters = genotype_parameters
        if self.genotype_parameters is None:
            self.genotype_parameters = {}

        assert issubclass(agent_class, BaseEvolutionaryAgent)
        self.genotype_class = agent_class.genotype_class()

        self.genotypes_by_id = {}

    def get_genotype_with_id(self, agent_id) -> GenotypeType:
        """Return the genotype with the given ID.

        Args:
            agent_id: The ID of the genotype being requested.

        Returns: The genotype associated with the ID ``agent_id``.

        """
        if agent_id not in self.genotypes_by_id:
            raise ValueError(f"The agent ID {agent_id} is not present in this generator."
                             f"Ensure the correct generator is being queried.")
        return self.genotypes_by_id[agent_id]

    def get_tracker_with_id(self, agent_id: GenotypeID) -> BaseObjectiveTracker:
        return self.get_genotype_with_id(agent_id).objective_tracker

    def _build_agent_from_id(self, agent_id: GenotypeID, active: bool) -> AgentType:
        """Return a new instance of an agent based on the given agent ID.

        Args:
            agent_id: The ID associated with the agent being requested.
            active: Used for the ``active`` parameter in :meth:`.BaseAgent.__init__`.

        Returns: A newly created agent associated with the ID ``agent_id`` and with ``active`` as its activity state.

        """
        if agent_id not in self.genotypes_by_id:
            raise ValueError(f"The agent ID {agent_id} is not present in this generator."
                             f"Ensure the correct generator is being queried.")
        agent = self.agent_class(genotype=self.genotypes_by_id[agent_id], active=active, parameters=self.agent_parameters)
        return agent