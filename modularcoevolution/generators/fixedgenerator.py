from typing import Any, Type, TypeVar

from modularcoevolution.agents.baseagent import BaseAgent
from modularcoevolution.agents.baseevolutionaryagent import BaseEvolutionaryAgent
from modularcoevolution.generators.basegenerator import BaseGenerator
from modularcoevolution.genotypes.baseobjectivetracker import BaseObjectiveTracker
from modularcoevolution.utilities.specialtypes import GenotypeID


AgentType = TypeVar("AgentType", bound=BaseAgent)


class FixedGenerator(BaseGenerator[AgentType]):
    """A generator that only returns a single fixed agent."""

    agent_class: Type[AgentType]
    agent_parameters: dict[str, Any]
    agent: AgentType
    agent_id: GenotypeID
    reuse_agent: bool

    @property
    def population_size(self) -> int:
        return 1

    def __init__(
            self,
            agent_class: Type[AgentType],
            population_name: str,
            agent_parameters: dict[str, Any] = None,
            reuse_agent: bool = False,
            *args,
            **kwargs
    ):
        """
        Args:
            agent_class: The type of agent to generate.
            population_name: The name of this population.
            agent_parameters: A dictionary of parameters sent to the agent's constructor.
            reuse_agent: If False, a fresh copy of the agent will be created each time the agent is requested.
                Set to True if the agent needs to cache data between evaluations.
        """
        super().__init__(population_name, *args, **kwargs)
        self.agent_class = agent_class
        self.agent_parameters = agent_parameters
        self.reuse_agent = reuse_agent

        self.agent = self.agent_class(agent_parameters, active=self.reuse_agent)

    def get_genotype_with_id(self, agent_id: GenotypeID) -> BaseObjectiveTracker:
        objective_tracker = self.agent.objective_tracker
        if objective_tracker.id != agent_id:
            raise ValueError(f"ID {agent_id} does not match the ID of the fixed agent ({objective_tracker.id}).")
        return objective_tracker

    def build_agent_from_id(self, agent_id: GenotypeID, active: bool) -> AgentType:
        if self.agent.id != agent_id:
            raise ValueError(f"ID {agent_id} does not match the ID of the fixed agent ({self.agent_id}).")

        if self.reuse_agent:
            return self.agent
        else:
            objective_tracker = self.agent.objective_tracker
            new_agent = self.agent_class(self.agent_parameters, active=active)
            new_agent.objective_tracker.id = objective_tracker.id
            new_agent.objective_tracker.share_metrics_from(objective_tracker)
            return new_agent

    def get_individuals_to_test(self) -> list[GenotypeID]:
        return [self.agent.id]

    def get_representatives_from_generation(self, generation: int, amount: int, force: bool = False) -> list[GenotypeID]:
        return [self.agent.id]

    def end_generation(self) -> None:
        pass

    def next_generation(self) -> None:
        pass