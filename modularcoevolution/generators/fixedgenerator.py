from typing import Any, Type, TypeVar

from modularcoevolution.agents.baseagent import BaseAgent
from modularcoevolution.generators.basegenerator import BaseGenerator, AgentType
from modularcoevolution.genotypes.baseobjectivetracker import BaseObjectiveTracker
from modularcoevolution.utilities.specialtypes import GenotypeID


AgentType = TypeVar("AgentType", bound=BaseAgent)


class FixedGenerator(BaseGenerator[AgentType]):
    """A generator that only returns a single fixed agent."""

    agent_class: Type[AgentType]
    agent_parameters: dict[str, Any]
    agent: AgentType

    @property
    def population_size(self) -> int:
        return 1

    def __init__(
            self,
            agent_class: Type[AgentType],
            population_name: str,
            agent_parameters: dict[str, Any] = None,
            *args,
            **kwargs
    ):
        super().__init__(population_name, *args, **kwargs)
        self.agent_class = agent_class
        self.agent_parameters = agent_parameters

        self.agent = self.agent_class(agent_parameters, active=False)

    def get_genotype_with_id(self, agent_id: GenotypeID) -> BaseObjectiveTracker:
        return self.agent.genotype

    def build_agent_from_id(self, agent_id: GenotypeID, active: bool) -> AgentType:
        return self.agent_class(self.agent_parameters, active=active)

    def get_individuals_to_test(self) -> list[GenotypeID]:
        return [self.agent.genotype.id]

    def get_representatives_from_generation(self, generation: int, amount: int, force: bool = False) -> list[
        GenotypeID]:
        return [self.agent.genotype.id]

    def end_generation(self) -> None:
        pass

    def next_generation(self) -> None:
        pass
