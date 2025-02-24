from modularcoevolution.genotypes.baseobjectivetracker import BaseObjectiveTracker
from modularcoevolution.utilities.agenttyperegistry import AgentTypeRegistry

from typing import Any, ClassVar

import abc


class BaseAgent(metaclass=AgentTypeRegistry):
    """The superclass of all agents, including :class:`.BaseEvolutionaryAgent`.

    :class:`.BaseGenerator` exist to generate :class:`.BaseAgent`, such as through evolution.
    """

    agent_type_name: ClassVar[str] = "no agent type name"
    """A class can have an `agent_type_name` that will be used by default as a key for logging.
    
    Alternatively, agent type names can be set per instance as an argument to :meth:`__init__`,
    which overrides this value.
    """

    active: bool
    """Defines whether the agent is "active".
    
    If an agent needs to be created but not executed, for example if an agent is generated by evolution
    but needs to send its parameters off to be run elsewhere, this can be referenced to prevent certain
    behaviors from running locally.
    """

    agent_type_name: str
    """The agent type name for this specific instance.
    Uses the class-defined agent type name by default, or can be overridden by an argument to :meth:`__init__`.
    """

    @property
    @abc.abstractmethod
    def objective_tracker(self) -> BaseObjectiveTracker:
        """Returns the :class:`.BaseObjectiveTracker` associated with this agent.
        This is usually a :class:`.BaseGenotype`, but for non-evolving agents it may not be.

        Returns:
            The objective tracker associated with this agent.
        """

        ...

    @property
    def id(self) -> Any:
        """Returns the ID associated with this agent's objective tracker.
        ID is unique between objective trackers (e.g. genotypes), but not necessarily between agents.

        Returns:
            The ID of this agent's objective tracker.
        """
        return self.objective_tracker.id

    @abc.abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Gets the parameters for this agent.

        Returns:
            A dictionary of parameters which could be passed to apply_parameters to create a copy of this agent.

        Todo:
            Allow subclasses to specify a specific parameter format. Use custom types?
        """
        pass

    @abc.abstractmethod
    def apply_parameters(self, parameters: dict[str, Any]) -> None:
        """Apply a dictionary of parameters to the agent.

        This function is called as part of the base class's :func:`__init__`.
        Most initialization should take place here, rather than in a subclass's ``__init__``.

        Args:
            parameters: A dictionary of parameters to initialize the agent with.

        """
        pass

    def parameter_string(self) -> str:
        """Return a string which displays relevant parameters to the agent for logging purposes,
        such that the agent can be reconstructed later. For example, a genome.

        Defaults to the parameter dictionary's string representation.

        Returns:
            A canonical string representation of the agent's parameters.

        """
        return repr(self.get_parameters())

    @abc.abstractmethod
    def perform_action(self, *args, **kwargs):
        """Request that the agent "perform an action" using the given parameters.

        This can modify the arguments, or just return an action, depending on implementation.

        Args:
            *args: Subclass-defined arguments.
            **kwargs: Subclass-defined keyword arguments.

        Returns:
            Subclass-defined return value, if any.

        """
        pass

    def __init__(self, parameters: dict[str, Any] = None, active: bool = True, *args, agent_type_name: str = None,
                 **kwargs):
        """

        Args:
            parameters: Parameters to pass to :meth:`apply_parameters`.
            active: Flags the agent as active or inactive.
            *args: Consumes additional arguments from the subclass.
            agent_type_name: Agent type name for this specific instance. Overrides the class-defined agent type name.
                Use this if you are evolving multiple populations using the same agent class.
            **kwargs: Consumes additional keyword arguments from the subclass.

        """
        super().__init__()

        self.active = active
        if agent_type_name is not None:
            self.agent_type_name = agent_type_name
        else:
            self.agent_type_name = type(self).agent_type_name

        if parameters is not None:
            self.apply_parameters(parameters)
