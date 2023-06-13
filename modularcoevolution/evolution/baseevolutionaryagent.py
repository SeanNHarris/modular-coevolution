from modularcoevolution.evolution.baseagent import BaseAgent

from typing import Any, Generic, Type, TypedDict, TypeVar

import abc

#if TYPE_CHECKING:
from modularcoevolution.evolution.basegenotype import BaseGenotype


AgentParameters = TypeVar("AgentParameters", bound=dict[str, Any])
GenotypeType = TypeVar("GenotypeType", bound=BaseGenotype)
GenotypeParameters = TypeVar("GenotypeParameters", bound=dict[str, Any])


class BaseEvolutionaryAgent(BaseAgent, Generic[GenotypeType], metaclass=abc.ABCMeta):
    """The superclass of all agents to be used in evolution (through :class:`.BaseEvolutionaryGenerator`).

    These agents have an associated :attr:`genotype`, which is a :class:`.BaseGenotype` that handles behavior related
    to evolution through the associated :class:`.BaseEvolutionaryGenerator`.

    """

    genotype: GenotypeType
    """The genotype associated with this agent. This is primarily interacted with by a
    :class:`.BaseEvolutionaryGenerator` to assign and view objective values from evaluation.
    
    Initialize this through :meth:`initialize_genotype`.
    """

    @classmethod
    @abc.abstractmethod
    def genotype_class(cls) -> Type[GenotypeType]:
        """Defines the class of this evolutionary agent's genotype.

        Returns: The class to be used for this agent's genotype.

        """
        pass

    @classmethod
    @abc.abstractmethod
    def genotype_default_parameters(cls) -> dict[str, Any]:
        """Defines the default parameters to be sent when creating this agent's genotype.

        Returns: The default parameters to be sent to this agent's genotype ``__init__``

        """
        pass

    @abc.abstractmethod
    def apply_parameters(self, parameters: dict[str, Any]) -> None:
        """See :meth:`.BaseAgent.apply_parameters`.

        This is a partial implementation that can be called with ``super``
        to automatically initialize the genotype from a provided ``"genotype_parameters"`` key in ``parameters``.

        In some cases, this might not be desired behavior, such as when the length of a linear genotype depends
        on agent parameters. In that case, do not call ``super`` and instead use :meth:`initialize_genotype` directly.

        .. note::
            Be aware that the :attr:`genotype` might already exist at this point if the agent was created by a
            :class:`.BaseEvolutionaryGenerator`. If parameters to the :attr:`genotype` are dependent on external
            factors, use the ``genotype_parameters`` parameter to :meth:`.BaseEvolutionaryGenerator.__init__`.

        Args:
            parameters: A dictionary of parameters to initialize the agent with.

        """
        if "genotype_parameters" in parameters:
            self.initialize_genotype(parameters["genotype_parameters"])

    def initialize_genotype(self, genotype_parameters: dict[str, Any], force: bool = False) -> None:
        """Sets :attr:`genotype` based on the given parameters and default parameters.

        This should be run during :meth:`apply_parameters`.

        :attr:`genotype` will be set to an instance of :meth:`genotype_class` using :meth:`genotype_default_parameters`
        and the ``genotype_parameters`` parameter. The provided parameters override any conflicting defaults.

        Args:
            genotype_parameters: A set of parameters to be passed to the genotype's ``__init__`` function,
                in addition to the default parameters. Overrides any conflicting defaults.
            force: If false, this method will do nothing if a genotype was already assigned
                (e.g. through :meth:`__init__`).

        """
        if hasattr(self, "genotype") and not force:
            return
        full_genotype_parameters = self.genotype_default_parameters()
        full_genotype_parameters.update(genotype_parameters)
        self.genotype = self.genotype_class()(**full_genotype_parameters)

    def __init__(self, parameters: dict[str, Any] = None, genotype: BaseGenotype = None, *args, **kwargs):
        """

        Args:
            parameters: Parameters to pass to :meth:`apply_parameters`.
            genotype: A pre-initialized genotype to adopt instead of creating a new one from parameters.
            *args: Passes extra arguments to :meth:`.BaseAgent.__init__`.
            **kwargs: Passes extra keyword arguments to :meth:`.BaseAgent.__init__`.

        """
        self.genotype = None
        if genotype is not None:
            self.genotype = genotype
        elif parameters is None and genotype is None:
            self.genotype = self.genotype_class()(self.genotype_default_parameters())

        super().__init__(parameters, *args, **kwargs)
