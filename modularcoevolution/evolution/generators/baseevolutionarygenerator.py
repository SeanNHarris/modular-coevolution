"""
Todo:
    * Figure out a more general way to implement a hall of fame.

"""
import statistics

from modularcoevolution.evolution.baseobjectivetracker import MetricConfiguration
from modularcoevolution.evolution.generators.basegenerator import BaseGenerator
from modularcoevolution.evolution.specialtypes import GenotypeID, EvaluationID

from typing import Any, Callable, Generic, Type, TypeVar

import abc

# if TYPE_CHECKING:
from modularcoevolution.evolution.basegenotype import BaseGenotype
from modularcoevolution.evolution.baseevolutionaryagent import BaseEvolutionaryAgent
from modularcoevolution.evolution.datacollector import DataCollector


AgentType = TypeVar("AgentType", bound=BaseEvolutionaryAgent)
AgentParameters = TypeVar("AgentParameters", bound=dict[str, Any])
GenotypeType = TypeVar("GenotypeType", bound=BaseGenotype)
GenotypeParameters = TypeVar("GenotypeParameters", bound=dict[str, Any])


class BaseEvolutionaryGenerator(BaseGenerator[AgentType], metaclass=abc.ABCMeta):
    """A base class for evolutionary algorithms (EAs) that implements many of the abstract functions from
    :class:`.BaseGenerator`.

    """

    population: list[BaseGenotype]
    """The current population of the EA."""
    past_populations: list[list[BaseGenotype]]
    """A list of populations from previous generations."""
    hall_of_fame: list[BaseGenotype]
    """A hall of fame storing high-quality individuals from past generations. Nothing adds to the hall of fame in this
    abstract base class."""
    using_hall_of_fame: bool
    """Whether to use the hall of fame."""
    genotypes_by_id: dict[GenotypeID, BaseGenotype]
    """A mapping from an ID to a genotype with that :attr:`.BaseGenotype.id`."""
    generation: int
    """The current generation of evolution."""

    agent_class: Type[AgentType]
    """The class to instantiate agents with."""
    genotype_class: Type[BaseGenotype]
    """The class to instantiate genotypes with, determined by :attr:`agent_class`."""
    agent_parameters: dict[str, Any]
    """The parameters to be sent to the ``__init__`` function of :attr:`agent_class`, other than genotype."""
    genotype_parameters: dict[str, Any]
    """The parameters to be sent to the ``__init__`` function of the :attr:`genotype_class`, in addition to the default
    parameters from :meth:`.BaseEvolutionaryAgent.genotype_default_parameters`. Overwrites any default parameters."""
    initial_size: int
    """The initial size of the population. Can be treated as the *mu* parameter."""
    copy_survivor_objectives: bool
    """If True, genotypes which survive to the next generation will keep their existing objective values. If False,
    objective values will be reset each generation."""
    reevaluate_per_generation: bool
    """If True, all genotypes will be evaluated each generation, even if they were previously evaluated. If False,
    already-evaluated individuals will be skipped."""

    data_collector: DataCollector
    """The :class:`.DataCollector` to be used for logging."""

    def __init__(self, agent_class: Type[AgentType],
                 population_name: str,
                 initial_size: int,
                 agent_parameters: dict[str, Any] = None,
                 genotype_parameters: dict[str, Any] = None,
                 seed: list = None,
                 data_collector: DataCollector = None,
                 copy_survivor_objectives: bool = False,
                 reevaluate_per_generation: bool = True,
                 using_hall_of_fame: bool = True):
        """

        Args:
            agent_class: The type of agent to be generated through evolution.
            population_name: The name of the population being generated. Used as a primary key for logging.
            initial_size: The initial size of the population.
            agent_parameters: The parameters to be sent to the ``__init__`` function of ``agent_class``,
                other than genotype.
            genotype_parameters: The parameters to be sent to the ``__init__`` function of the genotype specified by
                :meth:`.BaseEvolutionaryAgent.genotype_class`, in addition to the default parameters from
                :meth:`.BaseEvolutionaryAgent.genotype_default_parameters`. Overwrites any default parameters.
            seed: A list of genotype parameters which will each be used to add one genotype to the initial population.
            data_collector: The :class:`.DataCollector` to be used for logging.
            copy_survivor_objectives: If True, genotypes which survive to the next generation will keep their existing
                objective values. If False, objective values will be reset each generation.
            reevaluate_per_generation: If True, all genotypes will be evaluated each generation, even if they were
                previously evaluated. If False, already-evaluated individuals will be skipped. This should be True when
                the fitness landscape can change between generations, such as for coevolution. If this is set to False,
                `copy_survivor_objectives` should be set to True.
            using_hall_of_fame: If True, store a hall of fame and include it in the output of
                :meth:`get_individuals_to_test`.

        .. warning::
            ``using_hall_of_fame`` currently uses a non-standard implementation of the hall of fame and is subject to
            change. It was intended for a specific application and has not been expanded.
        """
        super().__init__(population_name)
        self.agent_class = agent_class
        self.agent_parameters = agent_parameters
        if self.agent_parameters is None:
            self.agent_parameters = dict()
        self.genotype_parameters = genotype_parameters
        if self.genotype_parameters is None:
            self.genotype_parameters = dict()
        self.initial_size = initial_size
        self.seed = seed
        self.data_collector = data_collector
        self.copy_survivor_objectives = copy_survivor_objectives
        self.reevaluate_per_generation = reevaluate_per_generation
        assert issubclass(agent_class, BaseEvolutionaryAgent)
        self.genotype_class = agent_class.genotype_class()

        self.generation = 0
        self.population_size = self.initial_size
        self.population = list()
        self.past_populations = list()
        self.using_hall_of_fame = using_hall_of_fame
        self.hall_of_fame = list()
        self.genotypes_by_id = dict()

        self._register_novelty_metric()

        population_set = set()
        for i in range(self.initial_size):
            default_parameters = self.agent_class.genotype_default_parameters(agent_parameters)
            default_parameters.update(self.genotype_parameters)
            if self.seed is not None and i < len(self.seed):
                parameters = default_parameters.copy()
                parameters.update(self.seed[i])
                individual = self.genotype_class(parameters)
                self.population.append(individual)
                population_set.add(hash(individual))
            else:
                unique = False
                individual = None
                while not unique:
                    individual = self.genotype_class(default_parameters.copy())
                    if hash(individual) not in population_set:
                        unique = True
                self.population.append(individual)
                population_set.add(hash(individual))
        for genotype in self.population:
            self.genotypes_by_id[genotype.id] = genotype

    def population_size(self) -> int:
        return len(self.population)

    def get_genotype_with_id(self, agent_id) -> BaseGenotype:
        """Return the genotype with the given ID.

        Args:
            agent_id: The ID of the genotype being requested.

        Returns: The genotype associated with the ID ``agent_id``.

        """
        if agent_id not in self.genotypes_by_id:
            raise ValueError(f"The agent ID {agent_id} is not present in this generator."
                             f"Ensure the correct generator is being queried.")
        return self.genotypes_by_id[agent_id]

    def build_agent_from_id(self, agent_id: GenotypeID, active: bool) -> BaseEvolutionaryAgent:
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
    
    def get_individuals_to_test(self) -> list[GenotypeID]:
        """Get a list of agent IDs in need of evaluation, skipping those already evaluated if
        :attr:`reevaluate_per_generation` is False.

        If :attr:`using_hall_of_fame` is true, the hall of fame will be added to this list.

        .. warning::
            This is a non-standard implementation of the hall of fame and is subject to change.
            It was intended for a specific application and has not been expanded.

        Returns: A list of IDs for agents which need to be evaluated.

        """
        result = [genotype.id for genotype in self.population
                  if self.reevaluate_per_generation or not genotype.is_evaluated]
        if self.using_hall_of_fame:
            result += [genotype.id for genotype in self.hall_of_fame]
        return result

    def submit_evaluation(self, agent_id: GenotypeID, evaluation_id: EvaluationID, evaluation_results: dict[str, Any]) -> None:
        """Called by a :class:`.BaseEvolutionWrapper` to record objectives and metrics from evaluation results
        for the agent with given index.

        Args:
            agent_id: The index of the agent associated with the evaluation results.
            evaluation_id: The ID of the evaluation.
            evaluation_results: The results of the evaluation.

        """

        super().submit_evaluation(agent_id, evaluation_id, evaluation_results)
        individual = self.get_genotype_with_id(agent_id)
        if "novelty" not in individual.metrics:
            novelty = self.get_diversity(agent_id, min(100, len(self.population)))
            self.submit_metric(agent_id, "novelty", novelty)

        if self.data_collector is not None:
            self.data_collector.set_individual_data(
                self.population_name,
                individual.id,
                individual.get_raw_genotype(),
                individual.evaluation_ids.copy(),
                individual.metrics.copy(),
                individual.metric_statistics.copy(),
                individual.metric_histories.copy(),
                individual.parent_ids.copy(),
                individual.creation_method,
            )

    def get_diversity(self, reference_id: GenotypeID = None, samples: int = None) -> float:
        """Calculates the diversity of the population with respect to a reference individual.

        Population diversity is calculated by averaging the diversity metric from the reference individual to several
        other individuals.

        Args:
            reference_id: The id of the genotype to calculate diversity with respect to.
                If omitted, the highest-fitness genotype will be used, or a random one if no fitness is assigned.
            samples: The number of genotypes to compare against, selected at random without replacement.
                If omitted, the entire population will be compared against.

        Returns: The average diversity from the reference individual to other members of the population.

        """
        if reference_id is not None:
            reference = self.genotypes_by_id[reference_id]
        else:
            reference = None
        return reference.diversity_function(self.population, reference, samples)

    def log_generation(self) -> None:
        """Log the current generation and its associated statistics to the data collector, if one is present.

        This method should be called during :meth:`end_generation`."""
        if self.data_collector is not None:
            diversity = self.population[0].metrics["novelty"]  # Diversity measured from the best individual
            population_IDs = [individual.id for individual in self.population]
            metric_statistics = dict()
            for metric, configuration in self.metric_configurations.items():
                sample_metric = self.population[0].metrics[metric]
                if isinstance(sample_metric, (int, float)):
                    population_metrics = [individual.metrics[metric] for individual in self.population]
                    metric_mean = statistics.mean(population_metrics)
                    metric_standard_deviation = statistics.stdev(population_metrics)
                    metric_minimum = min(population_metrics)
                    metric_maximum = max(population_metrics)
                    metric_statistics[metric] = {
                        "mean": metric_mean,
                        "standard deviation": metric_standard_deviation,
                        "minimum": metric_minimum,
                        "maximum": metric_maximum
                    }
            population_metrics = {
                "diversity": diversity,
            }
            self.data_collector.set_generation_data(
                self.population_name,
                self.generation,
                population_IDs,
                metric_statistics,
                population_metrics
            )

        print("Best individual of this generation: (fitness score of " + str(self.population[0].fitness) + ")")
        print(self.population[0])

        print(str([individual.fitness for individual in self.population]))

    def _register_novelty_metric(self) -> None:
        """Automatically register a diversity metric called ``"novelty"``."""
        metric_configuration: MetricConfiguration = {
            "name": "novelty",
            "is_objective": False,
            "repeat_mode": "replace",
            "log_history": False,
            "automatic": False
        }
        self.register_metric(metric_configuration, None)
