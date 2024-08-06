"""
Todo:
    * Figure out a more general way to implement a hall of fame.

"""
import math
import statistics

from modularcoevolution.genotypes.baseobjectivetracker import MetricConfiguration, compute_shared_objectives
from modularcoevolution.generators.basegenerator import BaseGenerator
from modularcoevolution.utilities.dictutils import deep_update_dictionary
from modularcoevolution.utilities.specialtypes import GenotypeID, EvaluationID

from typing import Any, Type, TypeVar

import abc

# if TYPE_CHECKING:
from modularcoevolution.genotypes.basegenotype import BaseGenotype
from modularcoevolution.agents.baseevolutionaryagent import BaseEvolutionaryAgent
from modularcoevolution.utilities.datacollector import DataCollector
from modularcoevolution.managers.coevolution import Coevolution


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
    previous_population: list[BaseGenotype]
    """The population from the previous generation. Unaffected by :attr:`past_population_width`."""
    past_populations: list[list[BaseGenotype]]
    """A list of populations from previous generations. To save memory, the :attr:`past_population_width` parameter
    can be set to only store the top individuals from each generation."""
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
    compute_diversity: bool
    """If True, compute the diversity of each genotype as a metric."""
    past_population_width: int
    """If non-negative, only store this many of the top individuals per generation in :attr:`past_populations`.
    Useful for saving memory."""
    competitive_fitness_sharing: bool
    """If True, use competitive fitness sharing to compute alternative objective values."""
    shared_sampling_size: int
    """If greater than zero, use shared sampling to provide mandatory opponents
    with high competitive fitness sharing values."""
    _shared_objective_map: dict[str, str]
    """A mapping from shared objective names to their original objective names.
        Only used if :attr:`competitive_fitness_sharing` is True."""

    data_collector: DataCollector
    """The :class:`.DataCollector` to be used for logging."""
    manager: Coevolution
    """The :class:`.Coevolution` manager managing opponents for this generator, if any."""

    def __init__(
            self,
            agent_class: Type[AgentType],
            population_name: str,
            initial_size: int,
            agent_parameters: dict[str, Any] = None,
            genotype_parameters: dict[str, Any] = None,
            seed: list = None,
            data_collector: DataCollector = None,
            manager: Coevolution = None,
            copy_survivor_objectives: bool = False,
            reevaluate_per_generation: bool = True,
            using_hall_of_fame: bool = False,
            compute_diversity: bool = False,
            past_population_width: int = -1,
            competitive_fitness_sharing: bool = False,
            shared_sampling_size: int = -1,
    ):
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
            manager: The :class:`.Coevolution` manager managing opponents for this generator, if any.
            copy_survivor_objectives: If True, genotypes which survive to the next generation will keep their existing
                objective values. If False, objective values will be reset each generation.
            reevaluate_per_generation: If True, all genotypes will be evaluated each generation, even if they were
                previously evaluated. If False, already-evaluated individuals will be skipped. This should be True when
                the fitness landscape can change between generations, such as for coevolution. If this is set to False,
                `copy_survivor_objectives` should be set to True.
            using_hall_of_fame: If True, store a hall of fame and include it in the output of
                :meth:`get_mandatory_opponents`.
            compute_diversity: If True, compute the diversity of each genotype as a metric.
            past_population_width: If non-negative, only store this many of the top individuals per generation in
                :attr:`past_populations`. Useful for saving memory.
            competitive_fitness_sharing: If True, use competitive fitness sharing
                to compute alternative objective values.
            shared_sampling_size: If greater than zero, use shared sampling to provide mandatory opponents
                with high competitive fitness sharing values.
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
        self.manager = manager
        self.copy_survivor_objectives = copy_survivor_objectives
        self.reevaluate_per_generation = reevaluate_per_generation
        assert issubclass(agent_class, BaseEvolutionaryAgent)
        self.genotype_class = agent_class.genotype_class()

        self.generation = 0
        self.population_size = self.initial_size
        self.population = list()
        self.previous_population = list()
        self.past_populations = list()
        self.past_population_width = past_population_width
        self.using_hall_of_fame = using_hall_of_fame
        self.hall_of_fame = list()
        self.genotypes_by_id = dict()

        self.compute_diversity = compute_diversity
        if self.compute_diversity:
            self._register_novelty_metric()

        self.competitive_fitness_sharing = competitive_fitness_sharing
        self.shared_sampling_size = shared_sampling_size
        if shared_sampling_size > 0 and not competitive_fitness_sharing:
            raise ValueError("Shared sampling requires competitive fitness sharing to be enabled.")
        self._shared_objective_map = dict()

        population_set = set()
        for i in range(self.initial_size):
            default_parameters = self.agent_class.genotype_default_parameters(agent_parameters)
            default_parameters.update(self.genotype_parameters)
            if self.seed is not None and i < len(self.seed):
                parameters = default_parameters.copy()
                deep_update_dictionary(parameters, self.seed[i])
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

        Returns: A list of IDs for agents which need to be evaluated.

        """
        result = [genotype.id for genotype in self.population
                  if self.reevaluate_per_generation or not genotype.is_evaluated]
        return result

    def get_mandatory_opponents(self) -> list[GenotypeID]:
        """Get a list of agent IDs which must be evaluated against all opponents.
        This implementation returns the hall of fame if :attr:`using_hall_of_fame` is True, and an empty list otherwise.

        Returns: A list of mandatory opponent IDs.

        """
        mandatory_list = []
        if self.using_hall_of_fame:
            mandatory_list.extend(genotype.id for genotype in self.hall_of_fame)
        if self.shared_sampling_size > 0 and self.generation > 0:
            mandatory_list.extend(self._construct_shared_sample(self.shared_sampling_size))
        return mandatory_list

    def submit_evaluation(
            self,
            agent_id: GenotypeID,
            evaluation_results: dict[str, Any],
            opponents: list[GenotypeID] = None,
    ) -> None:
        """Called by a :class:`.BaseEvolutionManager` to record objectives and metrics from evaluation results
        for the agent with given index.

        Args:
            agent_id: The index of the agent associated with the evaluation results.
            evaluation_results: The results of the evaluation.
            opponents: The IDs of the opponents the agent was evaluated against, if any.
        """

        super().submit_evaluation(agent_id, evaluation_results, opponents)
        individual = self.get_genotype_with_id(agent_id)
        if self.compute_diversity and "novelty" not in individual.metrics:
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

    @abc.abstractmethod
    def end_generation(self) -> None:
        """Called by a :class:`.BaseEvolutionManager` to signal that the current generation has ended.

        Sorting the population and any logging of the generation should be performed here.

        This method should not add or remove individuals from the population.

        """
        # TODO: Separate sorting of population into a separate method, move more to base class
        if self.competitive_fitness_sharing:
            opponents = set()
            for individual in self.population:
                individual_opponents = individual.get_opponents()
                opponents.update(individual_opponents)

            for shared_objective in self._shared_objective_map:
                base_objective = self._shared_objective_map[shared_objective]
                for opponent in opponents:
                    shared_scores = compute_shared_objectives(self.population, opponent, base_objective)
                    for individual, score in zip(self.population, shared_scores):
                        self.submit_metric(individual.id, shared_objective, score)

        super().end_generation()

    @abc.abstractmethod
    def next_generation(self) -> None:
        """Signals the generator that a generation has completed and that the generator may modify its population.

        Changes to the population should only occur as a result of this method being called. However, modifying the
        population at all is optional.

        This function will only be called after :meth:`.end_generation`, so it can be assumed that the population is sorted.
        """
        self.previous_population = self.population.copy()

        super().next_generation()

    def _construct_shared_sample(self, size: int) -> list[GenotypeID]:
        """Construct a sample of genotypes to be used for shared sampling.
        Shared sampling aims to construct a diverse sample set which maximizes marginal shared fitness for each
        individual added to the sample.
        Since we are using a continuous version of competitive fitness sharing, this method does not exactly match
        the version from New Methods for Competitive Coevolution.
        If there are multiple objectives, the sample set will be constructed to maximize the sum of shared objective
        values for each added individual.

        Args:
            size: The size of the sample set to be constructed.

        Returns: A list of genotype IDs of the requested size following the shared sampling algorithm.

        Raises:
            ValueError: If called in the first generation.
        """
        if self.previous_population is None:
            raise ValueError("Cannot use shared sampling in the first generation.")
        if size > len(self.previous_population):
            size = len(self.previous_population)
        # All single individuals will have the same shared objective value (in this continuous variant)
        # So pick a broadly "good" one to start.
        starting_individual = None
        starting_individual_score = None
        for individual in self.previous_population:
            objective_sum = 0
            for base_objective in self._shared_objective_map.values():
                objective_sum += individual.metrics[base_objective]
            if starting_individual is None or objective_sum > starting_individual_score:
                starting_individual = individual
                starting_individual_score = objective_sum
        sample = [starting_individual]

        opponents = set()
        for individual in self.previous_population:
            individual_opponents = individual.get_opponents()
            opponents.update(individual_opponents)

        while len(sample) < size:
            best_individual = None
            best_individual_score = None
            for individual in self.previous_population:
                if individual in sample:
                    continue
                augmented_sample = sample.copy()
                augmented_sample.append(individual)
                # Compute the total shared objective value this individual would get if included in the sample.
                # More distinct individuals to the existing sample will have higher shared objective values.
                total_score = 0
                for shared_objective in self._shared_objective_map:
                    base_objective = self._shared_objective_map[shared_objective]
                    for opponent in opponents:
                        shared_scores = compute_shared_objectives(augmented_sample, opponent, base_objective)
                        total_score += shared_scores[-1]
                if best_individual is None or total_score > best_individual_score:
                    best_individual = individual
                    best_individual_score = total_score
            sample.append(best_individual)
        return [individual.id for individual in sample]

    def register_metric(self, metric_configuration: MetricConfiguration, metric_function: callable) -> None:
        """Register a metric with this generator.

        Args:
            metric_configuration: The metric to register.
            metric_function: A function which computes the metric from the dictionary of evaluation results.
                Alternatively, a string key in the dictionary of evaluation results which contains the metric value.

        """
        if self.competitive_fitness_sharing and metric_configuration['is_objective']:
            modified_metric: MetricConfiguration = metric_configuration.copy()
            modified_metric['name'] = 'base_' + metric_configuration['name']
            modified_metric['is_objective'] = False
            super().register_metric(modified_metric, metric_function)
            shared_metric: MetricConfiguration = metric_configuration.copy()
            shared_metric['automatic'] = False
            shared_metric['log_history'] = False
            super().register_metric(shared_metric, metric_function)
            self._shared_objective_map[shared_metric['name']] = modified_metric['name']
        else:
            super().register_metric(metric_configuration, metric_function)

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
            population_IDs = [individual.id for individual in self.population]
            metric_statistics = dict()
            for metric, configuration in self.metric_configurations.items():
                sample_metric = self.population[0].metrics[metric]
                if isinstance(sample_metric, (int, float)):
                    population_metrics = [individual.metrics[metric] for individual in self.population]
                    metric_statistics[metric] = self.get_metric_statistics(metric, configuration, population_metrics)
            population_metrics = self.get_population_metrics()
            self.data_collector.set_generation_data(
                self.population_name,
                self.generation,
                population_IDs,
                metric_statistics,
                population_metrics
            )

        for individual in self.population:
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

        objective_string = ", ".join([f"{objective}: {self.population[0].objectives[objective]}" for objective in self.population[0].objectives])
        print(f"Best individual of this generation: ({objective_string})")
        print(self.population[0])

        list_amount = min(self.population_size, 100)
        for objective in self.population[0].objectives:
            print(f"{objective}: {str([individual.objectives[objective] for individual in self.population[:list_amount]])}")

    def get_population_metrics(self) -> dict[str, Any]:
        # Diversity measured from the best individual
        if self.compute_diversity:
            diversity = self.population[0].metrics["novelty"]
        else:
            diversity = self.get_diversity(self.population[0].id, min(100, len(self.population)))

        return {
            "diversity": diversity,
        }

    def get_metric_statistics(
            self,
            metric: str,
            configuration: MetricConfiguration,
            population_metrics: list
    ) -> dict[str, Any]:
        finite_metrics = [value for value in population_metrics if math.isfinite(value)]
        metric_mean = statistics.mean(population_metrics)
        if len(finite_metrics) >= 2:
            metric_standard_deviation = statistics.stdev(finite_metrics)
        else:
            metric_standard_deviation = 0
        metric_minimum = min(population_metrics)
        metric_maximum = max(population_metrics)
        return {
            "mean": metric_mean,
            "standard_deviation": metric_standard_deviation,
            "minimum": metric_minimum,
            "maximum": metric_maximum
        }

    def _register_novelty_metric(self) -> None:
        """Automatically register a diversity metric called ``"novelty"``."""
        metric_configuration: MetricConfiguration = {
            "name": "novelty",
            "is_objective": False,
            "repeat_mode": "replace",
            "log_history": False,
            "automatic": False,
            "add_fitness_modifier": False,
        }
        self.register_metric(metric_configuration, None)
