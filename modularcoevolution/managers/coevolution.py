import heapq
import multiprocessing

from modularcoevolution.agents.baseagent import BaseAgent
from modularcoevolution.managers.baseevolutionmanager import EvolutionEndedException
from modularcoevolution.utilities.specialtypes import EvaluationID, GenotypeID, claim_evaluation_id

from typing import Any, Optional, Sequence

import random

# if TYPE_CHECKING:
from modularcoevolution.genotypes.basegenotype import BaseGenotype
from modularcoevolution.utilities.datacollector import DataCollector
from modularcoevolution.generators.basegenerator import BaseGenerator



class Coevolution:
    """A manager for coevolutionary evolution.
    Performs evolution on multiple populations of agents, each of which is generated by a :class:`.BaseGenerator`.
    """
    agent_generators: list[BaseGenerator]
    """The generators used in evolution, typically representing different evolving populations of agents."""
    current_agents_per_generator: dict[BaseGenerator, list[GenotypeID]]
    """The set of agents present in the current generation for each generator."""
    player_generators: tuple[int, ...]
    """The generator index which each player is drawn from.
    For a two-player game, this would be either [0, 0] or [0, 1] depending on
    whether the two players are drawn from the same population."""

    evaluation_table: dict[EvaluationID, tuple[GenotypeID]]
    """A table of evaluations which have been assigned, mapped to the IDs of the agents in the evaluation."""
    remaining_evolution_evaluations: list[EvaluationID]
    """A list of main evolution evaluations which have not yet received results."""
    evaluation_results: dict[EvaluationID, dict[GenotypeID, dict[str, Any]]]
    """A table of evaluation results, mapped to the evaluation ID and the genotype ID."""

    run_tournament: bool
    """Whether to run the master tournament, which performs additional evaluations to measure intergenerational fitness.
    This can be used to diagnose problems with the coevolutionary dynamics.
    Keep in mind that the dynamics of fitness within a coevolutionary run do not map exactly to a global perspective of fitness for the problem."""
    remaining_tournament_evaluations: list[EvaluationID]
    """A list of tournament evaluations which have not yet received results."""
    tournament_buffer: list[EvaluationID]
    """A buffer of tournament evaluations which are being withheld until there are :attr:`tournament_batch_size` buffered.
    This can sometimes prevent parallel evaluations from waiting on a very small number of evaluations."""
    tournament_evaluations: int
    """The number of evaluations that will be performed between each pair of generations."""
    tournament_evaluation_count: dict[tuple[int, ...], int]
    """Counts how many times have the keyed generations been evaluated against each other."""
    tournament_batch_size: int
    """The number of tournament evaluations to buffer in the :attr:`tournament_buffer` before sending them to be evaluated.
    A good value for this is the number of parallel evaluation processes being run."""
    tournament_ratio: int
    """The tournament will only consider agents from generations which are multiples of this value.
    This is done to prevent combinatorial explosion of tournament evaluations."""
    tournament_results: dict[tuple[int, ...], dict[str, Any]]
    """A table of tournament results, mapped to the generation pair being compared."""
    tournament_generations: dict[EvaluationID, tuple[int, ...]]
    """Stores which generations are being compared in each tournament evaluation."""

    generation: int
    """The current generation of coevolution."""
    num_generations: int
    """The number of generations of coevolution to perform.
    
    Todo:
        * Allow the use of an evaluation limit, or an early termination criterion."""
    finalizing: bool
    """If true, evolution has completed, and the only remaining evaluations are for secondary measurements such as tournament evaluations."""
    evaluations_per_individual: int
    """The number of evaluations each agent will participate in per generation. This relates to the fraction of opponents each agent will encounter."""
    evaluated_groups: dict[tuple[GenotypeID], int]
    """A table of how many times each set of agents has been evaluated together. Unlisted sets have not been evaluated together."""
    opponents_this_generation: dict[GenotypeID, set[GenotypeID]]
    """A table of which opponents each agent has encountered this generation. This is used to avoid duplicate evaluations."""

    data_collector: Optional[DataCollector]
    """The data collector used to collect data, if present."""

    def __init__(self,
                 agent_generators: Sequence[BaseGenerator],
                 player_generators: Sequence[int],
                 num_generations: int,
                 evaluations_per_individual,
                 run_tournament=False,
                 tournament_evaluations=None,
                 tournament_batch_size=None,
                 tournament_ratio=1,
                 data_collector=None,
                 **kwargs):
        self.agent_generators = list(agent_generators)
        self.current_agents_per_generator = {generator: [] for generator in self.agent_generators}
        self.player_generators = tuple(player_generators)

        self.evaluation_table = {}
        self.evaluation_results = {}
        self.remaining_evolution_evaluations = []
        self.remaining_tournament_evaluations = []
        self.run_tournament = run_tournament
        self.tournament_buffer = []
        self.tournament_evaluations = tournament_evaluations
        if self.tournament_evaluations is None:
            self.tournament_evaluations = evaluations_per_individual
        self.tournament_evaluation_count = {}
        self.tournament_batch_size = tournament_batch_size
        if tournament_batch_size is None:
            self.tournament_batch_size = multiprocessing.cpu_count()
        self.tournament_ratio = tournament_ratio
        self.tournament_results = {}
        self.tournament_generations = {}

        self.generation = 0
        self.num_generations = num_generations
        self.finalizing = False
        self.evaluations_per_individual = evaluations_per_individual
        self.evaluated_groups = {}
        self.opponents_this_generation = {}

        self.data_collector = data_collector

        self.start_generation()

    # Causes the attacker and defender to both run their next generation code, and log the results of the last generation
    def next_generation(self) -> None:
        """Triggers all generators to start their next generation, clears the per-generation data, and calls :meth:`start_generation`."""
        if self.finalizing:
            raise EvolutionEndedException

        # TODO: Handle debug printing better.
        print("Starting next generation----------------------------------")
        for generator in self.agent_generators:
            generator.end_generation()
        if self.generation >= self.num_generations:
            self.finalizing = True
        else:
            self.generation += 1
            for generator in self.agent_generators:
                generator.next_generation()

        self.remaining_evolution_evaluations.clear()
        self.remaining_tournament_evaluations.clear()
        self.tournament_generations.clear()
        self.evaluated_groups.clear()
        self.opponents_this_generation.clear()

        self.start_generation()

    def start_generation(self) -> None:
        """Schedules all evaluations for the current generation."""
        if not self.finalizing:
            self.current_agents_per_generator = {generator: generator.get_individuals_to_test() for generator in self.agent_generators}
            self.add_coevolutionary_evaluations()
            # If the previous generation was a multiple of the tournament ratio, add the new tournament evaluations.
            # TODO: Consider a toggle between this and adding all tournament evaluations at the end.
            if self.run_tournament and (self.generation - 1) % self.tournament_ratio == 0:
                self.add_tournament_evaluations()
        # Add buffered evaluations in batches.
        while len(self.tournament_buffer) >= self.tournament_batch_size:
            for _ in range(self.tournament_batch_size):
                self.remaining_tournament_evaluations.append(self.tournament_buffer.pop(0))
        if self.finalizing:
            # Add all remaining tournament evaluations.
            self.remaining_tournament_evaluations.extend(self.tournament_buffer)
            self.tournament_buffer.clear()

    def add_coevolutionary_evaluations(self) -> None:
        """Generates and adds all coevolutionary evaluations for the current generation."""
        evaluation_groups = self.build_evaluation_groups()
        for group in evaluation_groups:
            evaluation_id = claim_evaluation_id()
            self.evaluation_table[evaluation_id] = group
            self.remaining_evolution_evaluations.append(evaluation_id)

    def build_evaluation_groups_careful(self) -> list[tuple[GenotypeID, ...]]:
        """Builds groups of agents to evaluate together, and returns them as a list of tuples of agent ids.
        The minimum number of groups will be formed such that each agent is evaluated :prop:`evaluations_per_individual` times.

        Returns: A list of evaluation groups, where each group is a list of agent ids for that evaluation.

        Todo:
            * Optionally discourage duplicate evaluations, which are useless for deterministic games.
            The random sorting limits this some already.
            Still, Duplicate evaluations can be higher than 10% of the total with this method.
            This could be done by storing multiple candidate groups and selecting the one with the least pairwise overlap.
            """
        groups = []
        agent_lists = [self.current_agents_per_generator[generator] for generator in self.agent_generators]
        # Queue entries are (evaluation count, tiebreaker value, agent ID)
        agent_queues = [[(0, random.random(), agent) for agent in agent_list] for agent_list in agent_lists]
        for agent_queue in agent_queues:
            heapq.heapify(agent_queue)

        minimum_evaluation_count = 0
        while minimum_evaluation_count < self.evaluations_per_individual:
            group = []
            for player, generator_index in enumerate(self.player_generators):
                agent_entry = heapq.heappop(agent_queues[generator_index])
                group.append(agent_entry[2])
                # Add the agent back to the queue with an incremented evaluation count and a new random tiebreaker.
                heapq.heappush(agent_queues[generator_index], (agent_entry[0] + 1, random.random(), agent_entry[2]))
            groups.append(tuple(group))
            minimum_evaluation_count = min(heapq.nsmallest(1, agent_queue)[0][0] for agent_queue in agent_queues)
        return groups

    def build_evaluation_groups(self) -> list[tuple[GenotypeID, ...]]:
        """Builds groups of agents to evaluate together, and returns them as a list of tuples of agent ids.
        Groups are generated at random without the usual checks for evenness.

        Returns: A list of evaluation groups, where each group is a list of agent ids for that evaluation.
        """
        groups = []
        agent_lists = [self.current_agents_per_generator[generator].copy() for generator in self.get_generator_order()]
        for agent_list in agent_lists:
            random.shuffle(agent_list)
        max_length = max(len(agent_list) for agent_list in agent_lists)
        for separation in range(self.evaluations_per_individual):
            for position in range(max_length):
                group = []
                for player_index, agent_list in enumerate(agent_lists):
                    index = (position + separation * player_index) % len(agent_list)
                    group.append(agent_list[index])
                groups.append(tuple(group))
        return groups


    def get_remaining_evaluations(self) -> list[EvaluationID]:
        """Gets a list of evaluations that need to be run.
        Used together with :meth:`get_agent_pair` to run evaluations.

        Returns: A list of evaluation IDs that have not yet been completed.
        """
        remaining_evaluations = self.remaining_evolution_evaluations.copy()
        remaining_evaluations.extend(self.remaining_tournament_evaluations)
        return remaining_evaluations

    def get_generator_order(self) -> list[BaseGenerator]:
        """Gets a list of generators that matches the schema of :prop:`player_generators`.

        Returns: A list of generators of length ``len(self.player_generators)`` corresponding to the origin of agents in a group."""
        return [self.agent_generators[generator_index] for generator_index in self.player_generators]

    def get_genotype_group(self, evaluation_id: EvaluationID) -> list[BaseGenotype]:
        """Gets the genotypes used in a given evaluation.

        Args:
            evaluation_id: The ID of the evaluation.

        Returns: A list of genotypes used in the evaluation."""
        return [generator.get_genotype_with_id(agent_id) for generator, agent_id in zip(self.get_generator_order(), self.evaluation_table[evaluation_id])]

    def build_agent_pair(self, *args, **kwargs):
        raise NotImplementedError("This method has been removed. Use build_agent_group instead.")

    def build_agent_group(self, evaluation_id: EvaluationID, active: bool = True) -> list[BaseAgent]:
        """Builds agents to be used in a given evaluation.

        Args:
            evaluation_id: The ID of the evaluation.
            active: Whether the agents should be active or inactive. See :prop:`BaseAgent.active` for more information.

        Returns: A list of genotypes used in the evaluation."""
        return [generator.build_agent_from_id(agent_id, active) for generator, agent_id in zip(self.get_generator_order(), self.evaluation_table[evaluation_id])]

    def send_objectives(self, *args, **kwargs):
        raise NotImplementedError("This method has been removed. Use submit_evaluation instead.")

    def submit_evaluation(self, evaluation_id, evaluation_results: dict[GenotypeID, dict[str, Any]]) -> None:
        """Submits the results of an evaluation to the relevant generators and agents.
        This method calls :meth:`preprocess_submit_evaluation`, and then :meth:`main_submit_evaluation`.

        Args:
            evaluation_id: The ID of the evaluation.
            evaluation_results: A dictionary of results from the evaluation, indexed by genotype ID.

        """
        self.preprocess_submit_evaluation(evaluation_id, evaluation_results)
        self.main_submit_evaluation(evaluation_id, evaluation_results)

    def preprocess_send_objectives(self, *args, **kwargs):
        raise NotImplementedError("This method has been removed. Use preprocess_submit_evaluation instead.")

    def preprocess_submit_evaluation(self, evaluation_id, evaluation_results: dict[GenotypeID, dict[str, Any]]) -> None:
        """Represents actions that should be taken before the evaluation results are sent to the agents.
        Intended for use by subclasses.

        Args:
            evaluation_id: The ID of the evaluation.
            evaluation_results: A dictionary of results from the evaluation, indexed by genotype ID.

        """
        self.evaluation_results[evaluation_id] = evaluation_results

        if self.data_collector is not None:
            #TODO: Refactor how the data collector is written to for evaluations.
            pass

        if evaluation_id in self.remaining_evolution_evaluations:
            agent_ids = self.evaluation_table[evaluation_id]
            for i, agent_id in enumerate(agent_ids):
                if agent_id not in self.opponents_this_generation:
                    self.opponents_this_generation[agent_id] = set()
                self.opponents_this_generation[agent_id].update(agent_ids[:i] + agent_ids[i + 1:])

    def main_send_objectives(self, *args, **kwargs):
        raise NotImplementedError("This method has been removed. Use main_submit_evaluation instead.")

    def main_submit_evaluation(self, evaluation_id, evaluation_results: dict[GenotypeID, dict[str, Any]]) -> None:
        """Store the results of an evaluation in the appropriate locations, and keep track of which have been performed.

        Args:
            evaluation_id: The ID of the evaluation.
            evaluation_results: A dictionary of results from the evaluation, indexed by genotype ID.

        """
        agent_group = self.evaluation_table[evaluation_id]
        if evaluation_id in self.remaining_evolution_evaluations:
            for generator, agent_id in zip(self.get_generator_order(), agent_group):
                generator.submit_evaluation(agent_id, evaluation_results[agent_id])
            self.evaluated_groups[agent_group] = self.evaluated_groups.setdefault(agent_group, 0) + 1
            self.remaining_evolution_evaluations.remove(evaluation_id)
        elif evaluation_id in self.remaining_tournament_evaluations:
            # TODO: Redesign tournament to work in multiple dimensions

            self.remaining_tournament_evaluations.remove(evaluation_id)
            self.write_tournament_data()

        if self.data_collector is not None:
            self.data_collector.set_evaluation_data(evaluation_id, agent_group, evaluation_results)

    def write_tournament_data(self):
        raise NotImplementedError("Tournaments are not currently supported, sorry.")

    def validate_parameters(self) -> None:
        """Checks for potential issues with the parameters of the experiment and alerts the user."""
        total_estimated_evolution_evaluations = max(*self.agent_generators, key=lambda generator: generator.population_size) * self.num_generations * self.evaluations_per_individual
        total_tournament_evaluations = ((self.num_generations // self.tournament_ratio) ** len(self.player_generators)) * self.tournament_evaluations
        if total_tournament_evaluations > total_estimated_evolution_evaluations * 4:
            raise Warning(f"The number of tournament evaluations scheduled ({total_tournament_evaluations}) is much higher than the expected number of evolution evaluations scheduled."
                          f"Check tournament_ratio, or consider disabling tournaments for problems with many populations.")