# Evolution
This subdirectory contains the immediate components needed to run evolution or coevolution.
In particular, classes for evolutionary algorithms are located here, as well as several related abstract base classes.

### Abstract Base Classes
The following abstract base classes are used for this:

- `BaseGenotype` is the base class of all genotypes.
    These genotypes must be able to `mutate` themselves, `recombine` with another genotype of the same class, and `clone` themselves safely.
    Further, a static `diversity_function` must be provided to calculate their novelty compared to a population.
    `get_raw_genotype` must be implemented to return a dictionary of parameters that would produce a genetically identical copy of this individual through `__init__`.
    A `__hash__` function must be provided.
    They can optionally be given modifiers like parsimony pressure through `get_fitness_modifier`.
    
    A genotype stores a list of `objectives` from its evaluations.

    The method `set_objectives` is provided for setting objective scores obtained through evaluation. It has the following parameters:
    - `objective_list` - A dictionary mapping objective names to objective values
    - `average_flags` - A dictionary mapping objective names to a boolean determining whether they should be maintained as a running average (true), or replaced with the new value (false). Defaults to false for unlisted objectives.
    - `inactive_objectives` - A list of objective names that will be tracked, but not used for evolution.
    
    Genotypes generate a unique `ID` that will differ between all individuals across all subclasses.

    Genotypes are instantiated with `BaseGenotype(parameters)`.

- `BaseAgent` is the base class of all agents.
    Agents have a dictionary of parameters associated with them, which usually includes a genotype.
    Agents must be able to `get_parameters`, `apply_parameters`, and return a `parameter_string` that fully describes their unique parameters.
    Most importantly, agents need a `perform_action` method which gets called repeatedly over the course of an evaluation, prompting them to select and perform an action.
    Agents have the variable `active` which can be disabled to prevent actions.

    Agents are instantiated with `BaseAgent(parameters=None, active=True, *args, **kwargs)`.

- `BaseEvolutionaryAgent` is a type of agent that can be used for evolution.
    Non-evolutionary agents must be submitted to the evolution wrapper through a non-evolutionary generator.
    Evolutionary agents must additionally provide class methods that give a `genotype_class` and `genotype_default_parameters` (the default parameters to `__init__` that genotype) to be used with that agent type.

    Evolutionary agents are instantiated with `BaseEvolutionaryAgent(parameters = None, genotype = None, *args, **kwargs)`. If `genotype` is not provided, one will be generated according to `parameters` and the default genotype parameters for the class.

- `BaseGenerator` is the base class of all agent generators (such as an evolutionary algorithm).
    All generators must have some data structure to store a population of one or more `BaseAgent`s, possibly across several generations.
    This data structure should be reflected in their `populationSize` field, and should be accessible through `__getitem__`, indexing a consistent individual from the population by position (referred to as ID).
    If the population changes per generation, then `getFromGeneration` should be able to index by generation; otherwise, it can ignore the generation parameter and return the result of `__getitem__`.
    The generator should be able to `setFitness` and `getFitness` per ID, even if the agents generated are not evolutionary.
    The generator needs a `nextGeneration` function called at the end of every generation of evolution, that may do nothing.
    `finalizeResults` will be called at the end of an experiment to allow for final operations.
    A `generateIndividual` method should be provided to generate an agent of the appropriate type given a parameter list.

### Other Classes
Other classes here include:

- `BaseEvolutionaryGenerator` is a `BaseGenerator` with a partial implementation of evolutionary algorithm behavior, for extension by different kinds of evolutionary algorithm.
    This class handles generation of initial populations, and implements `get_from_generation`, `get_individuals_to_test`, `set_objectives`, `generate_individual`.
    It is configured with the following parameters: `BaseEvolutionaryGenerator(agent_class, initial_size=1, seed=None, fitness_function=None, data_collector=None, copy_survivor_objectives=False, using_hall_of_fame=True)`
    
    - `agentClass` - A subclass of `BaseEvolutionaryAgent` to be used for evolution.
    - `initial_size` - The starting population size, which will generally correspond to mu.
    - `seed` - If not None, sets half of the population's genotype parameters to those given.
    - `fitness_function` - An optional function that produces a single value based on the set of objectives, which will be recorded as a metric.
    - `data_collector` - An optional `DataCollector` object to store detailed log data.
    - `copy_survivor_objectives` - Whether to reuse old objective values in future generations, not recommended for coevolution.
    - `using_hall_of_fame` - Whether to maintain and include a hall of fame in the set of individuals to be sampled for evaluation. Only useful for coevolution.


- `EvolutionGenerator` is a `BaseEvolutionaryGenerator` configured as a genetic programming-style evolutionary algorithm.
    It is configured with the following parameters: `EvolutionGenerator(agent_class, initial_size=1, children_size=1, mutation_fraction=0.25,                 recombination_fraction=0.75,
                 parsimony_weight=0, diversity_weight=0, diverse_elites=False, seed=None, fitness_function=None,
                 data_collector=None, copy_survivor_objectives=False, using_hall_of_fame=True)`

    - `agentClass` - A subclass of `BaseEvolutionaryAgent` to be used for evolution.
    - `initial_size` - The starting population size and size the population is pruned to each generation, mu.
    - `children_size` - The number of children born per generation, lambda.
    - `mutation_fraction` - The proportion of the population that is born through mutation.
    - `recombination_fraction` - The proportion of the population that is born through recombination.
    - `parsimony_weight` - The weight of penalties due to `getFitnessModifier`.
    - `diversity_weight` - The weight of bonuses due to novelty.
    - `diverse_elites` - If false, diversity will only affect parent selection. If true, it will also affect survival selection.
    - `seed` - If not None, sets half of the population's genotype parameters to those given.
    - `fitness_function` - An function that produces a single value based on the set of objectives, which will be recorded as fitness. This is mandatory for `EvolutionGenerator`.
    - `data_collector` - An optional `DataCollector` object to store detailed log data.
    - `copy_survivor_objectives` - Whether to reuse old objective values in future generations, not recommended for coevolution.
    - `using_hall_of_fame` - Whether to maintain and include a hall of fame in the set of individuals to be sampled for evaluation. Only useful for coevolution.

- `NSGAIIGenerator` is a `BaseEvolutionaryGenerator` configured to run the NSGA-II multiobjective evolutionary algorithm.
    It is configured with the following parameters: `agent_class, initial_size=1, children_size=1, mutation_fraction=0.25, recombination_fraction=0.75, tournament_size=2, seed=None, fitness_function=None, data_collector=None, copy_survivor_objectives=False, using_hall_of_fame=True`
    - `agentClass` - A subclass of `BaseEvolutionaryAgent` to be used for evolution.
    - `initial_size` - The starting population size and size the population is pruned to each generation, mu.
    - `children_size` - The number of children born per generation, lambda.
    - `mutation_fraction` - The proportion of the population that is born through mutation.
    - `recombination_fraction` - The proportion of the population that is born through recombination.
    - `tournament_size` - The size of tournaments to be used for k-tournament selection.
    - `seed` - If not None, sets half of the population's genotype parameters to those given.
    - `fitness_function` - An optional function that produces a single value based on the set of objectives, which will be recorded as a metric.
    - `data_collector` - An optional `DataCollector` object to store detailed log data.
    - `copy_survivor_objectives` - Whether to reuse old objective values in future generations, not recommended for coevolution.
    - `using_hall_of_fame` - Whether to maintain and include a hall of fame in the set of individuals to be sampled for evaluation. Only useful for coevolution.

- `Coevolution` handles the logistics of coevolution between two agent generators.
    It is configured with the following parameters: `Coevolution(attacker_generator, defender_generator, num_generations, evaluations_per_individual, tournament_evaluations=None, tournament_batch_size=4, tournament_ratio=1, resume=False)`

    - `attacker_generator` - A `BaseGenerator` for the attacker agents
    - `defender_generator` - A `BaseGenerator` for the defender agents
    - `num_generations` - The number of generations to run
    - `evaluations_per_individual` - Each individual will receive this many evaluations, paired with a different opponent each time if possible; its fitness will be the average of these
    - `tournament_evaluations` - Evaluations per pairing in the master tournament (to reduce noise); defaults to `evaluations_per_individual`
    - `tournament_batch_size` - Tournament evaluations can be scheduled freely through the experiment; this parameter ensures the number of tournament evaluations per generation is a multiple of the batch size (which is ideally the number of parallel evaluations available)
    - `tournament_ratio` - Only run tournament evaluations for generations divisible by this number; significantly decreases evaluation count since tournament evaluation count is quadratic to generation count
    - `data_collector` - An optional `DataCollector` object to store detailed log data.
    - `log_subfolder` - A subfolder in the `Logs` folder to store results in. If not provided, results will be stored directly in `Logs`.
    
    To use 

- `EloCoevolution` is a subclass of `Coevolution` that performs coevolution using Elo ratings for pairing individuals, by organizing evaluation into several rounds and trying to pair individuals close in Elo rating, which is updated after each round.
    It has the following additional parameters: `k_factor=40, elo_pairing=True, elo_ranking=False`
    - `k_factor` - The k-factor used for Elo rating calculation. Represents how much an Elo score is updated after each evaluation.
    - `elo_pairing` - Whether to perform Elo-based pairings. If this is false, Elo ratings will be recorded but not used for pairing.
    - `elo_ranking` - Whether to pair based on the ordinal rank of each individual's Elo score instead of the raw Elo score.

- `StaticEloCoevolution` is a subclass of `EloCoevolution` that calculates Elo ratings using a static, iterative algorithm that takes into account transitive relationships, and which is more stable than rating updates for small numbers of evaluations.
    Elo is recalculated after each round using the full match history for this generation.
    
- `DataCollector` is a class that stores a variety of detailed data about the evolutionary process, to be logged to a single file for easy analysis. The exact format is only partially specified.
    It contains the following methods for storing data:
    - `set_experiment_parameters(parameters)` - Store the set of parameters used for the experiment.
    - `set_experiment_master_tournament_objective(objective, matrix)` - Store the master tournament `matrix` associated with `objective`.
    - `set_generation_data(agent_type, generation, individual_IDs, objective_statistics, metric_statistics)` - Store data about a completed generation for the agent generator for population `agent_type`.
    `generation` is a generation number, `individual_IDs` is a list of `ID`s for individuals in the population at this time, and `objective_statistics` and `metric_statistics` are dictionaries storing data about population statistics for each objective and metric.
    - `set_individual_data(agent_type, ID, genotype, evaluation_IDs, opponent_IDs, objective_statistics, metrics, parent_IDs, creation_information)` - Store data about a single individual from population `agent_type`.
    `ID` is that individual's genotype `ID`, and `genotype` is its `get_raw_genotype`.
    `evaluation_IDs` and `opponent_IDs` are ordered lists of the `ID`s of evaluations and associated opponents that this individual participated in.
    `objective_statistics` and `metrics` are that individual's `objective_statistics` and `metrics`.
    `parent_IDs` and `creation_information` record a list of parent genotype `ID`s (if any) and a string describing what method was used to generate the individual.
    - `set_evaluation_data(ID, attacker_ID, defender_ID, attacker_objectives, defender_objectives)` - Store the data about a single evaluation.
    `ID` is a unique ID associated with the evaluation (which should match with the `evaluation_IDs` in `set_individual_data`).
    `attacker_ID` and `defender_ID` are the `ID`s of the two individuals involved in this evaluation (modification is needed to support values other than two).
    `attacker_objectives` and `defender_objectives` are the objective name to value dictionaries for the two agents.
    
    Additionally, `update_experiment` is called after each update, which does nothing in the base implementation but is useful for subclasses which need to keep a log file or database updated in realtime.
    `DataCollector` does not maintain a log file itself, but instead exposes its `data` parameter which is a single nested dictionary with all of its data, which can be stored as `JSON` or a similar format.
    
- `AgentTypeRegistry` stores a global dictionary `name_lookup` mapping the string names of any imported agent classes to the class object, allowing instantiation from a string description.
    