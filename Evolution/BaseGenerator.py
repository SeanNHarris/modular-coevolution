import abc


# The BaseGenerator is the superclass of all generators which participate in co-evolution, both attackers and defenders.
# One of BaseAttackerGenerator or BaseDefenderGenerator must be implemented by generators used in co-evolution.
class BaseGenerator:
    __metaclass__ = abc.ABCMeta

    # Inheriting classes should call this.
    def __init__(self):
        self._population_size = -1
        self.next_ID = 0
        self.ID_table = dict()

    @property
    def population_size(self):
        assert self._population_size > 0  # The inheriting class MUST set a population size, even if it's just one
        return self._population_size

    @population_size.setter
    def population_size(self, value):
        self._population_size = value
    
    # Associates the next available ID with the given individual in the ID table, and returns it.
    # These are unique within a generator. If appropriate, 
    def claim_ID(self, individual):
        claimed_ID = self.next_ID
        self.ID_table[claimed_ID] = individual
        self.next_ID += 1
        return claimed_ID

    # Return the individual from the population with index ID, where ID will be an integer less than the population size
    # If there is not a list of individuals being kept and instead a single reconfigurable one with a population of parameters, an implementation might just set the parameters and return the one individual.
    @abc.abstractmethod
    def __getitem__(self, index):
        pass

    # Return an individual with given index from a previous generation of the given number. Previous populations must be stored for coevolution.
    # If the generator is not generational, the normal method of selecting an individual can be used.
    @abc.abstractmethod
    def get_from_generation(self, generation, index):
        pass

    # Return a set of high-quality representatives of the population from a certain generation, for intergenerational comparisons.
    # Duplicates can be returned if necessary, given the amount. If force is not set, fewer representatives may be returned.
    @abc.abstractmethod
    def get_representatives_from_generation(self, generation, amount, force=False):
        pass

    # Called by co-evolution to record objective results from evaluations for the individual with given index, with the intent that it be internally stored in the generator.
    # It is the responsibility of the generator to deal with what to do when multiple fitnesses are returned, such as averaging them.
    # Fitness can be calculated here if desired.
    @abc.abstractmethod
    def set_objectives(self, index, objectives, average_flags=None, average_fitness=False, opponent=None, evaluation_number=None, inactive_objectives=None):
        pass

    # Signals the generator that a generation of co-evolution has completed. Generational generators such as evolutionary algorithms should generate a new generation.
    # If the generator is not generational, nothing needs to be done.
    # Two log files may be passed for logging statistics and agent data, but might be None, so check.
    @abc.abstractmethod
    def next_generation(self, result_log=None, agent_log=None):
        pass

    # Returns a new individual of the type associated with the generator based on the parameter string passed.
    # The requested format of the string should correspond to the individual's parameter string
    @abc.abstractmethod
    def generate_individual(self, parameter_string):
        pass