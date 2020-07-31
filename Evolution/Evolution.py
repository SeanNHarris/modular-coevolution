from Evolution.BaseAgent import *
from Evolution.EvolutionGenerator import EvolutionGenerator
import os


class EvolutionWrapper:
    def __init__(self, agent_generator, num_generations=50, resume=False):
        assert isinstance(agent_generator, EvolutionGenerator)
        self.agent_generator = agent_generator
        self.num_generations = num_generations

        self.agent_id = -1
        self.generation = 0

        if not os.path.exists("Logs"):
            os.makedirs("Logs")
        self.fitness_log = open("Logs/fitnessLog.txt", "a+")
        self.solution_log = open("Logs/solutionLog.txt", "a+")
        self.miscellaneous_log = open("Logs/miscellaneousLog.txt", "a+")

        if not resume:
            self.fitness_log.truncate(0)
            self.fitness_log.write("Gen.\tBest\tAvg.\n")
            self.solution_log.truncate(0)
            self.solution_log.write("Gen.\tFit.\tBest\n")
            self.miscellaneous_log.truncate(0)
            self.miscellaneous_log.write("Gen.\tDiv.\n")

    def next_generation(self):
        print("Writing logs...")
        total_fitness = 0
        best_index = None
        best_fitness = None
        for i in range(self.agent_generator.population_size):
            fitness = self.agent_generator.get_fitness(i)
            if fitness is None:
                raise Exception("Fitness not set for population member with ID " + str(i))
            if best_index is None or fitness > best_fitness:
                best_index = i
                best_fitness = fitness
            total_fitness += fitness
        average_fitness = total_fitness / self.agent_generator.population_size
        diversity = self.agent_generator.get_diversity()
        self.fitness_log.write(str(self.generation) + "\t" + str(best_fitness) + "\t" + str(average_fitness) + "\n")
        self.solution_log.write(str(self.generation) + "\t" + str(best_fitness) + "\t" + str(
            self.agent_generator[best_index].parameterString()) + "\n")
        self.miscellaneous_log.write(str(self.generation) + "\t" + str(diversity) + "\n")

        self.fitness_log.flush()
        self.solution_log.flush()
        self.miscellaneous_log.flush()

        print("Starting next generation----------------------------------")
        self.generation += 1
        self.agent_id = -1
        if self.generation >= self.num_generations:
            self.finalize_results()
            raise EvolutionEndedException()

        self.agent_generator.next_generation()

    def next_agent(self, continue_to_next_generation=True):
        self.agent_id += 1
        if continue_to_next_generation and self.check_generation_end():
            self.next_generation()
        agent = self.agent_generator[self.agent_id]
        return agent

    # Returns a numbered dictionary (might not be complete) of the whole rest of the generation's individuals not yet returned. Restart ensures it is complete.
    def collect_generation(self, restart=False):
        if self.check_generation_end():
            self.next_generation()
        if restart:
            self.agent_id = -1
        agents = dict()
        while self.agent_id < self.agent_generator.population_size - 1:
            agents[self.agent_id] = self.next_agent(False)
        return agents

    def send_fitness(self, agent_fitness, target_agent_id=None):
        if target_agent_id is None:
            target_agent_id = self.agent_id
        self.agent_generator.set_fitness(target_agent_id, agent_fitness)

    def check_generation_end(self):
        return self.agent_id >= self.agent_generator.population_size - 1

    def finalize_results(self):
        self.agent_generator.finalize_results()

        self.solution_log.write("---Final Results---\n")
        self.solution_log.write("id\tFit.\tBest\n")
        for i in range(self.agent_generator.population_size):
            self.solution_log.write(str(i) + "\t" + str(self.agent_generator.get_fitness(i)) + "\t" + str(
                self.agent_generator[i].parameterString()) + "\n")
        self.solution_log.flush()


class EvolutionEndedException(Exception):
    pass
