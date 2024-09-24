import ast

from modularcoevolution.agents.baseevolutionaryagent import BaseEvolutionaryAgent
from modularcoevolution.generators.basegenerator import BaseGenerator


class EvolutionaryHillClimberGenerator(BaseGenerator):
	def __init__(self, agentClass, parallelEvaluations = 1):
		super().__init__()
		self.agentClass = agentClass
		assert issubclass(agentClass, BaseEvolutionaryAgent)
		self.genotypeClass = agentClass.genotype_class()

		self.parallelEvaluations = parallelEvaluations
		self.population_size = parallelEvaluations
		self.population = list()
		self.pastPopulations = list()
		for _ in range(parallelEvaluations):
			self.population.append(self.genotypeClass(self.agentClass.genotype_default_parameters()))

	def __getitem__(self, item):
		agent = self.agentClass(genotype = self.population[item], active = False)
		agent.scanTime = 4
		return agent

	def set_fitness(self, index, fitness, average=False):
		fitnessModifier = self.population[index].get_fitness_modifier()
		self.population[index].set_fitness(fitness + fitnessModifier, average)

	def get_fitness(self, index):
		return self.population[index].fitness

	def get_from_generation(self, generation, index):
		agent = self.agentClass(genotype = self.pastPopulations[generation][index])
		agent.scanTime = 4
		return agent

	#Creates a new generation
	def next_generation(self):
		self.population.sort(key = lambda x: x.fitness, reverse = True) #High fitness is good

		print("Best individual of this iteration: (fitness score of " + str(self.population[0].fitness))
		print(self.population[0])

		print(str([individual.fitness for individual in self.population]))

		nextGeneration = list()
		nextGeneration.append(self.population[0].clone())

		for _ in range(self.parallelEvaluations):
			child = self.population[0].clone(copy_objectives=False)
			child.mutate()
			nextGeneration.append(child)

		self.pastPopulations.append(self.population)
		self.population = nextGeneration
		self.population_size = 1 + self.parallelEvaluations

	#Prints final data and sorts the end result
	def finalize_results(self):
		self.population.sort(key = lambda x: x.fitness, reverse = True)
		self.pastPopulations.append(self.population)

		print("Experiment complete.")
		print("Best individual of this experiment: (fitness score of " + str(self.population[0].fitness) + ")")
		print(str(self.population[0]))

	def generate_individual(self, parameter_string):
		idList = ast.literal_eval(parameter_string)
		return self.agentClass(self.genotypeClass(idList = idList))