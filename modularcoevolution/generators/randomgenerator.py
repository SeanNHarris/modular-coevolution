from .basegenerator import BaseGenerator


class RandomGenerator(BaseGenerator):
	def __init__(self, agentClass, totalAgents = 900, parallelEvaluations = 1):
		super().__init__()
		self.totalAgents = totalAgents
		self.population_size = parallelEvaluations
		self.agentClass = agentClass

		self.generation = 0
		self.agents = list()
		self.fitnesses = dict()

		for _ in range(totalAgents):
			self.agents.append(self.agentClass(active = False))

		self.bestAgent = None
		self.bestFitness = 0

	def __getitem__(self, item):
		return self.agents[self.generation * self.population_size + item]

	def get_from_generation(self, generation, index):
		return self.agents[generation * self.population_size + index]

	def set_fitness(self, index, fitness, average=False):
		if average:
			raise NotImplementedError("Fitness averaging is not implemented yet for random agents because they're coded weirdly.")  # TODO: Implement fitness averaging for random
		linearIndex = self.generation * self.population_size + index
		self.fitnesses[linearIndex] = fitness
		print("Agent " + str(linearIndex) + " fitness: " + str(self.fitnesses[linearIndex]))
		if self.bestAgent is None or fitness > self.bestFitness:
			self.bestAgent = self.agents[linearIndex]
			self.bestFitness = fitness
			print("New best fitness!")
			print(self.bestAgent)

	def get_fitness(self, index):
		return self.agents[index].fitness

	def next_generation(self):
		self.generation += 1

	def finalize_results(self):
		print("Best agent has fitness: " + str(self.bestAgent.fitness))
		print(self.bestAgent)