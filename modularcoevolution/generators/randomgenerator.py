#  Copyright 2025 BONSAI Lab at Auburn University
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

__author__ = 'Sean N. Harris'
__copyright__ = 'Copyright 2025, BONSAI Lab at Auburn University'
__license__ = 'Apache-2.0'

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