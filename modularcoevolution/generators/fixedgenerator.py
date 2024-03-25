class FixedGenerator(BaseGenerator):
    def __init__(self, agentClass, agentParameters):
        super().__init__()
        self.agentClass = agentClass
        self.agentParameters = agentParameters
        self.population_size = 1

        self.generation = 0
        self.agent = self.agentClass(agentParameters, active=False)

    def __getitem__(self, item):
        return self.agent

    def get_from_generation(self, generation, index):
        return self.agent

    def set_fitness(self, index, fitness, average=False):
        self.agent.genotype.set_fitness(fitness, average=average)

    def get_fitness(self, index):
        return self.agent.genotype.fitness

    def getDiversity(self, referenceID = None):
        return 0

    def next_generation(self):
        self.generation += 1
        self.agent.genotype.fitness = None
        self.agent.genotype.fitnessSet = False
        self.agent.genotype.fitnessCount = 0

    def finalize_results(self):
        print("Static agent:")
        print(self.agent)
