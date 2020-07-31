class DataCollector:
    def __init__(self):
        self.data = dict()
        self.data["experiment"] = dict()
        self.data["experiment"]["masterTournamentObjectives"] = dict()
        self.data["generations"] = dict()
        self.data["individuals"] = dict()
        self.data["evaluations"] = dict()

    def update_experiment(self):
        pass

    def set_experiment_parameters(self, parameters):
        self.data["experiment"]["parameters"] = parameters
        self.update_experiment()

    def set_experiment_master_tournament_objective(self, objective, matrix):
        self.data["experiment"]["masterTournamentObjectives"][objective] = matrix
        self.update_experiment()
    
    def set_generation_data(self, agent_type, generation, individual_IDs, objective_statistics, metric_statistics):
        if agent_type not in self.data["generations"]:
            self.data["generations"][agent_type] = dict()
        self.data["generations"][agent_type][generation] = dict()
        self.data["generations"][agent_type][generation]["individualIDs"] = individual_IDs
        self.data["generations"][agent_type][generation]["objectiveStatistics"] = objective_statistics
        self.data["generations"][agent_type][generation]["metricStatistics"] = metric_statistics
        self.update_experiment()

    def set_individual_data(self, agent_type, ID, genotype, evaluation_IDs, opponent_IDs, objective_statistics, metrics, parent_IDs, creation_information):
        if agent_type not in self.data["individuals"]:
            self.data["individuals"][agent_type] = dict()
        self.data["individuals"][agent_type][ID] = dict()
        self.data["individuals"][agent_type][ID]["genotype"] = genotype
        self.data["individuals"][agent_type][ID]["evaluationIDs"] = evaluation_IDs
        self.data["individuals"][agent_type][ID]["opponentIDs"] = opponent_IDs
        self.data["individuals"][agent_type][ID]["objectiveStatistics"] = objective_statistics
        self.data["individuals"][agent_type][ID]["metrics"] = metrics
        self.data["individuals"][agent_type][ID]["parentIDs"] = parent_IDs
        self.data["individuals"][agent_type][ID]["creationInformation"] = creation_information
        self.update_experiment()

    # TODO: Rename attacker and defender to be more generic, expand to more (or less) than a fixed two agents
    def set_evaluation_data(self, ID, attacker_ID, defender_ID, attacker_objectives, defender_objectives):
        self.data["evaluations"][ID] = dict()
        self.data["evaluations"][ID]["attackerID"] = attacker_ID
        self.data["evaluations"][ID]["defenderID"] = defender_ID
        self.data["evaluations"][ID]["attackerObjectives"] = attacker_objectives
        self.data["evaluations"][ID]["defenderObjectives"] = defender_objectives
        self.update_experiment()

    def __getstate__(self):
        state = self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)