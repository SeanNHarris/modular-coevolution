import gzip
import json
import os

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

    def set_individual_data(self, agent_type, ID, genotype, evaluation_IDs, objective_statistics, metrics, parent_IDs,
                            creation_information):
        if agent_type not in self.data["individuals"]:
            self.data["individuals"][agent_type] = dict()
        self.data["individuals"][agent_type][ID] = dict()
        self.data["individuals"][agent_type][ID]["genotype"] = genotype
        self.data["individuals"][agent_type][ID]["evaluationIDs"] = evaluation_IDs
        self.data["individuals"][agent_type][ID]["objectiveStatistics"] = objective_statistics
        self.data["individuals"][agent_type][ID]["metrics"] = metrics
        self.data["individuals"][agent_type][ID]["parentIDs"] = parent_IDs
        self.data["individuals"][agent_type][ID]["creationInformation"] = creation_information
        self.update_experiment()

    def set_evaluation_data(self, ID, agent_IDs: dict, agent_objectives: dict):
        self.data["evaluations"][ID] = dict()
        self.data["evaluations"][ID]["agentIDs"] = agent_IDs
        self.data["evaluations"][ID]["agentObjectives"] = agent_objectives
        self.update_experiment()

    def save_to_file(self, filename, clear_memory=False):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with gzip.open(filename, 'wt+', encoding='UTF-8') as log_file:
            json.dump(self.data, log_file, cls=StringDefaultJSONEncoder)

        if clear_memory:
            self.data["individuals"] = dict()
            self.data["evaluations"] = dict()

    def load_from_file(self, filename):
        with gzip.open(filename, 'rt', encoding='UTF-8') as log_file:
            new_data = json.load(log_file)
            for table in new_data:
                if table in ["generations", "individuals"]:
                    for agent_type in new_data[table]:
                        if agent_type not in self.data[table]:
                            self.data[table][agent_type] = dict()
                        self.data[table][agent_type].update(new_data[table][agent_type])
                else:
                    self.data[table].update(new_data[table])

    def load_directory(self, pathname):
        files = [file for file in os.scandir(pathname) if file.is_file()]
        files.sort(key=lambda file: int("".join(filter(str.isdigit, file.name))))
        for file in files:
            self.load_from_file(file.path)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class StringDefaultJSONEncoder(json.JSONEncoder):
    def default(self, o):
        try:
            return super().default(o)
        except TypeError:
            return str(o)
