#  Copyright 2026 BONSAI Lab at Auburn University
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
__copyright__ = 'Copyright 2026, BONSAI Lab at Auburn University'
__license__ = 'Apache-2.0'

import cmd
import colorsys

from modularcoevolution.experiments.baseexperiment import BaseExperiment
from modularcoevolution.genotypes.basegenotype import BaseGenotype
from modularcoevolution.postprocessing import postprocessingutils
from modularcoevolution.utilities import commandlineutils
from modularcoevolution.utilities.datacollector import DataSchema, GenerationData

"""Command line tool to interactively view genotypes from experiment logs."""




class GenotypeBrowser(cmd.Cmd):
    experiment: BaseExperiment
    experiment_data: DataSchema
    genotype_cache: dict[int, BaseGenotype]
    """Previously loaded genotypes, keyed by (original) genotype ID."""

    recent_genotypes: list[int]
    """A list of recently mentioned genotype IDs, most recent first.
    Allows autocompletion of genotype IDs."""
    population: str
    """The population context to use for commands."""

    min_metrics: dict[str, float]
    """The minimum observed value for each metric among loaded genotypes, for normalization."""
    max_metrics: dict[str, float]
    """The maximum observed value for each metric among loaded genotypes, for normalization."""

    def __init__(self, experiment: BaseExperiment, experiment_data: DataSchema):
        super().__init__()
        self.experiment = experiment
        self.experiment_data = experiment_data
        self.genotype_cache = {}
        self.recent_genotypes = []
        self.population = None  # Set in `preloop` to the first population.
        self.min_metrics = {}
        self.max_metrics = {}

    def _load_genotype(self, genotype_id: int) -> BaseGenotype | None:
        if genotype_id in self.genotype_cache:
            return self.genotype_cache[genotype_id]

        for population_name in self.experiment_data['individuals']:
            if genotype_id not in self.experiment_data['individuals'][population_name]:
                continue
            individual_data = self.experiment_data['individuals'][population_name][genotype_id]

            genotype_parameters = individual_data['genotype']
            genotype = self.experiment.create_test_genotype(population_name, genotype_parameters)

            genotype.objective_tracker.metrics = individual_data['metrics']
            genotype.objective_tracker.metric_statistics = individual_data['metric_statistics']
            genotype.objective_tracker.metric_histories = individual_data['metric_histories']
            genotype.parent_ids = individual_data['parent_ids']
            genotype.creation_method = individual_data['creation_information']

            self.genotype_cache[genotype_id] = genotype
            self._update_metric_ranges(genotype)

            return genotype
        return None

    def _parse_genotype_id(self, arg: str) -> tuple[int, BaseGenotype] | tuple[None, None]:
        if arg == "":
            # Default to the most recently mentioned genotype.
            if len(self.recent_genotypes) == 0:
                print("No recent genotypes to fill for empty command.")
                return None, None
            genotype_id = self.recent_genotypes[0]
            genotype = self._load_genotype(genotype_id)
            print(f"Filling in most recent genotype {genotype_id}.")
            return genotype_id, genotype

        try:
            genotype_id = int(arg)
        except ValueError:
            print(f"Expected a genotype ID (integer), but got '{arg}' instead.")
            return None, None

        genotype = self._load_genotype(genotype_id)
        if genotype is None:
            print(f"A genotype with ID {genotype_id} was not found in any population.")
            return None, None
        return genotype_id, genotype

    def _parse_generation(self, arg: str) -> int | None:
        if self.population is None:
            print("No population set. Use the 'population' command to set a population context first.")
            return None

        def list_generations():
            generations = sorted(self.experiment_data['generations'][self.population].keys())
            print(f"{self.population.upper()} generations: {min(generations)} - {max(generations)}")

        if arg == "":
            list_generations()
            return None

        try:
            generation = int(arg)
        except ValueError:
            print(f"Expected a generation number (integer), but got '{arg}' instead.")
            return None

        if generation not in self.experiment_data['generations'][self.population]:
            print(f"Generation '{generation}' not found for {self.population}.")
            list_generations()
            return None

        return generation

    def _store_recent_genotype_id(self, genotype_id: int):
        if genotype_id in self.recent_genotypes:
            self.recent_genotypes.remove(genotype_id)
        self.recent_genotypes.insert(0, genotype_id)
        if len(self.recent_genotypes) > 10:
            self.recent_genotypes.pop()

    def _set_population(self, population_name: str):
        if population_name in self.experiment_data['individuals']:
            self.population = population_name
            self.prompt = f"({self.population}) "
        else:
            raise ValueError(f"Population name '{population_name}' does not exist.")

    def _str_objectives(self, genotype: BaseGenotype, color: bool = False) -> str:
        objectives = genotype.objective_tracker.metrics  # Todo: the objectives list is not populated; using metrics here.
        objective_strings = [f"{name}: {value:.4f}" for name, value in objectives.items()]
        if color:
            objective_strings = [self._color_by_metric(string, name, value) for string, (name, value) in zip(objective_strings, objectives.items())]
        return ', '.join(objective_strings)

    def _update_metric_ranges(self, genotype: BaseGenotype):
        for metric_name, metric_value in genotype.objective_tracker.metrics.items():
            if metric_name not in self.min_metrics or metric_value < self.min_metrics[metric_name]:
                self.min_metrics[metric_name] = metric_value
            if metric_name not in self.max_metrics or metric_value > self.max_metrics[metric_name]:
                self.max_metrics[metric_name] = metric_value

    def _color_by_metric(self, text: str, metric_name: str, metric_value: float) -> str:
        if metric_name not in self.min_metrics or metric_name not in self.max_metrics:
            return text

        min_value = self.min_metrics[metric_name]
        max_value = self.max_metrics[metric_name]
        if max_value == min_value:
            return text

        normalized_value = (metric_value - min_value) / (max_value - min_value)
        rgb = colorsys.hsv_to_rgb(normalized_value / 3, 1, 1)  # Between red and green.
        red = int(rgb[0] * 255)
        green = int(rgb[1] * 255)
        blue = int(rgb[2] * 255)
        return commandlineutils.color_string_24_bit(text, red, green, blue)

    def complete(self, text, state):
        return super().complete(text, state)

    def completedefault(self, text, line, begidx, endidx):
        """Autocompletion for genotype IDs based on recently mentioned genotypes."""
        if text == "":
            completions = [str(genotype_id) for genotype_id in self.recent_genotypes]
        else:
            completions = [str(genotype_id) for genotype_id in self.recent_genotypes if str(genotype_id).startswith(text)]
        return completions

    def do_id(self, arg):
        """Accept a genotype ID and display the corresponding genotype."""
        genotype_id, genotype = self._parse_genotype_id(arg)
        if genotype is None:
            return

        print(genotype)
        self._store_recent_genotype_id(genotype.id)

    def do_details(self, arg):
        """Display detailed information about the given genotype ID."""
        genotype_id, genotype = self._parse_genotype_id(arg)
        if genotype is None:
            return

        print(genotype)
        print("Metrics:")
        for metric_name in genotype.metrics:
            print(f"  {metric_name}: {genotype.metrics[metric_name]}")

    def do_parents(self, arg):
        """List the parent genotypes of the given genotype ID."""
        genotype_id, genotype = self._parse_genotype_id(arg)
        if genotype is None:
            return

        parents = genotype.parent_ids
        print(f"Created through {genotype.creation_method.lower()}.")
        if parents:
            print(f"Parents: {', '.join([str(parent) for parent in parents])}")

        for parent_id in parents:
            self._store_recent_genotype_id(parent_id)

    def do_population(self, arg):
        """Set the population context for commands."""
        population_name = arg
        if population_name in self.experiment_data['individuals']:
            self._set_population(population_name)
            print(f"Population set to '{population_name}'.")
        else:
            print(f"Population '{population_name}' not found. Available populations: {', '.join(self.experiment_data['individuals'])}")

    def complete_population(self, text, line, begidx, endidx):
        if text == "":
            completions = self.experiment_data['individuals']
        else:
            completions = [population for population in self.experiment_data['individuals'] if population.startswith(text)]
        return completions

    def do_top(self, arg):
        """Display the top genotype from the given generation in the current population."""
        generation = self._parse_generation(arg)
        if generation is None:
            return

        generation_data: GenerationData = self.experiment_data['generations'][self.population][generation]
        individual_ids = generation_data['individual_ids']

        top_id = individual_ids[0]
        print(f"Displaying ID {top_id}.")
        self._store_recent_genotype_id(top_id)

        genotype_id, genotype = self._parse_genotype_id(top_id)
        if genotype is None:
            return

        print(genotype)

    def do_generation(self, arg):
        """Display all genotypes from the given generation in the current population."""
        generation = self._parse_generation(arg)
        if generation is None:
            return

        generation_data: GenerationData = self.experiment_data['generations'][self.population][generation]
        individual_ids = generation_data['individual_ids']

        loaded_ids = []
        print(f"Displaying IDs from {self.population} generation {generation}:")
        for individual_id in individual_ids:
            genotype_id, genotype = self._parse_genotype_id(individual_id)
            if genotype_id is not None:
                print(f"{genotype_id} - {self._str_objectives(genotype)}")
                loaded_ids.append(genotype_id)
            else:
                print(f"{individual_id} - genotype not found")
        self.recent_genotypes = loaded_ids  # Allow exceeding the length limit. Top-rated genotypes are prioritized.

    def complete_generation(self, text, line, begidx, endidx):
        if self.population is None:
            return []

        generations = sorted(self.experiment_data['generations'][self.population].keys())
        generation_strings = [str(generation) for generation in generations]
        if text == "":
            completions = generation_strings
        else:
            completions = [generation for generation in generation_strings if generation.startswith(text)]
        return completions

    def do_gen(self, arg):
        """Shortcut for 'generation'."""
        self.do_generation(arg)

    def complete_gen(self, *args):
        self.complete_generation(*args)

    def do_ancestry(self, arg):
        """Display the ancestry tree for the given genotype ID."""
        genotype_id, genotype = self._parse_genotype_id(arg)
        if genotype is None:
            return

        parents = {}
        depth_limited = set()

        def collect_parents(genotype_id: int, depth: int, depth_limit: int):
            current = self._load_genotype(genotype_id)
            if current is None:
                return
            if depth < depth_limit:
                parents[genotype_id] = current.parent_ids
                for parent_id in current.parent_ids:
                    if parent_id not in parents:
                        collect_parents(parent_id, depth + 1, depth_limit)
            elif len(current.parent_ids) > 0:
                depth_limited.add(genotype_id)

        collect_parents(genotype_id, 0, 16)

        displayed = set()
        def display_ancestry(genotype_id: int, depth: int, last_child: bool, pipe_depths: list[int]):
            """
            Recursively display the ancestry tree at the current genotype ID.
            Args:
                genotype_id: The genotype ID for this line.
                depth: The depth of this genotype in the tree.
                last_child: Whether this genotype is the last child (tree-wise) of its parent, for drawing.
                pipe_depths: Which depths should have a vertical pipe drawn because they are in between children.
            """
            line = ""
            if depth > 0:
                for space_depth in range(depth - 1):
                    if space_depth in pipe_depths:
                        line += "│ "
                    else:
                        line += "  "
                if last_child:
                    line += "└─"
                else:
                    line += "├─"

            new_genotype = genotype_id not in displayed

            if genotype_id not in self.genotype_cache:
                # Genotype not loaded.
                line += f"{genotype_id} - genotype not found"
            else:
                genotype = self.genotype_cache[genotype_id]
                line += str(genotype_id)
                if genotype.creation_method == "Cloning":
                    line += " (survived)"
                if not new_genotype:
                    line += " (displayed above)"
                if genotype_id in depth_limited:
                    line += " (depth limit)"
                line += f" - {self._str_objectives(genotype, color=True)}"

            print(line)
            displayed.add(genotype_id)

            if new_genotype and genotype_id in parents:
                # Skips if repeated or depth-limited.
                parent_ids = parents[genotype_id]
                for index, parent_id in enumerate(parent_ids):
                    last_child = index == len(parent_ids) - 1
                    sub_pipe_depths = pipe_depths.copy()
                    if not last_child:
                        sub_pipe_depths.append(depth)
                    display_ancestry(parent_id, depth + 1, last_child, sub_pipe_depths)

        print(f"Ancestry for genotype {genotype_id}:")
        display_ancestry(genotype_id, 0, True, [])
        self.recent_genotypes = list(displayed)

    def do_exit(self, arg):
        """Exit the program."""
        print("Exiting.")
        return True

    def preloop(self):
        experiment_name = self.experiment.config['log_subfolder']
        experiment_type_name = type(self.experiment).__name__
        self.intro =\
        f"""Genotype browser initialized for experiment '{experiment_name}' ({experiment_type_name}).
        "Type 'help' for a list of commands."""
        self.population = list(self.experiment_data['individuals'])[0]
        self.prompt = f"({self.population}) "


def main():
    # Request the path to the experiment logs.
    path = commandlineutils.prompt_experiment_path()

    # Request a run to load.
    run_path = commandlineutils.prompt_run(path)

    experiment = postprocessingutils.load_experiment_definition(path)

    print(f"Loading data from {run_path}...")
    experiment_data = postprocessingutils.load_run_data(run_path)

    browser = GenotypeBrowser(experiment, experiment_data)
    browser.cmdloop()

if __name__ == "__main__":
    main()