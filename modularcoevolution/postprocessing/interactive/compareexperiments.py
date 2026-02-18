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

"""
Compares the results of multiple experiments with compatible agents.
Use case: comparing the impact of different EA parameters against a common problem environment.
"""

import json
from pathlib import Path

from modularcoevolution.postprocessing import postprocessingutils
from modularcoevolution.utilities import commandlineutils


def main():
    # Request the number of experiments to compare.
    experiment_count = commandlineutils.prompt_int("How many experiments to compare?", minimum=2)

    # Request the number of representatives to use from each run.
    representative_count = commandlineutils.prompt_int("How many representatives from each run?", minimum=1)

    # Request the number of repeat evaluations per pairing.
    repeat_count = commandlineutils.prompt_int("How many evaluations for each pairing?", minimum=1, default=1)

    # Request the experiment log paths.
    experiment_paths = []
    experiment_archives = []
    for index in range(experiment_count):
        path = commandlineutils.prompt_experiment_path(f"Input the path to experiment (treatment) {index + 1} within the logs folder:")
        experiment_paths.append(path)

        experiment, _, archives = postprocessingutils.easy_load_experiment_results(
            path,
            representative_size=representative_count,
            last_generation=True,
            strip_dictionaries=True,
            parallel=True,
        )
        experiment_archives.append(archives)

    # Compare the experiments.
    results = postprocessingutils.compare_experiments(
        experiment_archives,
        experiment,
        repeat_evaluations=repeat_count,
        parallel=True
    )

    results_per_population = {}
    for population_name in results[0][0]:
        results_per_population[population_name] = {
            experiment_path: [run_results[population_name]['fitness']['mean'] for run_results in experiment_results]
            for experiment_path, experiment_results in zip(experiment_paths, results)
        }

    print("Comparison Results:")
    print(results_per_population)

    # Save the results.
    save_filename = input("Input a filename to save the results, or press Enter to skip saving:\n")
    if save_filename == "":
        save_filename = None

    if save_filename is not None:
        save_paths = set()
        for experiment_path in experiment_paths:
            save_path = Path(experiment_path).parent / f"{save_filename}.json"
            if save_path in save_paths:
                continue
            save_paths.add(save_path)

            with open(save_path, 'w+') as save_file:
                json.dump(results, save_file)
            print(f"Results saved to {save_path}.")


if __name__ == '__main__':
    main()
