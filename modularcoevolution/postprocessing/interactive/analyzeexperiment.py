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

from modularcoevolution.generators.archivegenerator import ArchiveGenerator

"""
Analyzes the final population from multiple runs of a single experiment,
getting information about performance per agent, run, and population.
"""

import json

from modularcoevolution.postprocessing import postprocessingutils
from modularcoevolution.utilities import commandlineutils


def main():
    # Request the number of representatives to use from each run.
    representative_count = commandlineutils.prompt_int("How many representatives from each run?", minimum=1)

    # Request the number of repeat evaluations per pairing.
    repeat_count = commandlineutils.prompt_int("How many evaluations for each pairing?", minimum=1, default=1)

    # Request the experiment log path.
    experiment_path = commandlineutils.prompt_experiment_path()
    experiment, _, archives = postprocessingutils.easy_load_experiment_results(
        experiment_path,
        representative_size=representative_count,
        last_generation=True,
        strip_dictionaries=True,
        parallel=True,
    )

    # For each run name, a dictionary mapping population names to archive generators containing the loaded individuals.
    archives: dict[str, dict[str, ArchiveGenerator]]

    population_archives = {}
    for run_name, run_archives in archives.items():
        for population_name, population_archive in run_archives.items():
            if population_name not in population_archives:
                population_archives[population_name] = []
            population_archives[population_name].append(population_archive)

    merged_archives = []
    for population_name, archive_list in population_archives.items():
        merged_archives.append(ArchiveGenerator.merge_archives(archive_list))

    results = postprocessingutils.round_robin_evaluation(
        merged_archives,
        experiment,
        repeat_evaluations=repeat_count,
        parallel=True,
    )

    population_metrics = {population_name: archive.aggregate_metrics() for population_name, archive in zip(population_archives.keys(), merged_archives)}
    print("Aggregate metrics across all runs:")
    print(json.dumps(population_metrics, indent=4))

    run_metrics = {
        run_name: {population_name: archive.aggregate_metrics() for population_name, archive in run_archives.items()}
        for run_name, run_archives in archives.items()
    }

    for run_name, run_metrics in run_metrics.items():
        short_run_name = run_name.split("/")[-1].split("\\")[-1]
        print(f"{short_run_name} metrics:")
        print(json.dumps(run_metrics, indent=4))
        # for population_name, population_metrics in run_metrics.items():
        #    print(f"\t{population_name}")
        #    for metric_name, metric_statistics in population_metrics.items():
        #        print(f"\t\t{metric_name}: {metric_statistics['mean']}")

    return experiment, archives, merged_archives, results


if __name__ == '__main__':
    main()
