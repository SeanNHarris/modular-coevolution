import ast
import csv
import os.path
from typing import Any

import matplotlib.axes
from matplotlib import pyplot

import modularcoevolution.utilities.fileutils
from modularcoevolution.postprocessing import postprocessingutils
from modularcoevolution.utilities import fileutils
from modularcoevolution.utilities.datacollector import DataSchema

try:
    import readline  # readline doesn't seem to work on Windows?
except ImportError:
    readline = None

try:
    import tkinter
except ImportError:
    tkinter = None


def load_generational_data(experiment_path: str, run_number: int = None):
    if run_number is None:
        run_numbers = None
    else:
        run_numbers = [run_number]

    return postprocessingutils.load_experiment_data(
        experiment_path,
        last_generation=True,  # This looks wrong, but the full generational data is included in the last generation's data file.
        load_only=['generations'],
        parallel=False,
        run_numbers=run_numbers
    )


def get_plottable_keys(data: DataSchema, partial_key: str = None):
    if partial_key is None:
        partial_key_parts = []
    else:
        partial_key_parts = partial_key.split('.')
        if len(partial_key_parts) > 3:
            return []  # Key is too long.

    plottable_keys = []

    if len(partial_key_parts) > 0:
        population_name = partial_key_parts.pop(0)
        if population_name not in data['generations']:
            return []  # Population does not exist.
        population_names = [population_name]
    else:
        population_names = list(data['generations'].keys())
    for population_name in population_names:
        population_data = data['generations'][population_name]
        last_generation = list(population_data.keys())[-1]
        generation_data = population_data[last_generation]
        individual_metric_data = generation_data['metric_statistics']
        population_metric_data = generation_data['population_metrics']

        if len(partial_key_parts) > 0:
            metric_name = partial_key_parts.pop(0)
            individual_metric_names = [metric_name]
            population_metric_names = [metric_name]
        else:
            individual_metric_names = individual_metric_data.keys()
            population_metric_names = population_metric_data.keys()

        for individual_metric_name in individual_metric_data.keys():
            if individual_metric_name not in individual_metric_names:
                continue

            metric_data = individual_metric_data[individual_metric_name]
            if len(partial_key_parts) > 0:
                statistic_name = partial_key_parts.pop(0)
                statistic_names = [statistic_name]
            else:
                statistic_names = metric_data.keys()
            for statistic_name in statistic_names:
                if statistic_name not in metric_data:
                    continue
                plottable_keys.append(".".join((population_name, individual_metric_name, statistic_name)))

        for population_metric_name in population_metric_data.keys():
            if len(partial_key_parts) > 0:
                continue
            if population_metric_name not in population_metric_names:
                continue
            plottable_keys.append(".".join((population_name, population_metric_name)))

    return plottable_keys


def access_key(data: DataSchema, generation: int, key: str):
    key_parts = key.split('.')
    if len(key_parts) < 2 or len(key_parts) > 3:
        raise ValueError("Key must have either two parts (population metric) or three parts (individual metric). For example, 'attacker.fitness.mean'.")
    population_name = key_parts[0]
    metric_name = key_parts[1]
    try:
        if len(key_parts) == 2:
            return data['generations'][population_name][str(generation)]['population_metrics'][metric_name]
        else:
            statistic_name = key_parts[2]
            return data['generations'][population_name][str(generation)]['metric_statistics'][metric_name][statistic_name]
    except KeyError:
        raise KeyError(f"Key '{key}' not found in data.")


def get_key_overlap(keys: list[str]) -> tuple[str, list[str]]:
    key_matrix = [key.split('.') for key in keys]
    overlap_count = 0
    while True:
        if any(len(key_parts) <= overlap_count for key_parts in key_matrix):
            break
        if len(set(key_parts[overlap_count] for key_parts in key_matrix)) == 1:
            overlap_count += 1
        else:
            break
    overlap = '.'.join(key_matrix[0][:overlap_count])
    remainders = ['.'.join(key_parts[overlap_count:]) for key_parts in key_matrix]
    return overlap, remainders


def format_key(key: str, key_aliases: dict[str, str] = None) -> str:
    if key_aliases is not None:
        for string, replacement in key_aliases.items():
            key = key.replace(string, replacement)
    key = key.replace('_', ' ')
    key = key.replace('.', ' - ')
    key = key.replace('\\_', '_')
    key = key.replace('\\.', '.')
    key = key.title()
    return key


def plot_generational(
        experiment_path: str,
        data_keys: str | list[str] | list[list[str]] = None,
        run_number: int = None,
        title: str = None,
        subplot_titles: list[str] = None,
        subplot_ylabels: list[str] = None,
        key_aliases: dict[str, str] = None,
        subplots_kwargs: dict[str, Any] = None,
        show_plot: bool = True,
        save_filename: str = None,
):
    """
    Automatically plot one or more metrics over time from the provided experiment data.

    Args:
        experiment_path: The path within the local logs subfolder for the experiment data.
        data_keys: Which keys to plot. If omitted, a list of available keys will be printed.
            - Providing a single key plots that key.
            - Providing a list of keys plots all of those keys on a single axes.
            - Providing a list of lists of keys produces a subplot for each of the inner lists,
            plotting the keys in that list.
        run_number: Which run within the experiment folder to plot. If omitted, all runs will be plotted together.
        title: The suptitle for the plot.
        subplot_titles: A list of titles for the subplots. If omitted, these will be generated from the keys.
        subplot_ylabels: A list of labels for the y-axis of each subplot. If omitted, "Value" will be used.
        key_aliases: A dictionary of string replacements to apply for formatting key names on the plots.
            `.` and `-` are special characters for keys and need to be escaped in the values.
        subplots_kwargs: Additional kwargs to pass to `pyplot.subplots`.
        show_plot: If true, will call `pyplot.show()` after generating the plot.
        save_filename: If provided, will save the plot and export the data to the specified filename.

    Returns: A tuple containing the figure and list of axes.
    """
    experiment_data = load_generational_data(experiment_path, run_number)
    first_data = list(experiment_data.values())[0]

    if data_keys is None:
        print("No data keys provided.")
        print("Available keys in this log:")
        print("\n".join(get_plottable_keys(first_data)))
        return
    elif isinstance(data_keys, str):
        data_keys = [[data_keys]]
    elif isinstance(data_keys[0], str):
        data_keys = [data_keys]

    if subplot_titles is None:
        subplot_titles = [None] * len(data_keys)
    if subplot_ylabels is None:
        subplot_ylabels = [None] * len(data_keys)

    subplots_default_kwargs = {
        "ncols": 1,
        "figsize": (10, 5 * len(data_keys)),
    }
    if subplots_kwargs is not None:
        subplots_default_kwargs.update(subplots_kwargs)

    figure, axs = pyplot.subplots(len(data_keys), **subplots_default_kwargs)
    if len(data_keys) == 1:
        axs = [axs]

    for key_index, key_list in enumerate(data_keys):
        plot_generational_subplot(
            experiment_data,
            key_list,
            axs[key_index],
            subplot_title=subplot_titles[key_index],
            subplot_ylabel=subplot_ylabels[key_index],
            key_aliases=key_aliases
        )

    if title is not None:
        figure.suptitle(title)
    figure.tight_layout()

    if save_filename is not None:
        long_save_filename = os.path.join("logs", experiment_path, save_filename)
        plot_filename = long_save_filename + ".png"
        print(f"Saving plot to {plot_filename}")
        figure.savefig(plot_filename)

        csv_filename = long_save_filename + ".csv"
        print(f"Exporting data to {csv_filename}")
        export_csv_generational(experiment_data, data_keys, csv_filename)

    if show_plot:
        pyplot.show()

    return figure, axs


def plot_generational_subplot(
        experiment_data: dict[str, DataSchema],
        keys: list[str],
        axes: matplotlib.axes.Axes,
        subplot_title: str = None,
        subplot_ylabel: str = None,
        key_aliases: dict[str, str] = None,
):
    key_overlap, key_remainders = get_key_overlap(keys)
    if subplot_title is None and len(key_overlap) > 0:
        subplot_title = format_key(key_overlap, key_aliases)
    if subplot_title is not None:
        axes.set_title(subplot_title)

    axes.set_xlabel("Generation")
    if subplot_ylabel is not None:
        axes.set_ylabel(subplot_ylabel)
    else:
        axes.set_ylabel("Value")

    for key_index, key in enumerate(keys):
        key_parts = key.split('.')
        population_name = key_parts[0]
        for run_name, run_data in experiment_data.items():
            generation_data = run_data['generations'][population_name]
            generations = [int(generation) for generation in generation_data.keys()]
            generation_values = [access_key(run_data, generation, key) for generation in generations]

            key_remainder = key_remainders[key_index]
            short_run_name = run_name.split('\\')[-1]
            short_run_name = short_run_name.split('/')[-1]  # Handle both Windows and Linux-style paths

            if len(experiment_data) == 1:
                label = key_remainder
            elif len(keys) == 1:
                label = short_run_name
            else:
                label = f"{key_remainder} - {short_run_name}"
            formatted_label = format_key(label, key_aliases)
            axes.plot(generations, generation_values, label=formatted_label)

    if len(keys) > 1 or len(experiment_data) > 1:
        axes.legend()


def export_csv_generational(
        experiment_data: dict[str, DataSchema],
        keys: list[str] | list[list[str]],
        output_path: str,
):
    with open(output_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        if isinstance(keys[0], list):
            keys = [key for sub_list in keys for key in sub_list]

        for key in keys:
            key_parts = key.split('.')
            population_name = key_parts[0]
            for run_name, run_data in experiment_data.items():
                generation_data = run_data['generations'][population_name]
                generations = [int(generation) for generation in generation_data.keys()]
                generation_values = [access_key(run_data, generation, key) for generation in generations]

                short_run_name = run_name.split('\\')[-1]
                short_run_name = short_run_name.split('/')[-1]

                csvwriter.writerow([key, short_run_name] + generation_values)


def interactive_plot_generational():
    if tkinter is not None:
        matplotlib.use('TkAgg')

    try:
        logs_path = fileutils.get_logs_path()
    except FileNotFoundError:
        logs_path = None

    if readline is not None and logs_path is not None:
        # Auto-complete keys?
        def completer(text: str, state: int):
            # print(f"Text: {text}, State: {state}")
            head, tail = os.path.split(text)
            # print(f"Head: {head}, Tail: {tail}")
            current_path = logs_path / head
            try:
                options = os.listdir(current_path)
            except FileNotFoundError:
                options = []
            # print(f"Options: {options}")
            matches = [key for key in options if key.startswith(tail)]
            # print(f"Matches: {matches}")
            if state < len(matches):
                return os.path.join(head, matches[state]) + "/"
            else:
                return None

        readline.set_completer(completer)
        readline.set_completer_delims("")
        readline.parse_and_bind("tab: complete")

    experiment_path = None
    while experiment_path is None:
        experiment_path = input("Input the experiment folder path within the logs folder:\n")
        try:
            run_folders = modularcoevolution.utilities.fileutils.get_run_paths(experiment_path)
        except FileNotFoundError:
            print("Experiment folder not found. Ensure that you are executing this script from the project root.")
            experiment_path = None

    if readline is not None:
        readline.set_completer()

    first_run = os.path.basename(run_folders[0])
    last_run = os.path.basename(run_folders[-1])
    print(f"Available runs: {first_run} - {last_run}")
    run_number = -1
    while run_number == -1:
        run_number = input("Input the run number to plot, or leave blank to plot all runs: ")
        if run_number == '':
            run_number = None
        else:
            try:
                run_number = int(run_number)
            except ValueError:
                print("Invalid run number.")
                run_number = -1

    print("Getting list of plottable keys...")
    experiment_data = load_generational_data(experiment_path, 0)
    plottable_keys = get_plottable_keys(list(experiment_data.values())[0])
    print("Available keys:")
    print("\n".join(plottable_keys))

    if readline is not None:
        # Auto-complete keys?
        def completer(text: str, state: int):
            # quote_position = max(text.rfind('"'), text.rfind("'"))
            # text_past_quote = text[(quote_position + 1):]  # If there is no quote, quote_position is -1, so this works
            matches = [key for key in plottable_keys if key.startswith(text)]
            if state < len(matches):
                return matches[state]
            else:
                return None

        readline.set_completer(completer)
        readline.set_completer_delims("\"'")
        readline.parse_and_bind("tab: complete")

    keys = None
    while keys is None:
        key_input = input("Input the keys to plot, or type 'help' for more information:\n")
        if key_input.lower() == 'help':
            print("Input a single key to plot that key. Example: 'attacker.fitness.mean'")
            print("Input a list of keys to plot all keys on a single axes. Example: '[attacker.fitness.mean, attacker.fitness.max]'")
            print("Input a list of lists of keys to produce a subplot for each inner list. Example: '[[attacker.fitness.mean, attacker.fitness.max], [defender.fitness.mean, defender.fitness.max]]'")
            continue
        try:
            keys = ast.literal_eval(key_input)
        except (ValueError, SyntaxError) as error:
            print(f"Invalid input: {error}")
            if len(key_input) >= 1 and key_input[0] != '[':
                print("Note: lists of keys need to be enclosed in [brackets].")
            if len(key_input) >= 3 and (key_input[0] not in ('"', "'") or key_input[1] not in ('"', "'") or key_input[2] not in ('"', "'")):
                print("Note: keys should be enclosed in quotes.")
            continue

        key_set = set()
        if isinstance(keys, str):
            key_set.add(keys)
        elif isinstance(keys, list) and all(isinstance(key, str) for key in keys):
            key_set.update(keys)
        elif isinstance(keys, list) and all(isinstance(key, list) for key in keys):
            for key_list in keys:
                key_set.update(key_list)
        else:
            print("Invalid input format.")
            keys = None
            continue

        for key in key_set:
            if key not in plottable_keys:
                print(f"Key '{key}' is invalid.")
                keys = None
                continue

    if readline is not None:
        readline.set_completer()

    skip = input("Use the default plot settings? (Y/n): ")
    if skip.lower() == "n":
        title = input("Input the plot title: ")
        subplot_titles = input("Input the subplot titles, separated by commas:\n").split(',')
        subplot_ylabels = input("Input the subplot y-axis labels, separated by commas:\n").split(',')

        key_aliases = None
        while key_aliases is None:
            key_aliases = input("Input key aliases as a dictionary string (or leave empty to skip):\n")
            if key_aliases != "":
                try:
                    key_aliases = ast.literal_eval(key_aliases)
                except (ValueError, SyntaxError) as error:
                    print(f"Invalid input: {error}")
                    key_aliases = None
            else:
                key_aliases = {}
        print("Note: write a custom script calling 'plot_generational' for more advanced options.")
    else:
        title = None
        subplot_titles = None
        subplot_ylabels = None
        key_aliases = None

    save_filename = input("Input a filename to save the plot and data, or press Enter to skip saving:\n")
    if save_filename == "":
        save_filename = None

    plot_generational(
        experiment_path,
        keys,
        run_number=run_number,
        title=title,
        subplot_titles=subplot_titles,
        subplot_ylabels=subplot_ylabels,
        key_aliases=key_aliases,
        show_plot=True,
        save_filename=save_filename
    )


if __name__ == "__main__":
    interactive_plot_generational()
