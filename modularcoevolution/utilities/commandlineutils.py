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

import contextlib
import functools
import os
import warnings
from pathlib import Path

from modularcoevolution.utilities import fileutils

try:
    import readline  # readline doesn't seem to work on Windows?
    readline_default_completer = readline.get_completer()
    readline_default_completer_delims = readline.get_completer_delims()
except ImportError:
    readline = None


def unset_completer() -> None:
    """Unset the readline completer, if it is set."""
    if readline is not None:
        readline.set_completer(readline_default_completer)
        readline.set_completer_delims(readline_default_completer_delims)


def managed_completer(set_completer_function):
    """Decorator to manage setting and unsetting a readline completer."""
    @functools.wraps(set_completer_function)
    @contextlib.contextmanager
    def wrapper(*args, **kwargs):
        set_completer_function(*args, **kwargs)
        try:
            yield  # See https://docs.python.org/3/library/contextlib.html#contextlib.contextmanager
        finally:
            unset_completer()
    return wrapper


@managed_completer
def set_path_completer(base_path: Path) -> None:
    """
    Configure tab completion for relative paths, using the `readline` library.
    Does not work on Windows.

    Args:
        base_path: A base path to use for relative path completion.
    """

    if readline is None:
        warnings.warn("Note: tab completion is not available on this platform (expected for Windows).")
        return

    def completer(text: str, state: int):
        # print(f"Text: {text}, State: {state}")
        head, tail = os.path.split(text)
        # print(f"Head: {head}, Tail: {tail}")
        current_path = base_path / head
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


def prompt_experiment_path(prompt: str = None) -> str:
    """
    Prompt the user for an experiment log path.

    Args:
        prompt: An optional message to display before the prompt.

    Returns:
        A string path to a valid experiment log folder.
    """
    if prompt is None:
        prompt = "Input the experiment folder path within the logs folder:"

    try:
        logs_path = fileutils.get_logs_path()
    except FileNotFoundError as error:
        print("Could not find the logs directory. Ensure that you are executing this script from the project root.")
        raise error

    with set_path_completer(logs_path):
        experiment_path = None
        while experiment_path is None:
            experiment_path = input(prompt + '\n')
            try:
                run_folders = fileutils.get_run_paths(experiment_path)
            except FileNotFoundError:
                print("Experiment folder not found. Ensure that you are executing this script from the project root.")
                experiment_path = None

    unset_completer()

    return experiment_path


def prompt_run(experiment_path: str) -> str:
    """
    Prompt the user for a valid run number for the given experiment.

    Args:
        experiment_path: The path to the experiment log folder.

    Returns:
        The path to the selected run's log folder.
    """
    run_folders = fileutils.get_run_paths(experiment_path)

    run_numbers = list(run_folders.keys())
    first_run = run_numbers[0]
    last_run = run_numbers[-1]
    print(f"Available runs: {first_run} - {last_run}")
    run_number = -1
    while run_number == -1:
        run_number = input("Input the run number to read from: ")
        try:
            run_number = int(run_number)
        except ValueError:
            print("Invalid run number.")
            run_number = -1

    return run_folders[run_number]


def prompt_int(prompt: str = None, minimum: int = None, default: int = None) -> int:
    """
    Prompt the user for an integer input.

    Args:
        prompt: An optional custom message to display before the prompt.
        minimum: An optional minimum acceptable value.
        default: An optional default value to use if the user inputs an empty string.

    Returns:
        An integer given by the user.
    """
    if prompt is None:
        prompt = "Input an integer:"

    if default is not None:
        prompt += f" (default: {default})"

    while True:
        user_input = input(prompt + '\n')
        if user_input == '' and default is not None:
            return default

        try:
            result = int(user_input)
        except ValueError:
            print("Invalid input. Please enter an integer.")
            continue

        if minimum is not None and result < minimum:
            print(f"Value must be at least {minimum}.")
            continue
        return result


def prompt_options(prompt: str = None, options: list[str] = None, default: str = None, ) -> str:
    """
    Prompt the user to select from a list of options. Defaults to a "y/n" prompt.

    Args:
        prompt: An optional custom message to display before the prompt.
        options: A list of valid options for the user to select from. Defaults to ['y', 'n'].
        default: An optional default value to use if the user inputs an empty string.

    Returns:
        A string from the options list, given by the user.
    """
    if options is None:
        options = ['y', 'n']
    else:
        options = [option.lower() for option in options]

    options_copy = options.copy()
    if default is not None:
        if default not in options_copy:
            raise ValueError("Default value must be in options.")
        default_index = options_copy.index(default)
        options_copy[default_index] = options_copy[default_index].upper()
    option_string = f"({'/'.join(options_copy)})"

    if prompt is None:
        prompt = f"Input one of the following options: {option_string}"
    else:
        prompt += f" {option_string}"

    while True:
        user_input = input(prompt + '\n')
        if user_input == '' and default is not None:
            return default

        if user_input.lower() not in options:
            print(f"Invalid input. Please enter one of the following options: {option_string}")
            continue
        return user_input.lower()


def color_string_24_bit(string: str, red: int, green: int, blue: int, background: bool = False) -> str:
    """
    Color the given string using 24-bit ANSI escape codes.
    May not work in all terminals, but should work in Linux and modern Windows.

    Args:
        string: The input text.
        red: The red byte of the color (0-255).
        green: The green byte of the color (0-255).
        blue: The blue byte of the color (0-255).
        background: If True, color the background.

    Returns:
        The input string wrapped in ANSI escape codes to apply a color.
    """
    if background:
        background_code = 48
    else:
        background_code = 38
    return f"\x1b[{background_code};2;{red};{green};{blue}m{string}\x1b[0m"


def color_string_256(string: str, red: int, green: int, blue: int, background: bool = False) -> str:
    """
    Color the given string using 256-color ANSI escape codes.
    Specifically uses the 6x6x6 cube, which is defined by:
    `code = 16 + 36 × r + 6 × g + b (0 ≤ r, g, b ≤ 5)`
    (Reference: https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797)

    Args:
        string: The input text.
        red: The red value of the color (0-5).
        green: The green value of the color (0-5).
        blue: The blue value of the color (0-5).
        background: If True, color the background.

    Returns:
        The input string wrapped in ANSI escape codes to apply a color.
    """
    color_code = 16 + 36 * red + 6 * green + blue
    if background:
        background_code = 48
    else:
        background_code = 38
    return f"\x1b[{background_code};5;{color_code}m{string}\x1b[0m"


def rgb_to_24_bit(red: float, green: float, blue: float) -> tuple[int, int, int]:
    """Convert (0-1) RGB values to 24-bit color values (0-255)."""
    color = (int(255 * red), int(255 * green), int(255 * blue))
    return color

def rgb_to_256(red: float, green: float, blue: float) -> tuple[int, int, int]:
    """Convert (0-1) RGB values to 256-color values (0-5)."""
    color = (int(5 * red), int(5 * green), int(5 * blue))
    return color