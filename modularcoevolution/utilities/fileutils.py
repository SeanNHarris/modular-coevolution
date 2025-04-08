import os
import re
import warnings
from pathlib import Path


def get_nearest_path(directory: str, strict: bool = True) -> Path:
    """
    Get the path to a directory, searching in the current working directory and its parents.
    Args:
        directory: The name of the directory to search for.
        strict: If True, only searches in the current working directory.
            Use this when writing, to prevent accidentally writing to the wrong directory.

    Returns:
        The path to the directory.
    """
    cwd = Path(os.getcwd())
    paths = [cwd]
    if not strict:
        paths.extend(cwd.parents)
    for path in paths:
        if os.access(path / directory, os.W_OK):
            return path / directory
    if strict:
        raise FileNotFoundError(f"Could not find the {directory} directory from the cwd (strict mode).")
    else:
        raise FileNotFoundError(f"Could not find the {directory} directory from the cwd or its parents.")


def get_logs_path(strict: bool = True, can_create: bool = False) -> Path:
    """
    Get the path to the logs directory.
    Searches for a directory named 'logs' in the current working directory or its parents, returning the first found.

    Args:
        strict: If True, only searches in the current working directory.
            Use this when writing logs, to prevent accidentally writing to the wrong directory.
        can_create: If True, creates the logs directory next to the config directory if it doesn't exist.

    Returns:
        The path to the logs directory.
    """
    try:
        return get_nearest_path("logs", strict)
    except FileNotFoundError as error:
        if can_create:
            warnings.warn(
                "Could not find the logs directory. Creating a new one adjacent to the config directory."
            )
            config_path = get_nearest_path("config", strict)
            logs_path = config_path.parent / "logs"
            logs_path.mkdir()
            return logs_path
        else:
            raise error


def get_config_path() -> Path:
    """
    Get the path to the config directory.
    Searches for a directory named 'config' in the current working directory or its parents, returning the first found.

    Returns:
        The path to the config directory.
    """
    return get_nearest_path("config", strict=False)


def resolve_experiment_path(experiment_path_str: str) -> Path:
    """
    Resolve a path to an experiment, checking in multiple locations with the following priority:
    1. The unmodified path.
    2. The path relative to the logs directory.

    Args:
        experiment_path_str:
            The path to the experiment folder, either absolute, local, or relative to the logs directory.

    Returns:
        A confirmed path to the experiment folder.
    """
    path = Path(experiment_path_str)
    if '~' in str(path):
        path = path.expanduser()  # ~ is not resolved by default.
    if path.exists():
        return path

    logs_path = Path(get_logs_path(strict=False))  # Non-strict, because we're reading.
    path = logs_path / experiment_path_str
    if path.exists():
        return path

    raise FileNotFoundError(f"Could not find the experiment folder at {experiment_path_str}")


def resolve_config_path(config_path_str: str) -> Path:
    """
    Resolve a path to a config file, checking in multiple locations with the following priority:
    1. The unmodified path.
    2. The path relative to the config directory.

    Args:
        config_path_str:
            The path to the config file, either absolute, local, or relative to the config directory.

    Returns:
        A confirmed path to the config file.
    """
    path = Path(config_path_str)
    if path.exists():
        return path

    logs_path = Path(get_config_path())
    path = logs_path / config_path_str
    if path.exists():
        return path

    raise FileNotFoundError(f"Could not find the config file at {config_path_str}")


def get_run_paths(experiment_path_str: str) -> list[str]:
    """
    Get a list of run folders within a given experiment folder.
    Only includes folders of the form `Run #`, sorted by run number.

    Args:
        experiment_path_str: The path to the experiment folder within the logs directory.

    Returns:
        A list of paths to the run folders within `experiment_folder`.
    """
    experiment_path = resolve_experiment_path(experiment_path_str)
    run_folders = [folder.path for folder in os.scandir(experiment_path) if folder.is_dir()]
    # Get folders of the form 'Run #', sorted by their number
    run_folders = [folder for folder in run_folders if re.match(r'.*Run \d+', folder)]
    run_folders.sort(key=lambda folder: int(folder.split(' ')[-1]))
    return run_folders
