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

import logging
import sys


def initialize_logger(log_path: str = None, console_output: bool = True, debug: bool = False) -> None:
    """Initialize the root `logging.Logger` for this experiment.

    Args:
        log_path: If provided, log messages will be saved to this file.
        console_output: If true, log messages will be printed to the console.
        debug: If true, log messages will include debug messages.
    """
    logger = logging.getLogger()

    # This might break something if handlers are being added outside of the main loop.
    if logger.hasHandlers():
        for handler in logger.handlers.copy():
            logger.removeHandler(handler)

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)

    if console_output:
        console_formatter = logging.Formatter('%(message)s')

        console_stderr_handler = logging.StreamHandler(sys.stderr)
        console_stderr_handler.setLevel(logging.WARNING)
        console_stderr_handler.setFormatter(console_formatter)
        logger.addHandler(console_stderr_handler)

        console_stdout_handler = logging.StreamHandler(sys.stdout)
        console_stdout_handler.setLevel(level)
        console_stdout_handler.setFormatter(console_formatter)
        console_stderr_handler.addFilter(lambda record: record.levelno <= logging.INFO)
        logger.addHandler(console_stdout_handler)

    if log_path is not None:
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    def except_hook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = except_hook
    logging.captureWarnings(True)