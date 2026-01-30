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
import warnings
from typing import Any, Tuple


def recursive_size(object: Any, seen_ids: set = None) -> Tuple[int, dict | None]:
    """Recursively calculates the size of an object in bytes, including all of its attributes."""

    if seen_ids is None:
        seen_ids = set()

    object_id = id(object)
    if object_id in seen_ids:
        return 0, None
    seen_ids.add(object_id)

    object_map = {}

    size = sys.getsizeof(object)
    items = None

    if isinstance(object, (list, tuple, set, frozenset)):
        items = {index: item for index, item in enumerate(object)}
    elif isinstance(object, dict):
        items = {f'key_{index}': key for index, key in enumerate(object.keys())}
        items.update({str(key): value for key, value in object.items()})
    elif _has_custom_sizeof(object):
        # Assume that a custom __sizeof__ is accurate (e.g. Numpy arrays)
        pass
    elif hasattr(object, '__slots__') and object.__slots__ is not None:
        items = {slot: getattr(object, slot) for slot in object.__slots__ if hasattr(object, slot)}
    elif hasattr(object, '__dict__'):
        items = object.__dict__
    
    if items is None:
        return size, None
    
    for key, item in items.items():
        item_size, item_map = recursive_size(item, seen_ids)
        size += item_size
        object_map[key] = (item_size, item_map)
    return size, object_map


def _has_custom_sizeof(test_object: Any) -> bool:
    """
    Tests whether an object has a custom `__sizeof__` method.

    Args:
        test_object: The object to test.
    Returns:
        False if the object uses the default `__sizeof__` implementation from `object`, True otherwise.
    """
    if not hasattr(type(test_object), 'mro'):
        return False
    if isinstance(test_object, type):
        return False
    for cls in type(test_object).mro():
        if cls is object:
            pass
        elif '__sizeof__' in cls.__dict__:
            return True
    return False


try:
    import psutil
except ImportError:
    psutil = None
def log_process_memory(logger: logging.Logger, label: str):
    if psutil is None:
        warnings.warn("psutil is not installed. Skipping log_process_memory calls.")
        return
    logger.debug(f"{label} - Current memory usage: {psutil.Process().memory_info().rss:,} B")