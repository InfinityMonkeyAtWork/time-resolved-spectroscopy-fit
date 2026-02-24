"""Typed HDF5 helper utilities used across trspecfit."""

from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np


#
def json_loads_attr(value: Any) -> Any:
    """Parse JSON stored in an HDF5 attribute with basic type normalization."""

    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise TypeError("Expected scalar JSON attribute, got array.")
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise TypeError(
            f"Expected JSON attribute as str/bytes, got {type(value).__name__}"
        )

    return json.loads(value)

#
def require_group(node: Any, path: str) -> h5py.Group:
    """Ensure an HDF5 node is a Group."""

    if not isinstance(node, h5py.Group):
        raise TypeError(f"Expected HDF5 group at '{path}', got {type(node).__name__}")
    return node

#
def require_dataset(node: Any, path: str) -> h5py.Dataset:
    """Ensure an HDF5 node is a Dataset."""

    if not isinstance(node, h5py.Dataset):
        raise TypeError(f"Expected HDF5 dataset at '{path}', got {type(node).__name__}")
    return node
