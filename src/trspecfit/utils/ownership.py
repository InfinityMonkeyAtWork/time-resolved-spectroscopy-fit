"""
Ownership mechanisms behind ``docs/design/api_ownership_contract.md``.

An ownership rule is declared on the attribute it governs, where the
attribute is defined, instead of being spread over setters and accessor
methods. This module holds the three mechanisms:

- :func:`freeze` / :func:`frozen_copy` — an array the package hands out is
  read-only, so an in-place edit raises numpy's own error. The flag is
  advisory and reversible; the copy is the boundary.
- :class:`owned` — a read-only attribute whose value the owning object
  writes through the private name ``_<name>``. Assignment from outside
  raises and names the supported route.
- :class:`detached` — a dataclass field whose container (a DataFrame, a
  dict, a list) is copied on set and on every read, so a record is a
  snapshot and editing what a read returned never reaches it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Never, NoReturn, cast, overload

import numpy as np
import pandas as pd

_MISSING: Any = object()


#
def freeze(arr: np.ndarray) -> np.ndarray:
    """Clear the write flag of *arr* in place and return it."""

    arr.flags.writeable = False
    return arr


#
def frozen_copy(arr: np.ndarray) -> np.ndarray:
    """Copy *arr* with the write flag cleared — the snapshot ownership boundary."""

    return freeze(np.array(arr, copy=True))


#
#
class owned[T]:
    """
    Read-only attribute of an owning object.

    Declared on the class as ``name = owned[T]("<route>")``. Reads return
    the object stored under ``instance._name``, which the owner writes
    directly; assigning ``instance.name`` raises ``AttributeError`` that
    names the route (``"call define_baseline()"``, ``"construct a new
    File"``). The stored object should itself be immutable (a frozen array,
    a tuple, a frozen dataclass) so that a read cannot be edited in place
    either.
    """

    def __init__(self, route: str) -> None:
        self._route = route

    #
    def __set_name__(self, owner: type, name: str) -> None:
        self._owner = owner.__name__
        self._name = name
        self._private = f"_{name}"

    #
    @overload
    def __get__(self, instance: None, owner: type | None = None) -> owned[T]: ...

    @overload
    def __get__(self, instance: object, owner: type | None = None) -> T: ...

    def __get__(self, instance: object | None, owner: type | None = None):
        if instance is None:
            return self
        try:
            return instance.__dict__[self._private]
        except KeyError:
            raise AttributeError(
                f"{self._owner}.{self._name} has not been set by its owner"
            ) from None

    #
    def __set__(self, instance: object, value: Never) -> NoReturn:
        raise AttributeError(
            f"{self._owner}.{self._name} is owned by the {self._owner.lower()} "
            f"and cannot be assigned; {self._route}."
        )


#
def detach(value: Any) -> Any:
    """
    A detached copy of a container; anything else is returned as is.

    Mappings (as plain dicts), lists and tuples are rebuilt member by
    member, DataFrames copied, and arrays inside them come back as frozen
    copies, so a container a record hands out holds read-only arrays like
    every other array the package hands out. Scalars and strings pass
    through.
    """

    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: detach(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detach(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach(item) for item in value)
    if isinstance(value, np.ndarray):
        return frozen_copy(value)
    return value


#
#
class detached[T]:
    """
    Dataclass field holding a container that no reader can edit in place.

    Declared on a frozen dataclass as ``name: detached[T] = detached()``
    (``detached(default=None)`` for an optional field). The generated
    ``__init__`` stores through ``__set__``, which keeps a detached copy
    (closing any alias the builder passed in); every read returns another
    detached copy, so ``record.name`` is the caller's own object.
    ``dataclasses.replace``, keyword construction and pickling work as for
    a plain field; a read of the class attribute returns the default, or
    raises ``AttributeError`` when there is none, which is how dataclasses
    tell a required field from an optional one.
    """

    def __init__(self, *, default: T = _MISSING) -> None:
        self._default = default

    #
    def __set_name__(self, owner: type, name: str) -> None:
        self._owner = owner.__name__
        self._name = name
        self._private = f"_{name}"

    #
    @overload
    def __get__(self, instance: None, owner: type | None = None) -> T: ...

    @overload
    def __get__(self, instance: object, owner: type | None = None) -> T: ...

    def __get__(self, instance: object | None, owner: type | None = None) -> T:
        if instance is None:
            if self._default is _MISSING:
                raise AttributeError(f"{self._owner}.{self._name} has no default")
            return self._default
        try:
            stored = instance.__dict__[self._private]
        except KeyError:
            raise AttributeError(
                f"{self._owner}.{self._name} has not been set"
            ) from None
        return cast("T", detach(stored))

    #
    def __set__(self, instance: object, value: T) -> None:
        instance.__dict__[self._private] = detach(value)
