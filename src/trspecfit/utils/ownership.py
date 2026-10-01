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
"""

from __future__ import annotations

from typing import Never, NoReturn, overload

import numpy as np


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
