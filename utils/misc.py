from collections.abc import Collection
from typing import TypeVar

T = TypeVar('T')

def map_to_index(iter: Collection[T]) -> dict[T, int]:
    return dict(zip(iter, range(len(iter))))
