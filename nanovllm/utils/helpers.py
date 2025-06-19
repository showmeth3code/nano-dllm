from __future__ import annotations


def clamp(value: int, min_value: int, max_value: int) -> int:
    """Clamp ``value`` to the inclusive range [min_value, max_value]."""
    return max(min(value, max_value), min_value)


def flatten(list_of_lists: list[list[int]]) -> list[int]:
    """Flatten a list of lists into a single list."""
    return [item for sublist in list_of_lists for item in sublist]


def chunked(lst: list[int], size: int) -> list[list[int]]:
    """Split a list into chunks of the given size."""
    if size <= 0:
        return []
    return [lst[i : i + size] for i in range(0, len(lst), size)]

