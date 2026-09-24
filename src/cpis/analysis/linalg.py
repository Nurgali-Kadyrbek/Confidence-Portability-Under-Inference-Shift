"""Small deterministic linear-algebra helpers for prespecified statistics."""

from __future__ import annotations


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(column) for column in zip(*matrix, strict=True)]


def matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    right_t = transpose(right)
    return [
        [sum(a * b for a, b in zip(row, column, strict=True)) for column in right_t]
        for row in left
    ]


def matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(a * b for a, b in zip(row, vector, strict=True)) for row in matrix]


def inverse(matrix: list[list[float]], tolerance: float = 1e-12) -> list[list[float]]:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("matrix inverse requires a nonempty square matrix")
    augmented = [
        [*map(float, row), *(1.0 if index == column else 0.0 for column in range(size))]
        for index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= tolerance:
            raise ValueError("singular design matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(
                        augmented[row], augmented[column], strict=True
                    )
                ]
    return [row[size:] for row in augmented]


def weighted_crossproduct(
    design: list[list[float]], weights: list[float]
) -> list[list[float]]:
    columns = len(design[0])
    return [
        [
            sum(weight * row[left] * row[right] for row, weight in zip(design, weights, strict=True))
            for right in range(columns)
        ]
        for left in range(columns)
    ]


def weighted_rhs(
    design: list[list[float]], weights: list[float], outcome: list[float]
) -> list[float]:
    columns = len(design[0])
    return [
        sum(
            weight * row[column] * value
            for row, weight, value in zip(design, weights, outcome, strict=True)
        )
        for column in range(columns)
    ]
