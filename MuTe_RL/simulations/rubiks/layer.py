from math import sqrt
from typing import Callable, List, Tuple

import numpy as np


class Layer:
    def __init__(
        self,
        data: List[List[int]],
        face_index: int | None,
        face_clockwise_rotator: Callable,
        face_anticlockwise_rotator: Callable,
        sides_indexes: List[Tuple[int, List[int]]],
        backward_face: bool,
    ) -> None:
        self.data = data
        self.face_index = face_index
        self.face_clockwise_rotator = face_clockwise_rotator
        self.face_anticlockwise_rotator = face_anticlockwise_rotator
        self.sides_indexes = sides_indexes
        self.side_length = int(sqrt(len(self.data[0])))
        self.backward_face = backward_face

    def rotate_clockwise(self):
        self._rotate(self.data, self.face_index, self.sides_indexes, clockwise=True)

    def rotate_anticlockwise(self):
        self._rotate(self.data, self.face_index, self.sides_indexes, clockwise=False)

    def _rotate(
        self,
        data,
        face_index: int | None,
        edges: List[Tuple[int, List[int]]],
        clockwise: bool,
    ):
        edge_data = [
            [data[face_idx][idx] for idx in cell_idx] for face_idx, cell_idx in edges
        ]
        default = face_index or list(set(range(6)) - set(i for i, _ in edges))[0]
        grid = np.full(
            (self.side_length + 2, self.side_length + 2),
            data[default][int(len(data[0]) / 2)],
        )
        grid[0, 1:-1] = edge_data[0]
        grid[1:-1, -1] = edge_data[1]
        grid[-1, 1:-1] = edge_data[2]
        grid[1:-1, 0] = edge_data[3]

        if face_index is not None:
            grid[1:-1, 1:-1] = np.array(data[face_index]).reshape(
                self.side_length, self.side_length
            )
            if self.backward_face:
                grid[1:-1, 1:-1] = grid[1:-1, 1:-1][::-1, ::-1]

        # print("before")
        # _ = [print(row.tolist()) for row in grid]

        grid = (
            self.face_clockwise_rotator(grid)
            if clockwise
            else self.face_anticlockwise_rotator(grid)
        )

        # print("after")
        # _ = [print(row.tolist()) for row in grid]

        if face_index is not None:
            data[face_index] = [item for row in grid[1:-1, 1:-1] for item in row]

        next_edge_data = [
            grid[0, 1:-1],
            grid[1:-1, -1],
            grid[-1, 1:-1],
            grid[1:-1, 0],
        ]

        for (face_idx, cell_idx), edge_data in zip(edges, next_edge_data):
            for idx, val in zip(cell_idx, edge_data):
                data[face_idx][idx] = val
