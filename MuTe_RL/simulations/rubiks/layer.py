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
    ) -> None:
        self.data = data
        self.face_index = face_index
        self.face_clockwise_rotator = face_clockwise_rotator
        self.face_anticlockwise_rotator = face_anticlockwise_rotator
        self.sides_indexes = sides_indexes
        self.side_length = int(sqrt(len(self.data[0])))

    def rotate_clockwise(self):
        if self.face_index is not None:
            self.data[self.face_index] = [
                item
                for row in self.face_clockwise_rotator(
                    np.array(self.data[self.face_index]).reshape(
                        self.side_length, self.side_length
                    )
                )
                for item in row
            ]
        self._cycle_edges(self.data, self.sides_indexes)

    def rotate_anticlockwise(self):
        if self.face_index is not None:
            self.data[self.face_index] = [
                item
                for row in self.face_anticlockwise_rotator(
                    np.array(self.data[self.face_index]).reshape(
                        self.side_length, self.side_length
                    )
                )
                for item in row
            ]
        self._cycle_edges(self.data, self.sides_indexes[::-1])

    def _cycle_edges(self, data, edges: List[Tuple[int, List[int]]]):
        next_edge_data = [
            [data[face_idx][idx] for idx in cell_idx]
            for face_idx, cell_idx in edges[-1:] + edges[:-1]
        ]

        for (face_idx, cell_idx), edge_data in zip(edges, next_edge_data):
            for idx, val in zip(cell_idx, edge_data):
                data[face_idx][idx] = val
