import enum

import random
from math import sqrt
from typing import Dict, List

import numpy as np

from MuTe_RL.simulations.rubiks.color import Colors
from MuTe_RL.simulations.rubiks.layer import Layer
from MuTe_RL.simulations.rubiks.rubik import Rubik


class Face(enum.Enum):
    FRONT = 0
    BACK = 1

    TOP = 2
    DOWN = 3

    LEFT = 4
    RIGHT = 5


class Cube(Rubik):
    def __init__(self, data=None, n_layers: int | None = None) -> None:
        self.data = data
        assert not (data and n_layers), "provide at least 'n' or 'data'"

        if self.data is None:
            # initialize with solved Cube
            self.data = self._solved_state(n_layers)

        n_layers = sqrt(len(self.data[0]))
        assert n_layers == int(n_layers), "provide a cube with square faces"
        n = int(n_layers)
        self.n = int(n_layers)

        self.layers: Dict[str, Layer] = {
            "x0": Layer(
                data=self.data,
                face_index=Face.LEFT.value,
                face_clockwise_rotator=self._rotate_matrix_clockwise,
                face_anticlockwise_rotator=self._rotate_matrix_anticlockwise,
                sides_indexes=[
                    (Face.TOP.value, np.arange(n**2).reshape(n, n)[:, 0].tolist()),
                    (Face.FRONT.value, np.arange(n**2).reshape(n, n)[:, 0].tolist()),
                    (Face.DOWN.value, np.arange(n**2).reshape(n, n)[:, 0].tolist()),
                    (Face.BACK.value, np.arange(n**2).reshape(n, n)[:, 0].tolist()),
                ],
            ),

            # :TODO: track cell index from display and provide right sides_indexes.
            "z0": Layer(
                data=self.data,
                face_index=Face.FRONT.value,
                face_clockwise_rotator=self._rotate_matrix_clockwise,
                face_anticlockwise_rotator=self._rotate_matrix_anticlockwise,
                sides_indexes=[
                    (Face.TOP.value, np.arange(n**2).reshape(n, n)[-1, :].tolist()),
                    (Face.RIGHT.value, np.arange(n**2).reshape(n, n)[:, 0].tolist()),
                    (Face.DOWN.value, np.arange(n**2).reshape(n, n)[0, :].tolist()),
                    (Face.LEFT.value, np.arange(n**2).reshape(n, n)[:, -1].tolist()),
                ],
            ),
            # :TODO: observe the pattern & automate generation of all layer possible objects
        }

    def moves_list(self):
        return [item for key in self.layers.keys() for item in [key, key + "'"]]

    def move(self, move_id: str):
        layer_id = move_id.strip("'")
        if move_id[-1] == "'":
            self.layers[layer_id].rotate_anticlockwise()
        else:
            self.layers[layer_id].rotate_clockwise()

        return self

    def shuffle(self):
        for _ in range(random.randint(1, 50 * self.n**2)):
            move_id = random.choice(self.moves_list())
            self.move(move_id)

        return self

    def to_vector(self) -> List[int]:
        # :TODO: get rid of ".value" for enumerated objects by initializing the data with the value object

        vector = [item.value.int for face in self.data for item in face]

        return vector

    def reset(self):
        solved_data = self._solved_state(self.n)
        for i in range(len(self.data)):
            self.data[i] = solved_data[i]

        return self

    def _solved_state(self, n_layers):
        return [[clr for _ in range(n_layers**2)] for clr in list(Colors)[:6]]

    def _rotate_matrix_clockwise(self, matrix: np.ndarray):
        matrix = np.array(matrix)
        return matrix.T[..., ::-1]

    def _rotate_matrix_anticlockwise(self, matrix: np.ndarray):
        matrix = np.array(matrix)
        return matrix.T[::-1]

    def __repr__(self) -> str:
        # face_idx:
        # 0, 1 = front, back
        # 2, 3 = top, down
        # 4, 5 = left, right

        def make_matrix(face, n):
            face = np.array(face).reshape(n, n)
            return [
                # :TODO: get rid of ".value" for enumerated objects by initializing the data with the value object
                " ".join([item.value.str for i, item in enumerate(line)])
                for j, line in enumerate(face)
            ]

        lines = []
        lines.extend(
            [
                " " * 2 * (line.count(" ") + 1) + line
                for line in make_matrix(self.data[2], self.n)
            ]
        )
        lines.extend(
            [
                " ".join(lines[:3]) + " " * 2 + lines[3]
                for lines in zip(
                    make_matrix(self.data[4], self.n),
                    make_matrix(self.data[0], self.n),
                    make_matrix(self.data[5], self.n),
                    make_matrix(self.data[1], self.n),
                )
            ]
        )
        lines.extend(
            [
                " " * 2 * (line.count(" ") + 1) + line
                for line in make_matrix(self.data[3], self.n)
            ]
        )

        return "\n".join(lines) + "\n"
