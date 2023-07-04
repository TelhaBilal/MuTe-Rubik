import enum
import random
import re
from copy import deepcopy
from math import sqrt
from typing import Dict, List, Iterable

import numpy as np

from MuTe_RL.simulations.rubiks.color import BackgroundColors, Colors
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
    def __init__(
        self,
        data: List[List] | None = None,
        n_layers: int | None = None,
        numbered=True,
        disabled_layers: Iterable[int] | None = None,
        # :TODO: n_axis: int = 3 # x,y,z for cube, # 6 for dodecahedron?? 2 for pyramid??
    ) -> None:
        self.data = data
        assert not (data and n_layers), "provide at least 'n' or 'data'"

        self.numbered = numbered
        self.disabled_layers = disabled_layers

        if self.data is None:
            # initialize with solved Cube
            self.data = self._solved_state(n_layers, numbered=numbered)

        n_layers = sqrt(len(self.data[0]))
        assert n_layers == int(n_layers), "provide a cube with square faces"
        n = int(n_layers)
        self.n = int(n_layers)

        grid = np.arange(n**2).reshape(n, n)

        self.layers: Dict[str, Layer] = {
            key: val
            for i in range(self.n)
            for key, val in {
                f"x{i}": Layer(
                    data=self.data,
                    face_index=(
                        Face.LEFT.value
                        if i == 0
                        else Face.RIGHT.value
                        if i == self.n - 1
                        else None
                    ),
                    face_clockwise_rotator=self._rotate_matrix_clockwise,
                    face_anticlockwise_rotator=self._rotate_matrix_anticlockwise,
                    sides_indexes=[
                        (Face.TOP.value, grid[:, i].tolist()),
                        (Face.FRONT.value, grid[:, i].tolist()),
                        (Face.DOWN.value, grid[:, i].tolist()[::-1]),
                        (Face.BACK.value, grid[:, -1 - i].tolist()),
                    ],
                    backward_face=i == self.n - 1,
                ),
                f"y{i}": Layer(
                    data=self.data,
                    face_index=(
                        Face.DOWN.value
                        if i == 0
                        else Face.TOP.value
                        if i == self.n - 1
                        else None
                    ),
                    face_clockwise_rotator=self._rotate_matrix_clockwise,
                    face_anticlockwise_rotator=self._rotate_matrix_anticlockwise,
                    sides_indexes=[
                        (Face.FRONT.value, grid[-(i + 1), :].tolist()),
                        (Face.RIGHT.value, grid[-(i + 1), :].tolist()),
                        (Face.BACK.value, grid[-(i + 1), :].tolist()[::-1]),
                        (Face.LEFT.value, grid[-(i + 1), :].tolist()[::-1]),
                    ],
                    backward_face=i == self.n - 1,
                ),
                f"z{i}": Layer(
                    data=self.data,
                    face_index=(
                        Face.FRONT.value
                        if i == 0
                        else Face.BACK.value
                        if i == self.n - 1
                        else None
                    ),
                    face_clockwise_rotator=self._rotate_matrix_clockwise,
                    face_anticlockwise_rotator=self._rotate_matrix_anticlockwise,
                    sides_indexes=[
                        (Face.TOP.value, grid[-(i + 1), :].tolist()),
                        (Face.RIGHT.value, grid[:, i].tolist()),
                        (Face.DOWN.value, grid[i, :].tolist()),
                        (Face.LEFT.value, grid[:, -(i + 1)].tolist()),
                    ],
                    backward_face=i == self.n - 1,
                ),
            }.items()
        }

        self.moves_stack: List[str] = []
        self.disabled_layers = disabled_layers
        self._disable_layers(disabled_layers)
        self.solved_backup = (
            deepcopy(self.data)
            if data is None
            else self._solved_state(self.n, numbered=self.numbered)
        )
        self.possible_moves = [
            item for key in self.layers.keys() for item in [key, key + "'"]
        ]

    def move(self, move_id: str):
        # parse input
        move_id = move_id.lower()
        layer_id = move_id.strip("'")

        # make move
        if move_id[-1] == "'":
            self.layers[layer_id].rotate_anticlockwise()
        else:
            self.layers[layer_id].rotate_clockwise()

        # maintain history
        if self.moves_stack:
            if (
                # opposite moves --> pop
                self.moves_stack[-1].strip("'") == move_id.strip("'")
                and (
                    (self.moves_stack[-1][-1] == "'" and move_id[-1] != "'")
                    or (self.moves_stack[-1][-1] != "'" and move_id[-1] == "'")
                )
            ):
                self.moves_stack.pop(-1)
            else:
                self.moves_stack.append(move_id)
        else:
            self.moves_stack.append(move_id)

        return self

    def shuffle(self, verbose=False):
        for _ in range(random.randint(1, 50 * self.n**2)):
            move_id = random.choice(self.possible_moves)
            if verbose:
                print(move_id, end=" ")
            self.move(move_id)

        return self

    def is_solved(self) -> bool:
        # :TODO: Allow if cube is a rotated version of the solved state
        return self.to_vector() == Cube.data_to_vector(self.solved_backup)

    def fraction_solved(self) -> float:
        # :TODO: Allow if cube is a rotated version of the solved state
        current_vector = np.array(self.to_vector())
        solved_vector = np.array(Cube.data_to_vector(self.solved_backup))

        valid_mask = solved_vector != 0

        on_right_position_mask = current_vector == solved_vector
        solved = np.logical_and(on_right_position_mask, valid_mask)

        return solved.sum() / (valid_mask.sum() or 1)

    @staticmethod
    def data_to_vector(data):
        vector = [color_tile.int for face in data for color_tile in face]
        return vector

    def to_vector(self) -> List[int]:
        vector = Cube.data_to_vector(self.data)

        return vector

    def _grow_to_size(self, data: List[List], size: int):
        data = deepcopy(data)
        original_size = len(data[0])

        if size <= original_size:
            return data

        if original_size % 2 == 1:
            # :TODO: Implement cube grow/collapse function
            # repeat_center
            # moves?? softmax on x,y,z & sigmoid on layer_no ??
            pass
        else:
            # ???
            pass
        return data

    def reset(self):
        solved_data = deepcopy(self.solved_backup)
        for i in range(len(self.data)):
            self.data[i] = solved_data[i]

        return self

    def _solved_state(self, n_layers, numbered=False):
        if numbered:
            return [
                [
                    clr.value.with_content(
                        f"{i}", width=max(2, len(str(n_layers**2 - 1)))
                    )
                    for i in range(n_layers**2)
                ]
                for clr in list(BackgroundColors)[:6]
            ]
        return [[clr.value for i in range(n_layers**2)] for clr in list(Colors)[:6]]

    def _disable_layers(self, layer_ids: Iterable[str] | None):
        if not layer_ids:
            return

        for layer_id in layer_ids:
            if layer_id in self.layers:
                for face_index, tile_indexes in self.layers[layer_id].sides_indexes:
                    for ti in tile_indexes:
                        self.data[face_index][ti] = (
                            BackgroundColors.BLACK.value.with_content(
                                f"{ti}", width=max(2, len(str(self.n**2 - 1)))
                            )
                            if self.numbered
                            else Colors.BLACK.value
                        )
                if self.layers[layer_id].face_index is not None:
                    self.data[self.layers[layer_id].face_index] = [
                        BackgroundColors.BLACK.value.with_content(
                            f"{i}", width=max(2, len(str(self.n**2 - 1)))
                        )
                        if self.numbered
                        else Colors.BLACK.value
                        for i in range(self.n**2)
                    ]
                self.layers.pop(layer_id)

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
            return [" ".join([item.str for item in line]) for line in face]

        lines = []
        lines.extend(
            [
                " " * (len(re.sub("\033[^m]*m", "", line)) + 1) + line
                for line in make_matrix(self.data[Face.TOP.value], self.n)
            ]
        )
        lines.extend(
            [
                " ".join(lines)
                for lines in zip(
                    make_matrix(self.data[Face.LEFT.value], self.n),
                    make_matrix(self.data[Face.FRONT.value], self.n),
                    make_matrix(self.data[Face.RIGHT.value], self.n),
                    make_matrix(self.data[Face.BACK.value], self.n),
                )
            ]
        )
        lines.extend(
            [
                " " * (len(re.sub("\033[^m]*m", "", line)) + 1) + line
                for line in make_matrix(self.data[Face.DOWN.value], self.n)
            ]
        )

        return "\n".join(lines) + "\n"
