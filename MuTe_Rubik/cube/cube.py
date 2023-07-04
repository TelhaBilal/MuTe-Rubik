import enum
import random
import re
from copy import deepcopy
from math import sqrt
from typing import Dict, List, Iterable

import numpy as np

# from bin.color import BackgroundColors, Colors
from MuTe_Rubik.cube.layer import CubeLayer, get_cube_layers_from_data
from MuTe_Rubik.base.rubik import RubikPuzzle, ColorOptions, GameColor

from MuTe_Rubik.cube.utils import flatten_data_list


class Cube(RubikPuzzle):
    def __init__(
        self,
        data: List[List] | None = None,
        sides_length: int | None = None,
        numbered=True,
        disabled_layers: Iterable[int] | None = None,
        # :TODO: n_axis: int = 3 # x,y,z for cube, # 6 for dodecahedron?? 2 for pyramid??
    ) -> None:

        assert data or sides_length, "provide either 'sides_length' or 'data'"

        self.numbered = numbered
        self.disabled_layers = disabled_layers

        self.data = data if data else self._solved_state(sides_length, numbered=numbered)
        sides_length = sqrt(len(self.data[0]))

        assert sides_length == int(sides_length), "provide a cube with square faces"

        self.n = int(sides_length)

        self.layers: Dict[str, CubeLayer] = get_cube_layers_from_data(data, sides_length)

        self.moves_stack: List[str] = []
        self.disabled_layers = disabled_layers
        self._disable_layers(disabled_layers)
        self.solved_backup = (
            deepcopy(self.data)
            if data is None
            else self._solved_state(self.n, numbered=self.numbered)
        )
        self.possible_moves = self.get_all_possible_moves()


    def get_all_possible_moves(self):
        return [item for key in self.layers.keys() for item in [key, key + "'"]]


    # TODO: alias of make_move function for continuity purpose. delete after fixing uses
    def move(self, *args, **kwargs):
        return self.make_move(*args, **kwargs)


    def make_move(self, move_id: str):
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

    def to_vector(self) -> List[int]:
        # vector = Cube.data_to_vector(self.data)
        return flatten_data_list(self.data)

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

        # TODO: Facepalm your face with a metal wall
        # solved_data = deepcopy(self.solved_backup)
        # for i in range(len(self.data)):
        #     self.data[i] = solved_data[i]

        self.data = deepcopy(self.solved_data)

        # TODO: reset history?
        # self.moves_stack = []

        return self

    # def _solved_state(self, n_layers, numbered=False):
    #     if numbered:
    #         return [
    #             [
    #                 clr.value.with_content(
    #                     f"{i}", width=max(2, len(str(n_layers**2 - 1)))
    #                 )
    #                 for i in range(n_layers**2)
    #             ]
    #             for clr in list(BackgroundColors)[:6]
    #         ]
    #     return [[clr.value for i in range(n_layers**2)] for clr in list(Colors)[:6]]

    @staticmethod
    def get_solved(n_sides):

        data = [[CubeCell(clr, i+1) for i in range(n_sides**2)] for clr in list(ColorOptions)]
        return Cube(data)

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
                for line in make_matrix(self.data[FaceIndexes.TOP.value], self.n)
            ]
        )
        lines.extend(
            [
                " ".join(lines)
                for lines in zip(
                    make_matrix(self.data[FaceIndexes.LEFT.value], self.n),
                    make_matrix(self.data[FaceIndexes.FRONT.value], self.n),
                    make_matrix(self.data[FaceIndexes.RIGHT.value], self.n),
                    make_matrix(self.data[FaceIndexes.BACK.value], self.n),
                )
            ]
        )
        lines.extend(
            [
                " " * (len(re.sub("\033[^m]*m", "", line)) + 1) + line
                for line in make_matrix(self.data[FaceIndexes.DOWN.value], self.n)
            ]
        )

        return "\n".join(lines) + "\n"
