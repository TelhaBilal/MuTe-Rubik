import random
import re
from copy import deepcopy
from math import sqrt
from typing import Dict, List, Iterable

import numpy as np

# from bin.color import BackgroundColors, Colors
from MuTe_Rubik.cube.layer import CubeLayer, CubeCell, CubeFaceIndexes, get_cube_layers_from_data
from MuTe_Rubik.base.rubik import RubikPuzzle, ColorOptions

from MuTe_Rubik.cube.utils import get_id_vector_from_cube_data


class Cube(RubikPuzzle):
    def __init__(
        self,
        data: List[List] | None = None,
        sides_length: int | None = None,
        disabled_layers: Iterable[int] | None = None,
        # :TODO: n_axis: int = 3 # x,y,z for cube, # 6 for dodecahedron?? 2 for pyramid??
    ) -> None:

        assert data or sides_length, "provide either 'sides_length' or 'data'"

        self.disabled_layers = disabled_layers

        init_data = data if data else self.__get_solved_state(sides_length)
        self.__init_from_data(init_data)

        self.moves_stack: List[str] = []
        self.disabled_layers = disabled_layers
        self._disable_layers(disabled_layers)

        self.possible_moves = self.get_all_possible_moves()

    def __init_from_data(self, init_data):

        sides_length = sqrt(len(init_data[0]))
        assert sides_length == int(sides_length), "provide a cube with square faces"

        self.n = int(sides_length)

        self.data = init_data
        self.layers: Dict[str, CubeLayer] = get_cube_layers_from_data(self.data, self.n)

        self.solved_backup = self.__get_solved_state(self.n)

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

    def shuffle(self, verbose: bool = False):
        for _ in range(random.randint(1, 50 * self.n**2)):
            move_id = random.choice(self.possible_moves)
            if verbose:
                print(move_id, end=" ")
            self.move(move_id)

        return self

    def is_solved(self) -> bool:
        # :TODO: Allow if cube is a rotated version of the solved state
        return self.to_vector() == get_id_vector_from_cube_data(self.solved_backup)

    def fraction_solved(self) -> float:
        # :TODO: Allow if cube is a rotated version of the solved state
        current_vector = np.array(self.to_vector())
        solved_vector = np.array(get_id_vector_from_cube_data(self.solved_backup))

        valid_mask = solved_vector != 0

        on_right_position_mask = current_vector == solved_vector
        solved = np.logical_and(on_right_position_mask, valid_mask)

        return solved.sum() / (valid_mask.sum() or 1)

    def to_vector(self) -> List[int]:
        # vector = Cube.data_to_vector(self.data)
        return get_id_vector_from_cube_data(self.data)

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

        self.__init_from_data(deepcopy(self.solved_backup))

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

    @classmethod
    def get_solved_cube(cls, n_sides):

        data = cls.__get_solved_state(n_sides)
        return Cube(data=data)

    @staticmethod
    def __get_solved_state(n_sides):

        return [[CubeCell(clr.value, i+1) for i in range(n_sides**2)] for clr in list(ColorOptions)]

    def _disable_layers(self, layer_ids: Iterable[str] | None):

        if not layer_ids:
            return

        for layer_id in layer_ids:
            if layer_id in self.layers:
                for face_index, tile_indexes in self.layers[layer_id].sides_indexes:
                    for ti in tile_indexes:
                        # self.data[face_index][ti] = (
                        #     BackgroundColors.BLACK.value.with_content(
                        #         f"{ti}", width=max(2, len(str(self.n**2 - 1)))
                        #     )
                        #     if self.numbered
                        #     else Colors.BLACK.value
                        # )
                        self.data[face_index][ti] = CubeCell(ColorOptions.BLACK.value, ti)
                if self.layers[layer_id].face_index is not None:
                    self.data[self.layers[layer_id].face_index] = [
                        CubeCell(ColorOptions.BLACK.value, i)
                        for i in range(self.n**2)
                    ]
                self.layers.pop(layer_id)

    def __repr__(self) -> str:
        # face_idx:
        # 0, 1 = front, back
        # 2, 3 = top, down
        # 4, 5 = left, right

        max_char_length = len(str(self.n**2))

        def make_matrix(face: List[List[CubeCell]], n):
            face = np.array(face).reshape(n, n)
            return [" ".join([item.get_padded_str(total_width=max_char_length) for item in line]) for line in face]

        lines = []
        lines.extend(
            [
                " " * (len(re.sub("\033[^m]*m", "", line)) + 1) + line
                for line in make_matrix(self.data[CubeFaceIndexes.TOP.value], self.n)
            ]
        )
        lines.extend(
            [
                " ".join(lines)
                for lines in zip(
                    make_matrix(self.data[CubeFaceIndexes.LEFT.value], self.n),
                    make_matrix(self.data[CubeFaceIndexes.FRONT.value], self.n),
                    make_matrix(self.data[CubeFaceIndexes.RIGHT.value], self.n),
                    make_matrix(self.data[CubeFaceIndexes.BACK.value], self.n),
                )
            ]
        )
        lines.extend(
            [
                " " * (len(re.sub("\033[^m]*m", "", line)) + 1) + line
                for line in make_matrix(self.data[CubeFaceIndexes.DOWN.value], self.n)
            ]
        )

        return "\n".join(lines) + "\n"

    def __deepcopy__(self):

        return Cube(data=deepcopy(self.data))
