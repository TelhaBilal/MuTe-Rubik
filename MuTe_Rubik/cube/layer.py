from math import sqrt
from typing import Callable, List, Tuple
import enum

import numpy as np

from MuTe_Rubik.cube.utils import rotate_matrix_clockwise, rotate_matrix_anticlockwise
from MuTe_Rubik.utils import get_ansi_colored_text
from MuTe_Rubik.base.rubik import GameColor


class CubeFaceIndexes(enum.Enum):
    FRONT = 0
    BACK = 1

    TOP = 2
    DOWN = 3

    LEFT = 4
    RIGHT = 5


class CubeCell:

    def __init__(self, color: GameColor, cell_id: int | None = None, display_ids=True):

        self.color = color
        self.cell_id = cell_id
        self.__fg_color_rgb = [255-c for c in color.rgb]  # loop over rgb values

        self.display_ids = display_ids

    def __repr__(self):

        return get_ansi_colored_text(self.cell_id if self.display_ids and self.cell_id else " ",
                                     self.__fg_color_rgb,
                                     self.color.rgb)

    def get_padded_str(self,
                       total_width: int | None = None,
                       len_l_pad: int = 0,
                       len_r_pad: int = 0,
                       padding_char: str = ' '):

        printed_char = str(self.cell_id) if self.display_ids and self.cell_id else " "

        if total_width is not None:
            assert total_width >= 0, "total width cannot be negative"

            len_diff = total_width-len(printed_char)

            if len_diff >= 0:
                if len_diff % 2 == 0:
                    len_l_pad = len_r_pad = int(len_diff / 2)
                else:
                    len_r_pad = int((len_diff-1/2))
                    len_l_pad = len_r_pad + 1

        padded_char = (padding_char*len_l_pad)+printed_char+(padding_char*len_r_pad)

        return get_ansi_colored_text(padded_char,
                                     self.__fg_color_rgb,
                                     self.color.rgb)



class CubeLayer:
    def __init__(
        self,
        data: List[List[int]],
        face_index: int | None,
        sides_indexes: List[Tuple[int, List[int]]],
        backward_face: bool,
    ) -> None:
        self.data = data
        self.face_index = face_index
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

        grid = (
            rotate_matrix_clockwise(grid)
            if clockwise
            else rotate_matrix_anticlockwise(grid)
        )

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


def get_cube_layers_from_data(data: List[List[CubeCell]], side_length: int) -> List:

    grid = np.arange(side_length**2).reshape(side_length, side_length)

    layers = {
            key: val
            for i in range(side_length)
            for key, val in {
                f"x{i}": CubeLayer(
                    data=data,
                    face_index=(
                        CubeFaceIndexes.LEFT.value
                        if i == 0
                        else CubeFaceIndexes.RIGHT.value
                        if i == side_length - 1
                        else None
                    ),
                    sides_indexes=[
                        (CubeFaceIndexes.TOP.value, grid[:, i].tolist()),
                        (CubeFaceIndexes.FRONT.value, grid[:, i].tolist()),
                        (CubeFaceIndexes.DOWN.value, grid[:, i].tolist()[::-1]),
                        (CubeFaceIndexes.BACK.value, grid[:, -1 - i].tolist()),
                    ],
                    backward_face=i == side_length - 1,
                ),
                f"y{i}": CubeLayer(
                    data=data,
                    face_index=(
                        CubeFaceIndexes.DOWN.value
                        if i == 0
                        else CubeFaceIndexes.TOP.value
                        if i == side_length - 1
                        else None
                    ),
                    sides_indexes=[
                        (CubeFaceIndexes.FRONT.value, grid[-(i + 1), :].tolist()),
                        (CubeFaceIndexes.RIGHT.value, grid[-(i + 1), :].tolist()),
                        (CubeFaceIndexes.BACK.value, grid[-(i + 1), :].tolist()[::-1]),
                        (CubeFaceIndexes.LEFT.value, grid[-(i + 1), :].tolist()[::-1]),
                    ],
                    backward_face=i == side_length - 1,
                ),
                f"z{i}": CubeLayer(
                    data=data,
                    face_index=(
                        CubeFaceIndexes.FRONT.value
                        if i == 0
                        else CubeFaceIndexes.BACK.value
                        if i == side_length - 1
                        else None
                    ),
                    sides_indexes=[
                        (CubeFaceIndexes.TOP.value, grid[-(i + 1), :].tolist()),
                        (CubeFaceIndexes.RIGHT.value, grid[:, i].tolist()),
                        (CubeFaceIndexes.DOWN.value, grid[i, :].tolist()),
                        (CubeFaceIndexes.LEFT.value, grid[:, -(i + 1)].tolist()),
                    ],
                    backward_face=i == side_length - 1,
                ),
            }.items()
        }

    return layers
