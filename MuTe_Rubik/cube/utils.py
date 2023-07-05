from typing import List

import numpy as np


def rotate_matrix_clockwise(matrix: np.ndarray):

    matrix = np.array(matrix)
    return matrix.T[..., ::-1]


def rotate_matrix_anticlockwise(matrix: np.ndarray):

    matrix = np.array(matrix)
    return matrix.T[::-1]


def get_id_vector_from_cube_data(data: List[List]):

      return [cell.color.color_id for face in data for cell in face]
