import numpy as np


def rotate_matrix_clockwise(matrix: np.ndarray):

    matrix = np.array(matrix)
    return matrix.T[..., ::-1]


def rotate_matrix_anticlockwise(matrix: np.ndarray):

    matrix = np.array(matrix)
    return matrix.T[::-1]


def flatten_cube_data_list(data):

      return [cell for face in data for cell in face]
