"""simulate any rubik's puzzle like a cube or any other polyhedron"""

from enum import Enum
from abc import ABC, abstractmethod, abstractstaticmethod
from typing import List


class GameColor:

    def __init__(self, color_id: int, rgb: List):

        assert len(rgb) == 3, "Length of RGB param must be 3: for red blue and green colors"

        self.color_id = color_id
        self.rgb = rgb


class ColorOptions(Enum):

    RED = GameColor(1, (255, 0, 0))
    ORANGE = GameColor(2, (255, 165, 0))

    WHITE = GameColor(3, (255, 255, 255))
    YELLOW = GameColor(4, (255, 255, 0))

    BLUE = GameColor(5, (0, 0, 255))
    GREEN = GameColor(6, (0, 255, 0))


    BLACK = GameColor(0, (0, 0, 0))


class RubikPuzzle(ABC):

    @abstractmethod
    def get_all_possible_moves(self) -> List:
        pass

    @abstractmethod
    def make_move(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def is_solved(self) -> bool:
        pass

    def __repr__(self):
        raise NotImplementedError('Not Implemented')
