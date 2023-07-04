import enum


class Color:
    """binds together display color string and integer encoding of the color"""

    def __init__(self, integer: int, string: str) -> None:
        self.int = integer
        self.str = string

    def __repr__(self) -> str:
        # return f'"int: {self.int} str: {self.str}"'
        return self.str

    def with_content(self, content, width=0):
        new_object = Color(self.int, self.str.replace("■", f"{content:^{width}}"))
        return new_object


class Colors(enum.Enum):
    """enumerates all the colors on rubik's cubes"""

    # define opposing colors in pairs
    RED = Color(1, "\033[91m" + "■" + "\033[0m")
    ORANGE = Color(2, "\033[38;5;216m" + "■" + "\033[0m")

    BLUE = Color(3, "\033[94m" + "■" + "\033[0m")
    GREEN = Color(4, "\033[92m" + "■" + "\033[0m")

    WHITE = Color(5, "\033[97m" + "■" + "\033[0m")
    YELLOW = Color(6, "\033[93m" + "■" + "\033[0m")

    BLACK = Color(0, "\033[30m" + "■" + "\033[0m")


class BackgroundColors(enum.Enum):
    """defines Color object that add background color to text using ANSI escape sequence"""

    RED = Color(1, "\033[41m" + "■" + "\033[0m")
    ORANGE = Color(2, "\033[48;5;208m" + "■" + "\033[0m")

    BLUE = Color(3, "\033[44m" + "■" + "\033[0m")
    GREEN = Color(4, "\033[42m" + "■" + "\033[0m")

    WHITE = Color(5, "\033[48;5;15m" + "■" + "\033[0m")
    YELLOW = Color(6, "\033[43m" + "■" + "\033[0m")

    BLACK = Color(0, "\033[40m" + "■" + "\033[0m")

    PURPLE = Color(7, "\033[45m" + "■" + "\033[0m")
    CYAN = Color(8, "\033[46m" + "■" + "\033[0m")
