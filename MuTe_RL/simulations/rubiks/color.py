import enum


class Color:
    """binds together display color string and integer encoding of the color"""

    def __init__(self, integer: int, string: str) -> None:
        self.int = integer
        self.str = string

    def __repr__(self) -> str:
        # return f'"int: {self.int} str: {self.str}"'
        return self.str

    def with_content(self, content):
        self.str = self.str.replace("■", str(content))
        return self

class Colors(enum.Enum):
    """enumerates all the colors on rubik's cubes"""

    # define opposing colors in pairs
    RED = Color(0, "\033[91m" + "■" + "\033[0m")
    ORANGE = Color(1, "\033[38;5;216m" + "■" + "\033[0m")

    BLUE = Color(2, "\033[94m" + "■" + "\033[0m")
    GREEN = Color(3, "\033[92m" + "■" + "\033[0m")

    WHITE = Color(4, "\033[97m" + "■" + "\033[0m")
    YELLOW = Color(5, "\033[93m" + "■" + "\033[0m")

    BLACK = Color(6, "\033[30m" + "■" + "\033[0m")


class BackgroundColors(enum.Enum):
    """defines strings wrapping a box in ANSI escape sequence giving the effect of background colors"""

    # :TODO: define Color(...) objects instead
    # :TODO: create initial data such that tiles are colored indexes in face

    BLACK = "\033[40m" + "■" + "\033[0m"
    RED = "\033[41m" + "■" + "\033[0m"
    GREEN = "\033[42m" + "■" + "\033[0m"
    YELLOW = "\033[43m" + "■" + "\033[0m"
    BLUE = "\033[44m" + "■" + "\033[0m"
    PURPLE = "\033[45m" + "■" + "\033[0m"
    CYAN = "\033[46m" + "■" + "\033[0m"

    # BackgroundColors.GREEN.value.replace("■", f"{d:^{len(str(n))}}")