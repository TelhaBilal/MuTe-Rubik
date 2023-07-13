from typing import Tuple


def get_ansi_colored_text(text: str, fg_color: Tuple, bg_color: Tuple) -> str:

    assert len(fg_color) == 3, "fg_color param must have 3 values specifying r, g, b values of the color"
    assert len(bg_color) == 3, "fg_color param must have 3 values specifying r, g, b values of the color"

    r, g, b = fg_color
    result = f'\033[38;2;{r};{g};{b}m{text}'
    r, g, b = bg_color
    result = f'\033[48;2;{r};{g};{b}m{result}\033[0m'
    return result
