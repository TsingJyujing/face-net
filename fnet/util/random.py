from math import floor
import random

from fnet.util.label import Box, Range


def __rand_int(a: int, b: int, reverse_density: bool = False, density_enforce: int = 2) -> int:
    f = random.random() ** density_enforce
    s = (1 - f) if reverse_density else f
    return floor((b - a + 1) * s)


def sample_uniform_box(width: int, height: int, min_width: int = 1, min_height: int = 1):
    left = random.randint(0, width - 1 - min_width)
    top = random.randint(0, height - 1 - min_height)
    return Box(
        Range(left, random.randint(left + 1, width - 1)),
        Range(top, random.randint(top + 1, height - 1)),
    )


def sample_wide_box(width: int, height: int, min_width: int = 1, min_height: int = 1, density_enforce: int = 2):
    left = __rand_int(0, width - 1 - min_width, reverse_density=False, density_enforce=density_enforce)
    top = __rand_int(0, height - 1 - min_height, reverse_density=False, density_enforce=density_enforce)
    return Box(
        Range(left, __rand_int(left + 1, width - 1, reverse_density=True, density_enforce=density_enforce)),
        Range(top, __rand_int(top + 1, height - 1, reverse_density=True, density_enforce=density_enforce)),
    )
