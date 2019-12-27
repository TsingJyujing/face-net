from fnet.util.random import sample_uniform_box, sample_wide_box
import numpy as np
from PIL import Image

if __name__ == '__main__':
    size_x = 500
    size_y = 500
    M = np.zeros(shape=(size_x, size_y), dtype="float")
    sample_count = 10000
    for i in range(sample_count):
        box = sample_wide_box(
            width=size_x,
            height=size_y,
            density_enforce=3
        )
        M[box.left:box.right, box.top:box.bottom] += (255.0 / sample_count)
    Image.fromarray(M).show()
