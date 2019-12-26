from typing import List

import numpy
from PIL import Image

from fnet.util.label import Box, FaceLandmark


class Face:
    def __init__(self, box: Box, landmarks: FaceLandmark = None):
        self.landmarks = landmarks
        self.box = box

    def has_landmark(self):
        return self.landmarks is not None


class LabeledImage:
    def __init__(self, image: numpy.ndarray, faces: List[Face]):
        self.faces = faces
        self.image = image


def read_image(file) -> numpy.ndarray:
    return numpy.asarray(Image.open(file).convert("RGB"))
