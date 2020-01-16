from typing import List

from PIL import Image

from fnet.sample.wider import bbox_label_file_loader
from fnet.util.label import Box


def get_sample_from_image(image: Image.Image, faces: List[Box], output_size: int):
    """
    return X and y
    X is 3(channels)*output_size*output_size tensor
    y is 15 (1+4+10) unified label data
    :param image:
    :param faces:
    :param output_size:
    :return:
    """
    pass


if __name__ == '__main__':
    r = bbox_label_file_loader("data/wider/wider_face_split/wider_face_train_bbx_gt.txt")
    print("There're {} images.".format(len(r)))
    print("Include {} faces".format(sum(
        len(v) for k, v in r.items()
    )))
