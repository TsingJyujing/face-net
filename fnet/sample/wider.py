from typing import List, Dict
from fnet.util.image import Face
from fnet.util.label import Box, Range
import logging
from tqdm import tqdm

log = logging.getLogger(__file__)


def bbox_label_file_loader(filename: str) -> Dict[str, List[Face]]:
    """
    Read wider dataset labeling file (bbox only)
    :param filename:
    :return:
    """
    data = dict()
    flag = "name"
    current_name = None
    remain_logs = 0
    current_faces = []
    with open(filename, "r") as fp:
        for current_line in (rl.strip("\n") for rl in tqdm(fp.readlines()) if rl != "\n"):
            if flag == "name":
                current_name = current_line
                flag = "number"
            elif flag == "number":
                remain_logs = int(current_line)
                current_faces = []
                flag = "list"
            elif flag == "list":
                label_data = [int(s) for s in current_line.split(" ") if s != ""]
                if min(label_data[2:4]) > 0:
                    current_faces.append(Face(
                        box=Box(
                            x_range=Range(label_data[0], label_data[0] + label_data[2]),
                            y_range=Range(label_data[1], label_data[1] + label_data[3]),
                        )
                    ))
                remain_logs -= 1
                if remain_logs <= 0:
                    data[current_name] = current_faces
                    log.debug("Image {} have {} faces".format(
                        current_name, len(current_faces)
                    ))
                    flag = "name"
    return data
