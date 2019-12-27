from typing import Union, List, Callable


class Point:
    def __init__(self, x: int, y: int):
        self.x: int = x
        self.y: int = y

    def __sub__(self, o):
        return Point(- o.x + self.x, - o.y + self.y)

    def __add__(self, o):
        return Point(o.x + self.x, o.y + self.y)

    def __mul__(self, r: float):
        return Point(round(r * self.x), round(r * self.y))

    def rescale(self, x_scale: float, y_scale: float):
        return Point(round(x_scale * self.x), round(y_scale * self.y))


class Range:
    """
    Describe a range on R
    """

    __min = 0
    __max = 0

    def __init__(self, min_value: Union[float, int], max_value: Union[float, int]):
        assert max_value > min_value, "Max value less than min value: {}<{}".format(max_value, min_value)
        self.min_value = min_value
        self.max_value = max_value

    @property
    def length(self):
        return self.max_value - self.min_value

    def has_intersection(self, r):
        return not (r.min_value >= self.max_value or r.max_value < self.min_value)

    def __and__(self, r):
        """
        Get public (intersection) part of ranges
        :param r:
        :return:
        """
        if self.has_intersection(r):
            return Range(
                max(self.min_value, r.min_value),
                min(self.max_value, r.max_value)
            )
        else:
            raise Exception("Can't get intersect because of haven't public range")

    def __or__(self, r):
        """
        Get union result of ranges
        :param r:
        :return:
        """
        if self.has_intersection(r):
            return Range(
                min(self.min_value, r.min_value),
                max(self.max_value, r.max_value)
            )
        else:
            raise Exception("Can't get union because of haven't public range")


class Box:
    """
    A rectangle box
    """

    def __init__(self, x_range: Range, y_range: Range):
        self.x_range = x_range
        self.y_range = y_range

    def has_intersection(self, r):
        return self.x_range.has_intersection(r.x_range) and self.y_range.has_intersection(r.y_range)

    def __and__(self, r):
        if self.has_intersection(r):
            return Box(
                x_range=self.x_range & r.x_range,
                y_range=self.y_range & r.y_range,
            )
        else:
            raise Exception("Can't get intersect because of haven't public area")

    def calculate_iou(self, r):
        if self.has_intersection(r):
            intersection_area = (self & r).area
            return intersection_area / (self.area + r.area - intersection_area)
        else:
            return 0.0

    @property
    def width(self):
        return self.x_range.length

    @property
    def height(self):
        return self.y_range.length

    @property
    def area(self):
        return self.width * self.height

    @property
    def top(self):
        return self.y_range.min_value

    @property
    def bottom(self):
        return self.y_range.max_value

    @property
    def left(self):
        return self.x_range.min_value

    @property
    def right(self):
        return self.x_range.max_value


class ScoredBox(Box):
    """
    Box with scores
    """

    def __init__(self, x_range: Range, y_range: Range, score: float):
        super().__init__(x_range, y_range)
        self.score = score


class FaceLandmark:
    """
    Face 5 point landmark
    """

    def __init__(
            self,
            left_eye: Point,
            right_eye: Point,
            nose: Point,
            mouth_left: Point,
            mouth_right: Point
    ):
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.nose = nose
        self.mouth_left = mouth_left
        self.mouth_right = mouth_right

    def relative_transform(self, top_left_point: Point):
        return self._point_transformer(lambda p: p - top_left_point)

    def relative_transform_box(self, box: Box):
        x_scale = 1.0 / box.width
        y_scale = 1.0 / box.height
        top_left_point = Point(box.x_range.min_value, box.y_range.min_value)
        return self._point_transformer(lambda p: (p - top_left_point).rescale(x_scale, y_scale))

    def _point_transformer(self, func: Callable[[Point], Point]):
        return FaceLandmark(
            func(self.left_eye),
            func(self.right_eye),
            func(self.nose),
            func(self.mouth_left),
            func(self.mouth_right),
        )


def non_maximum_suppression(scored_boxes: List[ScoredBox], iou_threshold: float) -> List[ScoredBox]:
    """
    NMS algorithm on box list
    :param scored_boxes:
    :param iou_threshold:
    :return:
    """
    keep = []
    sorted_boxes = sorted(scored_boxes, key=lambda b: -b.score)
    while len(sorted_boxes) > 0:
        current_box = sorted_boxes.pop(0)
        keep.append(current_box)
        sorted_boxes = filter(
            lambda e: current_box.calculate_iou(e) >= iou_threshold,
            sorted_boxes,
        )
    return keep

# todo Soft NMS algorithm: https://arxiv.org/pdf/1704.04503.pdf
# ref code here: https://github.com/DocF/Soft-NMS/blob/master/soft_nms.py
