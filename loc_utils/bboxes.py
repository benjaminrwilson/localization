import torch

from coord_types import CoordType


class BBoxes(object):

    def __init__(self, bboxes, coord_type, img_size):
        self.attrs = {}
        self.bboxes = bboxes
        self.coord_type = coord_type
        self.size = img_size

    def convert(self, coord_type):
        if coord_type not in CoordType:
            return
        elif self.coord_type == coord_type:
            return self
        if coord_type == CoordType.XYXY:
            bboxes = self._convert2xyxy()
            res = BBoxes(bboxes, CoordType.XYXY, self.size)
        else:
            bboxes = self._convert2xywh()
            res = BBoxes(bboxes, CoordType.XYWH, self.size)
        return res

    def _convert2xywh(self):
        x1, y1, x2, y2 = self.bboxes
        x_min, y_min = 0, 0
        w = torch.clamp(x2 - x1 + 1, min=x_min)
        h = torch.clamp(y2 - y1 + 1, min=y_min)
        return x1, y1, w, h

    def _convert2xyxy(self):
        x1, y1, w, h = self.bboxes
        x_max, y_max = self.size - 1
        x2 = torch.clamp(x1 + w - 1, max=x_max)
        y2 = torch.clamp(y1 + h - 1, max=y_max)
        return x1, y1, x2, y2
