import torch

from loc_utils.coord_types import CoordType


class BBoxes(object):

    def __init__(self, coords, coord_type, img_size):
        self.attrs = {}
        self.coords = coords
        self.coord_type = coord_type
        self.size = img_size

    def convert(self, coord_type):
        if coord_type not in CoordType:
            return
        elif self.coord_type == coord_type:
            return self
        if coord_type == CoordType.XYXY:
            coords = self._convert2xyxy()
            bboxes = BBoxes(coords, CoordType.XYXY, self.size)
        else:
            coords = self._convert2xywh()
            bboxes = BBoxes(coords, CoordType.XYWH, self.size)
        return bboxes

    def resize(self):
        pass

    def _convert2xywh(self):
        x1, y1, x2, y2 = self.coords.split(1, dim=-1)
        x_min, y_min = 0, 0
        w = torch.clamp(x2 - x1 + 1, min=x_min)
        h = torch.clamp(y2 - y1 + 1, min=y_min)
        return torch.cat((x1, y1, w, h), dim=-1)

    def _convert2xyxy(self):
        x1, y1, w, h = self.coords
        x_max, y_max = self.size - 1
        x2 = torch.clamp(x1 + w - 1, max=x_max)
        y2 = torch.clamp(y1 + h - 1, max=y_max)
        return torch.cat((x1, y1, x2, y2), dim=-1)

    def __repr__(self):
        info = ("Num BBoxes: {}\n"
                "Coord Type: {}\n"
                "Size: {}").format(self.coords.shape[0],
                                   self.coord_type,
                                   self.size)
        return info
