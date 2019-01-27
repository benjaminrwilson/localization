from enum import Enum

import torch


class CoordType(Enum):
    XYXY = 0
    XYWH = 1


class BBoxes(object):

    def __init__(self, coords, coord_type, img_size):
        self._attrs = Attrs()
        self._coords = coords
        self._coord_type = coord_type
        self._shape = coords.shape[0]
        self._size = img_size

    def convert(self, coord_type):
        if not isinstance(coord_type, CoordType):
            raise ValueError("Invalid coordinate type!")
        elif self._coord_type == coord_type:
            return self
        if coord_type == CoordType.XYXY:
            coords = self._convert2xyxy()
            bboxes = BBoxes(coords, CoordType.XYXY, self._size)
        else:
            coords = self._convert2xywh()
            bboxes = BBoxes(coords, CoordType.XYWH, self._size)
        return bboxes

    def resize(self):
        pass

    @property
    def attrs(self):
        return self._attrs

    @attrs.setter
    def attrs(self, name, vals):
        _validate(vals)
        self._attrs[name] = vals

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, coords):
        _validate(coords)
        self._coords = coords

    @property
    def coord_type(self):
        return self._coord_type

    @coords.setter
    def coords(self, coord_type):
        _validate(coord_type)
        self._coord_type = coord_type

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size

    def _convert2xywh(self):
        x1, y1, x2, y2 = self._coords.split(1, dim=-1)
        x_min, y_min = 0, 0
        w = torch.clamp(x2 - x1 + 1, min=x_min)
        h = torch.clamp(y2 - y1 + 1, min=y_min)
        return torch.cat((x1, y1, w, h), dim=-1)

    def _convert2xyxy(self):
        x1, y1, w, h = self._coords
        x_max, y_max = self._size - 1
        x2 = torch.clamp(x1 + w - 1, max=x_max)
        y2 = torch.clamp(y1 + h - 1, max=y_max)
        return torch.cat((x1, y1, x2, y2), dim=-1)

    def __len__(self):
        return self._coords.shape[0]

    def __repr__(self):
        info = ("Num BBoxes: {}\n"
                "Coord Type: {}\n"
                "Size: {}").format(self._coords.shape[0],
                                   self._coord_type,
                                   self._size)
        return info


class Attrs(dict):

    def __setitem__(self, key, value):
        _validate(value)
        super().__setitem__(key, value)


def _validate(value):
    if not isinstance(value, torch.Tensor):
        raise ValueError(
            "{} type must be a PyTorch Tensor!".format(type(value)))
