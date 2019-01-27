import torch

from loc_utils.bboxes import BBoxes
from loc_utils.coord_types import CoordType


def run_demo():
    m, n = 10, 4
    coords = torch.randint(high=200, size=(m, n))

    # Coordinate conversion
    print(coords)
    bboxes = BBoxes(coords, CoordType.XYXY, (1000, 1000))
    bboxes = bboxes.convert(CoordType.XYWH)
    print(bboxes.coords)


def main():
    run_demo()


if __name__ == "__main__":
    main()
