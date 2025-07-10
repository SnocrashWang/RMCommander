import torch
from enum import Enum

from config import DEVICE

LEVEL_NEED_EXP = torch.tensor([
    0,
    400,
    800,
    1200,
    1600,
    2000,
    2400,
    2800,
    3200,
    4000,
], dtype=torch.int, device=DEVICE)

class CHASSIS_PROPERTY_TYPE(Enum):
    DEFAULT = 0
    POWER = 1
    HP = 2

CHASSIS_PROPERTY_HERO = torch.tensor([
    [
        [150, 50] for _ in range(10)
    ],
    [
        [200, 70],
        [225, 75],
        [250, 80],
        [275, 85],
        [300, 90],
        [325, 95],
        [350, 100],
        [375, 105],
        [400, 110],
        [500, 120],
    ],
    [
        [250, 55],
        [275, 60],
        [300, 65],
        [325, 70],
        [350, 75],
        [375, 80],
        [400, 85],
        [425, 90],
        [450, 100],
        [500, 120],
    ],
], dtype=torch.int, device=DEVICE)

CHASSIS_PROPERTY_STANDARD = torch.tensor([
    [
        [100, 40] for _ in range(10)
    ],
    [
        [150, 60],
        [175, 65],
        [200, 70],
        [225, 75],
        [250, 80],
        [275, 85],
        [300, 90],
        [325, 95],
        [350, 100],
        [400, 100],
    ],
    [
        [200, 45],
        [225, 50],
        [250, 55],
        [275, 60],
        [300, 65],
        [325, 70],
        [350, 75],
        [375, 80],
        [400, 90],
        [400, 100],
    ],
], dtype=torch.int, device=DEVICE)

class GIMBAL_PROPERTY_TYPE(Enum):
    DEFAULT = 0
    HEAT = 1
    COOL_DOWN = 2

GIMBAL_PROPERTY_17 = torch.tensor([
    [
        [40, 100] for _ in range(10)
    ],
    [
        [200, 10],
        [250, 15],
        [300, 20],
        [350, 25],
        [400, 30],
        [450, 35],
        [500, 40],
        [550, 45],
        [600, 50],
        [650, 60],
    ],
    [
        [50, 40],
        [85, 45],
        [120, 50],
        [155, 55],
        [190, 60],
        [225, 65],
        [260, 70],
        [295, 75],
        [330, 80],
        [400, 80],
    ],
], dtype=torch.int, device=DEVICE)

GIMBAL_PROPERTY_42 = torch.tensor([
    [
        [100, 40],
        [140, 48],
        [180, 56],
        [220, 64],
        [260, 72],
        [300, 80],
        [340, 88],
        [380, 96],
        [420, 104],
        [500, 120],
    ],
], dtype=torch.int, device=DEVICE)
