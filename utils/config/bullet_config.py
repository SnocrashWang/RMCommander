from dataclasses import dataclass

@dataclass
class SmallBullet():
    """小子弹"""
    DAMAGE: int = 10
    HEAT: int = 10
    PRICE: int = 1
    EXP: int = 1
    PURCHASE_NUM: int = 50

@dataclass
class LargeBullet():
    """大子弹"""
    DAMAGE: int = 100
    HEAT: int = 100
    PRICE: int = 10
    EXP: int = 10
    PURCHASE_NUM: int = 5
