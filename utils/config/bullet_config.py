import torch
from config import DEVICE

class SmallBullet():
    """小子弹"""
    DAMAGE: torch.Tensor = torch.tensor(10, dtype=torch.int, device=DEVICE)
    HEAT: torch.Tensor = torch.tensor(10, dtype=torch.int, device=DEVICE)
    PRICE: torch.Tensor = torch.tensor(1, dtype=torch.int, device=DEVICE)
    EXP: torch.Tensor = torch.tensor(1, dtype=torch.int, device=DEVICE)
    PURCHASE_NUM: torch.Tensor = torch.tensor(10, dtype=torch.int, device=DEVICE)

class LargeBullet():
    """大子弹"""
    DAMAGE: torch.Tensor = torch.tensor(100, dtype=torch.int, device=DEVICE)
    HEAT: torch.Tensor = torch.tensor(100, dtype=torch.int, device=DEVICE)
    PRICE: torch.Tensor = torch.tensor(10, dtype=torch.int, device=DEVICE)
    EXP: torch.Tensor = torch.tensor(10, dtype=torch.int, device=DEVICE)
    PURCHASE_NUM: torch.Tensor = torch.tensor(1, dtype=torch.int, device=DEVICE)
