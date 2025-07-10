import torch
from utils.config.game_config import GameType

DEVICE = (
    "cuda" if torch.cuda.is_available() else "cpu"
    # "cpu"
    # "cuda"
)

CURRENT_GAME = (
    GameType.BASE
    # GameType.RMUL
    # GameType.RMUC
)
