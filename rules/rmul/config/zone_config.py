from rules.rmul.config.env_config import EnvConfigRMUL

from utils.buff import Buff, ForbiddenZone
from utils.zone import Zone, make_opposite_zone
from utils.config.game_config import GameTeam

width = EnvConfigRMUL().field_width
height = EnvConfigRMUL().field_height

# 启动区
red_boot_zone = Zone(
    GameTeam.RED,
    [
        (0.0, 0.0), # 左上
        (1.5, 0),   # 右上
        (1.5, 2.0), # 右下
        (0.0, 2.0), # 左下
    ],
    True,
    Buff(name="boot", healing=0.25),
    ForbiddenZone(name="boot")
)
blue_boot_zone = make_opposite_zone(red_boot_zone, EnvConfigRMUL.field_size())

# 中心增益区
center_zone = Zone(
    None,
    [
        (width / 2 - 2.0 / 2, height / 2 - 2.0 / 2),    # 左上
        (width / 2 + 2.0 / 2, height / 2 - 2.0 / 2),    # 右上
        (width / 2 + 2.0 / 2, height / 2 + 2.0 / 2),    # 右下
        (width / 2 - 2.0 / 2, height / 2 + 2.0 / 2),    # 左下
    ],
    False,
    Buff(name="center"),
    None
)

RMUL_ZONES = {
    "red_boot": red_boot_zone,
    "blue_boot": blue_boot_zone,
    "center": center_zone
}
