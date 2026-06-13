from dataclasses import dataclass

# 此处通过是否有类型注解来控制是否可以通过初始化来赋值某些参数，用于环境随机初始化

@dataclass
class EnvConfigBase():
    env_name = "solo"

    # 仿真配置
    fps = 60

    # 场地尺寸（米）
    field_width = 5.0
    field_height = 5.0

    # 游戏时间限制（秒）
    game_time_limit = 60
    game_remaining_time : float = 60

    @classmethod
    def field_size(cls):
        return (cls.field_width, cls.field_height)
