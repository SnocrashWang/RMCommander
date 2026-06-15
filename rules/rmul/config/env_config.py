from dataclasses import dataclass

# 此处通过是否有类型注解来控制是否可以通过初始化来赋值某些参数，用于环境随机初始化

@dataclass
class EnvConfigRMUL():
    env_name = "RMUL"

    # 仿真配置
    fps = 60

    # 场地尺寸（米）
    field_width = 12.0
    field_height = 8.0

    # 游戏时间限制（秒）
    game_time_limit = 300
    game_remaining_time : float = 300

    # 占领目标进度
    victory_target = 200
    victory_progress_red : float = 0
    victory_progress_blue : float = 0

    # 初始经济
    economics_red : int = 0
    economics_blue : int = 0

    @classmethod
    def field_size(cls):
        return (cls.field_width, cls.field_height)
