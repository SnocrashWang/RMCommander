from utils.game_config import GameTeam
from utils import env_config
from utils import robot_config
from utils.utils import meters_to_pixels
from utils.grid_map import GridMap

from .physics import PhysicsEngine
from .robot import Robot
from .obstacle import Obstacle
from .game import GameStateManager

class Environment:
    def __init__(self):
        self.physics_engine = PhysicsEngine()
        self.state_manager = GameStateManager()
        self.robots = []  # 用列表存储所有机器人
        self.obstacles = []

        # 中心区域矩形（像素坐标）
        self.center_zone_rect = self._create_center_zone_rect()

        self.grid_map = GridMap(
            env_config.FIELD_WIDTH,
            env_config.FIELD_HEIGHT,
            cell_size=0.1  # 可调整
        )

        self.setup_environment()

    def _create_center_zone_rect(self):
        """创建中心区域矩形"""
        center_x = env_config.FIELD_WIDTH / 2
        center_y = env_config.FIELD_HEIGHT / 2
        size = env_config.CENTER_ZONE_SIZE

        return (
            meters_to_pixels(center_x - size/2, env_config.SCALE),
            meters_to_pixels(center_y - size/2, env_config.SCALE),
            meters_to_pixels(size, env_config.SCALE),
            meters_to_pixels(size, env_config.SCALE)
        )

    def setup_environment(self):
        """设置环境"""
        self.robots.clear()
        pos1 = robot_config.ROBOT1_INIT_POS
        pos2 = robot_config.ROBOT2_INIT_POS
        self.robots.append(Robot(self.physics_engine, pos1, team=GameTeam.RED, env=self))
        self.robots.append(Robot(self.physics_engine, pos2, team=GameTeam.BLUE, env=self))
        self.obstacles.clear()
        for obstacle_config in env_config.OBSTACLES:
            self.obstacles.append(Obstacle(self.physics_engine, obstacle_config))

    def _update_grid_map(self):
        self.grid_map = GridMap(
            env_config.FIELD_WIDTH,
            env_config.FIELD_HEIGHT,
            cell_size=0.1
        )
        self.grid_map.mark_obstacles(self.obstacles)

    def reset(self):
        """重置环境"""
        # 销毁现有机器人
        for robot in self.robots:
            robot.destroy(self.physics_engine)
        self.robots.clear()
        # 重置游戏状态
        self.state_manager.reset()
        # 创建新机器人
        pos1 = robot_config.ROBOT1_INIT_POS
        pos2 = robot_config.ROBOT2_INIT_POS
        self.robots.append(Robot(self.physics_engine, pos1, team=GameTeam.RED, env=self))
        self.robots.append(Robot(self.physics_engine, pos2, team=GameTeam.BLUE, env=self))

    def step(self, dt):
        """推进环境仿真"""
        # 更新物理引擎
        self.physics_engine.step(dt)
        for robot in self.robots:
            robot.update(self.center_zone_rect, env_config.SCALE)
        # 更新游戏状态（假设第一个和第二个robot分别为1、2队）
        self.state_manager.update(
            self.robots[0].in_center_zone,
            self.robots[1].in_center_zone,
            dt
        )
        self._update_grid_map()

    def get_game_state(self):
        """获取当前游戏状态"""
        state = {
            "state": self.state_manager.state,
            "progress": self.state_manager.center_zone_progress.copy(),
            "robots": [
                {
                    "pos": robot.get_position(),
                    "angle": robot.angle,
                    "team": robot.team
                }
                for robot in self.robots
            ],
            "grid_map": self.grid_map,
            "remaining_time": self.state_manager.get_remaining_time()
        }
        return state

    def is_game_over(self):
        """检查游戏是否结束"""
        return self.state_manager.is_game_over()
