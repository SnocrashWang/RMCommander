from utils.game_config import GameTeam
from utils import env_config
from utils import robot_config
from utils.utils import meters_to_pixels
from utils.grid_map import GridMap
import pymunk

from .physics import PhysicsEngine
from .robot import Robot
from .obstacle import Obstacle
from .game import GameStateManager

class Environment:
    def __init__(self):
        # 创建物理引擎
        self.physics_engine = pymunk.Space()
        self.physics_engine.gravity = (0, 0)  # 无重力

        # 创建机器人
        self.robots = []
        self._create_robots()

        # 创建墙壁
        self._create_walls()

        # 创建障碍物
        self.obstacles = []
        self._create_obstacles()

        # 创建中心区域
        self.center_zone_rect = env_config.CENTER_ZONE_RECT

        # 创建游戏状态管理器
        self.game_state_manager = GameStateManager()

        # 创建栅格地图
        self.grid_map = GridMap(
            env_config.FIELD_WIDTH,
            env_config.FIELD_HEIGHT,
            env_config.GRID_CELL_SIZE
        )
        self.grid_map.mark_obstacles(self.obstacles)

    def _create_robots(self):
        pos1 = robot_config.ROBOT1_INIT_POS
        pos2 = robot_config.ROBOT2_INIT_POS
        self.robots.append(Robot(self.physics_engine, pos1, team=GameTeam.RED, env=self))
        self.robots.append(Robot(self.physics_engine, pos2, team=GameTeam.BLUE, env=self))

    def _create_walls(self):
        # Implementation of _create_walls method
        pass

    def _create_obstacles(self):
        for obstacle_config in env_config.OBSTACLES:
            self.obstacles.append(Obstacle(self.physics_engine, obstacle_config))

    def _update_grid_map(self):
        self.grid_map = GridMap(
            env_config.FIELD_WIDTH,
            env_config.FIELD_HEIGHT,
            env_config.GRID_CELL_SIZE
        )
        self.grid_map.mark_obstacles(self.obstacles)

    def reset(self):
        """重置环境"""
        # 销毁现有机器人
        for robot in self.robots:
            robot.destroy(self.physics_engine)
        self.robots.clear()
        # 重置游戏状态
        self.game_state_manager.reset()
        # 创建新机器人
        self._create_robots()

    def step(self, dt):
        """推进环境仿真"""
        # 更新物理引擎
        self.physics_engine.step(dt)

        # 更新机器人状态
        for robot in self.robots:
            robot.update(self.center_zone_rect, env_config.SCALE)

        # 检查中心区域占领情况
        robots_in_zone = {GameTeam.RED: False, GameTeam.BLUE: False}
        for robot in self.robots:
            if robot.in_center_zone and robot.is_alive:
                robots_in_zone[robot.team] = True

        # 更新游戏状态
        self.game_state_manager.update(robots_in_zone, dt)
        
        # 更新网格地图
        self.grid_map.mark_obstacles(self.obstacles)

    def get_game_state(self):
        """获取当前游戏状态"""
        state = {
            "state": self.game_state_manager.state,
            "remaining_time": self.game_state_manager.get_remaining_time(),
            "center_zone_progress": self.game_state_manager.center_zone_progress.copy(),
            "grid_map": self.grid_map,
            "robots": self.robots,
        }
        return state

    def is_game_over(self):
        """检查游戏是否结束"""
        return self.game_state_manager.is_game_over()
