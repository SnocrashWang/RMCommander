import pymunk
import time
from typing import List, Dict, Optional

from utils import env_config
from utils.game_config import GameTeam
from utils.grid_map import GridMap
from utils.robot_config import RobotConfig, DEFAULT_ROBOT_CONFIGS
from utils.utils import meters_to_pixels

from .robot import Robot
from .obstacle import Obstacle
from .game import RMULGameStateManager

class Environment:
    def __init__(self, robot_configs: Optional[Dict[str, RobotConfig]] = None):
        # 创建物理引擎
        self.physics_engine = pymunk.Space()
        self.physics_engine.gravity = (0, 0)  # 无重力

        # 创建中心区域
        self.center_zone_rect = env_config.CENTER_ZONE_RECT

        # 创建障碍物
        self.obstacles = []
        self._create_obstacles()

        # 创建机器人
        self.robots: List[Robot] = []
        self.robot_configs = robot_configs or DEFAULT_ROBOT_CONFIGS
        self._create_robots()
        
        # 为每个机器人创建网格地图
        self._init_robot_grid_maps()

        # 创建游戏状态管理器
        self.game_state_manager = RMULGameStateManager()

    def _create_robots(self):
        """根据配置创建机器人"""
        for robot_id, config in self.robot_configs.items():
            robot = Robot(
                self.physics_engine,
                id=robot_id,
                team=config.team,
                init_pos=config.init_pos,
                chassis_property_type=config.chassis_property_type,
                gimbal_property_type=config.gimbal_property_type,
                forward_speed_efficiency=config.forward_speed_efficiency,
                rotation_speed_efficiency=config.rotation_speed_efficiency,
                radius=config.radius
            )
            self.robots.append(robot)

    def _init_robot_grid_maps(self):
        """初始化所有机器人的网格地图"""
        for robot in self.robots:
            grid_map = GridMap(
                width=env_config.FIELD_WIDTH,  # 场地宽度
                height=env_config.FIELD_HEIGHT,  # 场地高度
                cell_size=env_config.GRID_CELL_SIZE,  # 网格大小
                robot_radius=robot.radius
            )
            # 标记所有障碍物
            grid_map.mark_obstacles(self.obstacles)
            # 设置机器人的网格地图
            robot.set_grid_map(grid_map)

    # def update_robot_grid_maps(self):
    #     """更新所有机器人的网格地图"""
    #     for robot in self.robots:
    #         if robot.grid_map:
    #             robot.grid_map.clear()
    #             # 标记所有障碍物
    #             robot.grid_map.mark_obstacles(self.obstacles)

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
            robot.step(dt)

        # 检查中心区域占领情况
        robots_in_zone = {GameTeam.RED: False, GameTeam.BLUE: False}
        for robot in self.robots:
            if robot.is_alive:
                # 检查是否在中心区域
                pos = robot.body.position
                pixel_x = meters_to_pixels(pos.x, env_config.SCALE)
                pixel_y = meters_to_pixels(pos.y, env_config.SCALE)
                if self.center_zone_rect.collidepoint(pixel_x, pixel_y):
                    robots_in_zone[robot.team] = True

        # 更新游戏状态
        self.game_state_manager.update(robots_in_zone, dt)


    def get_game_state(self):
        """获取当前游戏状态"""
        state = {
            "game_state": self.game_state_manager,
            "robots": self.robots,
        }
        return state
