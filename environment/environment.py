import utils.global_config as global_config
from .physics import PhysicsEngine
from .robot import Robot
from .obstacle import Obstacle
from .game import GameStateManager
from utils.constants import GameState
from utils.utils import meters_to_pixels
from utils.grid_map import GridMap

class Environment:
    def __init__(self):
        self.physics_engine = PhysicsEngine()
        self.state_manager = GameStateManager()
        self.robot1 = None
        self.robot2 = None
        self.obstacles = []
        
        # 中心区域矩形（像素坐标）
        self.center_zone_rect = self._create_center_zone_rect()
        
        self.grid_map = GridMap(
            global_config.FIELD_WIDTH,
            global_config.FIELD_HEIGHT,
            cell_size=0.1  # 可调整
        )
        
        self.setup_environment()
    
    def _create_center_zone_rect(self):
        """创建中心区域矩形"""
        center_x = global_config.FIELD_WIDTH / 2
        center_y = global_config.FIELD_HEIGHT / 2
        size = global_config.CENTER_ZONE_SIZE
        
        return (
            meters_to_pixels(center_x - size/2, global_config.SCALE),
            meters_to_pixels(center_y - size/2, global_config.SCALE),
            meters_to_pixels(size, global_config.SCALE),
            meters_to_pixels(size, global_config.SCALE)
        )
    
    def setup_environment(self):
        """设置环境"""
        # 创建机器人
        pos1 = (1.0, 1.0)
        pos2 = (global_config.FIELD_WIDTH-1.0, global_config.FIELD_HEIGHT-1.0)
        self.robot1 = Robot(self.physics_engine, pos1, team=1, env=self)
        self.robot2 = Robot(self.physics_engine, pos2, team=2, env=self)
        
        # 创建障碍物
        for obstacle_config in global_config.OBSTACLES:
            self.obstacles.append(Obstacle(self.physics_engine, obstacle_config))

    def _update_grid_map(self):
        self.grid_map = GridMap(
            global_config.FIELD_WIDTH,
            global_config.FIELD_HEIGHT,
            cell_size=0.1
        )
        self.grid_map.mark_obstacles(self.obstacles)

    def reset(self):
        """重置环境"""
        # 销毁现有机器人
        if self.robot1:
            self.robot1.destroy(self.physics_engine)
        if self.robot2:
            self.robot2.destroy(self.physics_engine)
        
        # 重置游戏状态
        self.state_manager.reset()
        
        # 创建新机器人
        pos1 = (1.0, 1.0)
        pos2 = (global_config.FIELD_WIDTH-1.0, global_config.FIELD_HEIGHT-1.0)
        self.robot1 = Robot(self.physics_engine, pos1, team=1, env=self)
        self.robot2 = Robot(self.physics_engine, pos2, team=2, env=self)
    
    def step(self, dt):
        """推进环境仿真"""
        # 更新物理引擎
        self.physics_engine.step(dt)
        
        # 更新机器人状态
        self.robot1.update(self.center_zone_rect, global_config.SCALE)
        self.robot2.update(self.center_zone_rect, global_config.SCALE)
        
        # 更新游戏状态
        self.state_manager.update(
            self.robot1.in_center_zone,
            self.robot2.in_center_zone,
            dt
        )
        self._update_grid_map()

    def get_game_state(self):
        """获取当前游戏状态"""
        return {
            "state": self.state_manager.state,
            "progress": self.state_manager.center_zone_progress.copy(),
            "robot1_pos": self.robot1.get_position(),
            "robot2_pos": self.robot2.get_position(),
            "robot1_angle": self.robot1.angle,
            "robot2_angle": self.robot2.angle
        }
    
    def is_game_over(self):
        """检查游戏是否结束"""
        return self.state_manager.is_game_over()