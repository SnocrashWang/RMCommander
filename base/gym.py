import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

from utils.config.game_config import GameTeam, GameState
from utils.config.robot_config import RobotType
from utils.utils import meters_to_pixels
from visualization.renderer import Renderer

from base.config import env_config
from base.config.robot_config import BASE_ROBOT_CONFIGS, BASE_ROBOT_TYPE_LIST
from base.environment import Environment

@dataclass
class Action():
    navigation: Tuple[float, float] = None
    attack: bool = False
    target: RobotType = None

class RoboMasterGym(gym.Env):
   
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": env_config.FPS,
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # 创建底层环境
        self.env = Environment(
            env_config=env_config,
            obstacle_configs=env_config.OBSTACLES,
            robot_configs=BASE_ROBOT_CONFIGS
        )
        
        # 渲染
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.renderer = None
        self._init_render()
        
        # 定义动作空间
        self._setup_action_space()
        
        # 定义观察空间
        self._setup_observation_space()
    
    def _setup_action_space(self):
        """设置动作空间"""
        # 为每个机器人定义动作空间
        robot_action_spaces = {}
        
        for robot_id, robot in self.env.robots.items():
            # 导航动作：x, y坐标
            navigation_space = spaces.Box(
                low=np.array([0.0, 0.0], dtype=np.float32),
                high=np.array([env_config.FIELD_WIDTH, env_config.FIELD_HEIGHT], dtype=np.float32),
                dtype=np.float32
            )
            
            # 攻击动作：是否攻击
            attack_space = spaces.Discrete(2)  # 0: 不攻击, 1: 攻击
            
            # 目标动作：攻击目标类型
            target_space = spaces.Discrete(len(RobotType))  # 所有机器人类型
            
            # 组合动作空间
            robot_action_spaces[robot_id] = spaces.Dict({
                'navigation': navigation_space,
                'attack': attack_space,
                'target': target_space
            })
        
        self.action_space = spaces.Dict(robot_action_spaces)
    
    def _setup_observation_space(self):
        """设置观察空间"""
        # 游戏状态：剩余时间
        game_state_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # 机器人状态：位置(2) + 属性(2) + 等级(1) + 经验(1) + 血量(1) + 热量(1) = 8维
        robot_state_space = spaces.Box(
            low=np.array([0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 3, 3, 10, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # 计算机器人数量
        red_robots = [r for r in self.env.robots.values() if r.team == GameTeam.RED]
        blue_robots = [r for r in self.env.robots.values() if r.team == GameTeam.BLUE]
        
        # 组合观察空间
        observation_low = np.concatenate([
            game_state_space.low,  # 游戏状态
            np.tile(robot_state_space.low, len(red_robots)),  # 红队机器人
            np.tile(robot_state_space.low, len(blue_robots))  # 蓝队机器人
        ])
        
        observation_high = np.concatenate([
            game_state_space.high,  # 游戏状态
            np.tile(robot_state_space.high, len(red_robots)),  # 红队机器人
            np.tile(robot_state_space.high, len(blue_robots))  # 蓝队机器人
        ])
        
        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high,
            dtype=np.float32
        )
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置底层环境
        self.env.reset()
        
        # 获取初始观察
        observation = self._get_obs()
        info = {}
        
        if self.render_mode == "human":
            self.render()
        
        return observation, info
    
    def step(self, action: Dict[str, Dict[str, Any]]):
        """执行一步动作"""
        # 转换动作格式
        red_action = {}
        blue_action = {}
        
        for robot_id, robot_action in action.items():
            robot = self.env.get_robot(robot_id)
            if robot is None:
                continue
                
            # 转换动作格式
            converted_action = Action()
            
            # 导航动作
            if 'navigation' in robot_action:
                converted_action.navigation = tuple(robot_action['navigation'])
            
            # 攻击动作
            if 'attack' in robot_action:
                converted_action.attack = bool(robot_action['attack'])
            
            # 目标动作
            if 'target' in robot_action:
                converted_action.target = RobotType(robot_action['target'])
            
            # 按队伍分类
            if robot.team == GameTeam.RED:
                red_action[robot_id] = converted_action
            else:
                blue_action[robot_id] = converted_action
        
        # 执行环境步进
        self.env.step(self.env.dt, red_action, blue_action)
        
        # 获取观察
        observation = self._get_obs()
        
        # 计算奖励（以红队视角）
        reward = self.env.calculate_reward(GameTeam.RED, red_action)
        
        # 判断是否结束
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        
        # 信息
        info = {
            'game_state': self.env.game_state_manager.state,
            'remaining_time': self.env.game_state_manager.get_remaining_time(),
            'red_hp': {robot_id: robot.hp for robot_id, robot in self.env.robots.items() if robot.team == GameTeam.RED},
            'blue_hp': {robot_id: robot.hp for robot_id, robot in self.env.robots.items() if robot.team == GameTeam.BLUE},
        }
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """获取观察"""
        # 使用底层环境的编码方法
        return self.env._get_team_state(GameTeam.RED)
    
    def _is_terminated(self) -> bool:
        """判断是否自然结束"""
        return self.env.game_state_manager.get_remaining_time() <= 0
    
    def _is_truncated(self) -> bool:
        """判断是否被截断"""
        return self.env.game_state_manager.state in [GameState.RED_TEAM_WIN, GameState.BLUE_TEAM_WIN, GameState.DRAW]
    
    def _init_render(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (meters_to_pixels(env_config.FIELD_WIDTH), meters_to_pixels(env_config.FIELD_HEIGHT))
                )
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface((meters_to_pixels(env_config.FIELD_WIDTH), meters_to_pixels(env_config.FIELD_HEIGHT)))
            else:
                raise ValueError(f"Invalid render mode: {self.render_mode}")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.renderer is None:
            self.renderer = Renderer(env_config)

    def render(self):
        """渲染环境"""
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        
        self._init_render()

        screen = self.renderer.render(self.env)
        if self.render_mode == "human":
            # 使用自定义渲染器
            self.screen.blit(screen, (0, 0))
            pygame.display.flip()
            return None
        elif self.render_mode == "rgb_array":
            # 返回RGB数组
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    
    def close(self):
        """关闭环境"""
        if hasattr(self, 'renderer') and self.renderer is not None:
            # 关闭pygame显示
            import pygame
            pygame.display.quit()
            pygame.quit()
