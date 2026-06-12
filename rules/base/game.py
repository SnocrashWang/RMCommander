import pygame
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Optional, Tuple, Any

from utils.config.exp_prop_config import LEVEL_NEED_EXP
from utils.config.game_config import GameTeam, GameState
from utils.config.robot_config import RobotType, ROBOT_ID
from utils.grid_map import world_to_grid
from utils.utils import meters_to_pixels, calc_distance, opposite_team
from visualization.renderer import Renderer

from rules.base.config import env_config as BASE_ENV_CONFIG
from rules.base.config.action_config import ActionBase
from rules.base.config.observation_config import ObsBaseEnv, ObsBaseRobot, ObsBaseGame
from rules.base.config.obstacle_config import OBSTACLE_CONFIGS
from rules.base.config.robot_config import BASE_ROBOT_TYPE_ACTION, BASE_ROBOT_CONFIGS
from rules.base.curriculum import Curriculum
from rules.base.environment import Environment


class Game(gym.Env):
   
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": BASE_ENV_CONFIG.FPS,
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # 课程学习
        self.curriculum = Curriculum()
        obstacle_configs, robot_configs = self.curriculum.random_start(if_obstacles=True, if_robots=True)
        # 随机配置
        self._obstacle_configs = obstacle_configs
        self._robot_configs = robot_configs

        # 创建底层环境
        self.env = Environment(self._obstacle_configs, self._robot_configs)
        self.dt = 1 / BASE_ENV_CONFIG.FPS
        
        # 渲染
        self._render_mode = render_mode
        self._screen = None
        self._frame_start_time = time.perf_counter()
        self._renderer = None
        if self._render_mode:
            self._init_render()
        
        # 定义动作空间
        self._setup_action_space()
        
        # 定义观察空间
        self._setup_observation_space()

        # 状态记录
        self._last_observation = self._get_obs()
        self._last_action = None
    
    def _setup_action_space(self):
        self.action_space = spaces.Dict({
            ROBOT_ID[robot.team][robot.robot_type]: BASE_ROBOT_TYPE_ACTION[robot.robot_type].get_space()
            for robot in self.env.robots.values()
        })
    
    def _setup_observation_space(self):
        """设置观察空间"""
        self.observation_space = ObsBaseGame.get_space(self.env.robots.keys())
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = {}):
        """重置环境"""
        super().reset(seed=seed)

        # 重新课程随机
        obstacle_configs, robot_configs = self.curriculum.random_start(if_obstacles=True, if_robots=True)
        self._obstacle_configs = obstacle_configs
        self._robot_configs = robot_configs

        # 重置底层环境
        self.env.reset(self._obstacle_configs, self._robot_configs)
        
        # 渲染
        if self._render_mode:
            render_image = self.render()
        else:
            render_image = None
        
        # 获取初始观察
        observation = self._get_obs()
        self._last_observation = self._get_obs()
        self._last_action = None
        info = {
            'render_images': [render_image],
        }
        
        return observation, info
    
    def step(self, red_action: Dict[str, ActionBase], blue_action: Dict[str, ActionBase], control_steps: int = 1):
        """执行一步动作"""
        reward = 0
        render_images = []
        for _ in range(control_steps):
            # 记录帧开始时间
            self._frame_start_time = time.perf_counter()

            # 执行环境步进
            self.env.step(red_action, blue_action)
        
            # 获取观察
            observation = self._get_obs()
            
            # 计算奖励（以红队视角）
            reward += self._get_reward(GameTeam.RED, red_action)
            
            # 判断是否结束
            terminated = self._is_terminated()
            truncated = self._is_truncated()
            
            # 渲染
            if self._render_mode:
                render_images.append(self.render())

            if terminated or truncated:
                break
        
        # 信息
        info = {
            'game_state': self.env.game_state,
            'remaining_time': self.env._remaining_time,
            'red_hp': {robot_id: robot.hp for robot_id, robot in self.env.robots.items() if robot.team == GameTeam.RED},
            'blue_hp': {robot_id: robot.hp for robot_id, robot in self.env.robots.items() if robot.team == GameTeam.BLUE},
            'render_images': render_images,
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self) -> ObsBaseGame:
        """获取观察"""
        # 全局状态向量
        env_obs = ObsBaseEnv(
            remaining_time_norm=self.env._remaining_time / BASE_ENV_CONFIG.GAME_TIME_LIMIT,
        )

        # 机器人状态向量
        robots_obs = {}
        for robot_id, robot in self.env.robots.items():
            robots_obs[robot_id] = ObsBaseRobot.from_robot(robot)

        return ObsBaseGame(env_obs, robots_obs)
    
    def _get_reward(self, team: GameTeam, action: Dict[str, ActionBase]) -> float:
        return 0

    def _is_terminated(self) -> bool:
        """判断是否自然结束"""
        return self.env._remaining_time <= 0
    
    def _is_truncated(self) -> bool:
        """判断是否被截断"""
        return self.env.game_state in [GameState.RED_TEAM_WIN, GameState.BLUE_TEAM_WIN, GameState.DRAW]
    
    def _init_render(self):
        if self._screen is None:
            pygame.init()
            if self._render_mode == "human":
                pygame.display.init()
                self._screen = pygame.display.set_mode(
                    (meters_to_pixels(BASE_ENV_CONFIG.FIELD_WIDTH), meters_to_pixels(BASE_ENV_CONFIG.FIELD_HEIGHT))
                )
            elif self._render_mode == "rgb_array":
                self._screen = pygame.Surface((meters_to_pixels(BASE_ENV_CONFIG.FIELD_WIDTH), meters_to_pixels(BASE_ENV_CONFIG.FIELD_HEIGHT)))
            else:
                raise ValueError(f"Invalid render mode: {self._render_mode}")
        if self._renderer is None:
            self._renderer = Renderer(BASE_ENV_CONFIG)

    def set_render(self, control_state):
        self._renderer.control_state = control_state

    def render(self):
        """渲染环境"""
        if self._render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        
        self._init_render()

        screen = self._renderer.render(self.env)
        self._screen.blit(screen, (0, 0))
        if self._render_mode == "human":
            time_cost = time.perf_counter() - self._frame_start_time
            wait_time = max(0, self.dt - time_cost)
            if wait_time > 0:
                time.sleep(wait_time)
            pygame.display.flip()
            return None
        elif self._render_mode == "rgb_array":
            # 返回RGB数组
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._screen)), axes=(1, 0, 2)
            )
    
    def close(self):
        """关闭环境"""
        if hasattr(self, '_renderer') and self._renderer is not None:
            # 关闭pygame显示
            import pygame
            pygame.display.quit()
            pygame.quit()
