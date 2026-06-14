import pygame
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Optional, Tuple, Any

from utils.config.game_config import GameTeam, GameState
from utils.config.robot_config import ROBOT_ID
from utils.utils import meters_to_pixels
from visualization.renderer import Renderer

from rules.base.config.action_config import ActionBase
from rules.base.config.observation_config import ObsBaseEnv, ObsBaseRobot, ObsBaseGame
from rules.base.config.robot_config import BASE_ROBOT_TYPE_ACTION
from rules.base.curriculum import *
from rules.base.environment import Environment


class Game(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        curriculum_list: List[CurriculumBase] = [CurriculumBase()]
    ):
        super().__init__()

        # 课程学习
        self.curriculum_list = curriculum_list
        self.curriculum_stage = 0
        self.curriculum_random_env = False
        self.curriculum_random_obstacles = False
        self.curriculum_random_robots = False
        self.set_curriculum(0)
        env_config, obstacle_configs, robot_configs = self.curriculum_list[self.curriculum_stage].random_start(
            if_env=self.curriculum_random_env,
            if_obstacles=self.curriculum_random_obstacles,
            if_robots=self.curriculum_random_robots
        )
        # 随机配置
        self._env_config = env_config
        self._obstacle_configs = obstacle_configs
        self._robot_configs = robot_configs

        # 创建底层环境
        self.env = Environment(self._env_config, self._obstacle_configs, self._robot_configs)
        self.dt = 1 / env_config.fps
        
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

    def set_curriculum(self, stage: int = 0, random_env: bool = False, random_obstacles: bool = False, random_robots: bool = False):
        self.curriculum_stage = stage
        self.curriculum_random_env = random_env
        self.curriculum_random_obstacles = random_obstacles
        self.curriculum_random_robots = random_robots

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
        env_config, obstacle_configs, robot_configs = self.curriculum_list[self.curriculum_stage].random_start(
            if_env=self.curriculum_random_env,
            if_obstacles=self.curriculum_random_obstacles,
            if_robots=self.curriculum_random_robots
        )
        self._env_config = env_config
        self._obstacle_configs = obstacle_configs
        self._robot_configs = robot_configs
        # 重置底层环境
        self.env.reset(self._env_config, self._obstacle_configs, self._robot_configs)
        
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
    
    def step(
            self,
            red_action: Dict[str, ActionBase],
            blue_action: Dict[str, ActionBase] = None,
            control_steps: int = 1
        ):
        """
        执行一步动作
        Args:
            red_asction: 红方行动
            blue_action: 蓝方行动。None 则采用课程对应的自动脚本控制
            control_steps: 步进的帧数
        """
        render_images = []
        for _ in range(control_steps):
            # 记录帧开始时间
            self._frame_start_time = time.perf_counter()

            # 执行环境步进
            if not blue_action:
                blue_action = self.curriculum_list[self.curriculum_stage].get_enemy_action(self.env._remaining_time, self.env.robots)
            self.env.step(red_action, blue_action)
            # print(red_action)
            # print(blue_action)

            # 获取观察
            observation = self._get_obs()
            
            # 判断是否结束
            terminated = self._is_terminated()
            truncated = self._is_truncated()
            
            # 渲染
            if self._render_mode:
                render_images.append(self.render())

            if terminated or truncated:
                break

        # 计算奖励（以红队视角）
        reward = self.curriculum_list[self.curriculum_stage].reward(self.env.robots, red_action)

        # 信息
        info = {
            'game_state': self.env.game_state,
            'remaining_time': self.env._remaining_time,
            'red_hp': sum([robot.hp for robot in self.env.robots.values() if robot.team == GameTeam.RED]),
            'blue_hp': sum([robot.hp for robot in self.env.robots.values() if robot.team == GameTeam.BLUE]),
            'reward': reward,
            'render_images': render_images,
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self) -> ObsBaseGame:
        """获取观察"""
        # 全局状态向量
        env_obs = ObsBaseEnv(
            remaining_time_norm=self.env._remaining_time / self._env_config.game_time_limit,
        )

        # 机器人状态向量
        robots_obs = {}
        for robot_id, robot in self.env.robots.items():
            robots_obs[robot_id] = ObsBaseRobot.from_robot(robot)

        return ObsBaseGame(env_obs, robots_obs)

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
                    (meters_to_pixels(self._env_config.field_width), meters_to_pixels(self._env_config.field_height))
                )
            elif self._render_mode == "rgb_array":
                self._screen = pygame.Surface((meters_to_pixels(self._env_config.field_width), meters_to_pixels(self._env_config.field_height)))
            else:
                raise ValueError(f"Invalid render mode: {self._render_mode}")
        if self._renderer is None:
            self._renderer = Renderer(self._env_config)

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
