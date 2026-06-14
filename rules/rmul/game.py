import pygame
import time
import numpy as np
import gymnasium as gym
from copy import deepcopy
from gymnasium import spaces
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any

from rules.base.game import Game
from utils.config.exp_prop_config import LEVEL_NEED_EXP
from utils.config.game_config import GameTeam, GameState
from utils.config.robot_config import RobotType, ROBOT_ID
from utils.grid_map import world_to_grid
from utils.utils import meters_to_pixels, calc_distance, opposite_team, timer
from visualization.renderer import Renderer

from rules.rmul.config.env_config import EnvConfigRMUL
from rules.rmul.config.action_config import ActionRMUL
from rules.rmul.config.observation_config import ObsRMULEnv, ObsRMULRobot, ObsRMULGame, RMUL_ROBOT_TYPE_OBS
from rules.rmul.config.robot_config import RMUL_ROBOT_TYPE_ACTION, RMUL_ROBOT_CONFIGS
from rules.rmul.environment import EnvironmentRMUL


class GameRMUL(gym.Env):
   
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        robot_type_obs: Dict = RMUL_ROBOT_TYPE_OBS,
    ):
        super().__init__()
        self._robot_type_obs = robot_type_obs

        # 创建底层环境
        self.env = EnvironmentRMUL()
        self.dt = 1 / EnvConfigRMUL.fps
        
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

        self._time_stats = defaultdict(list)
    
    def _setup_action_space(self):
        self.action_space = self.get_action_space()
    
    def _setup_observation_space(self):
        """设置观察空间"""
        self.observation_space = self.get_observation_space(self._robot_type_obs)

    @classmethod
    def get_action_space(cls, robot_configs=RMUL_ROBOT_CONFIGS):
        return spaces.Dict({
            ROBOT_ID[robot_config.team][robot_config.robot_type]: RMUL_ROBOT_TYPE_ACTION[robot_config.robot_type].get_space()
            for robot_config in robot_configs
        })

    @classmethod
    def get_observation_space(
        cls,
        robot_type_obs=RMUL_ROBOT_TYPE_OBS,
        robot_configs=RMUL_ROBOT_CONFIGS,
        team: GameTeam = GameTeam.RED,
    ):
        return ObsRMULGame.get_space(robot_configs, robot_type_obs, team)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = {}):
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置底层环境
        self.env.reset()
        
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
            'game_state': self.env.game_state,
            'remaining_time': self.env._remaining_time,
            'victory_progress': self.env._victory_progress,
            'economics': self.env._economics,
            'robots': deepcopy(self.env.robots),
        }
        
        return observation, info

    def step(self, red_action: Dict[str, ActionRMUL], blue_action: Dict[str, ActionRMUL], control_steps: int = 1):
        """执行一步动作"""
        reward = 0
        render_images = []
        for _ in range(control_steps):
            # 记录帧开始时间
            self._frame_start_time = time.perf_counter()

            # 执行环境步进
            with timer(self._time_stats, 'env_step'):
                self.env.step(red_action, blue_action)
            
            # 获取观察
            with timer(self._time_stats, 'get_obs'):
                observation = self._get_obs()
            
            # 计算奖励（以红队视角）
            with timer(self._time_stats, 'get_reward'):
                reward += self._get_reward(GameTeam.RED, red_action)
            
            # 判断是否结束
            terminated = self._is_terminated()
            truncated = self._is_truncated()

            # 渲染
            with timer(self._time_stats, 'render'):
                if self._render_mode:
                    render_images.append(self.render())

            if terminated or truncated:
                break
        
        # 信息
        with timer(self._time_stats, 'make_info'):
            info = {
                'render_images': render_images,
                'game_state': self.env.game_state,
                'remaining_time': self.env._remaining_time,
                'victory_progress': self.env._victory_progress,
                'economics': self.env._economics,
                'robots': deepcopy(self.env.robots),
            }

        # print("\n性能统计:")
        # for key, times in self._time_stats.items():
        #     if times:  # 确保有数据
        #         avg_time = sum(times) / len(times)
        #         print(f"{key}: {avg_time:.6f}s")
        # print("=" * 50)
        # self._time_stats.clear()

        return observation, reward, terminated, truncated, info
    
    def _get_obs(self) -> ObsRMULGame:
        """获取观察"""
        # 全局状态向量
        env_obs = ObsRMULEnv(
            remaining_time_norm=self.env._remaining_time / EnvConfigRMUL.game_time_limit,
            victory_progress_red_norm=self.env._victory_progress[GameTeam.RED] / EnvConfigRMUL.occupation_target,
            victory_progress_blue_norm=self.env._victory_progress[GameTeam.BLUE] / EnvConfigRMUL.occupation_target,
        )

        # 机器人状态向量
        robots_env = {}
        for robot_id, robot in self.env.robots.items():
            obs_cls = ObsRMULGame._get_storage_robot_obs_cls(self._robot_type_obs, robot.robot_type)
            robots_env[robot_id] = obs_cls.from_robot(robot)
        
        return ObsRMULGame(env_obs, robots_env, self._robot_type_obs)
    
    def _get_reward(self, team: GameTeam, action: Dict[str, ActionRMUL]) -> float:
        """获取奖励"""
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
                    (meters_to_pixels(EnvConfigRMUL.field_width), meters_to_pixels(EnvConfigRMUL.field_height))
                )
            elif self._render_mode == "rgb_array":
                self._screen = pygame.Surface((meters_to_pixels(EnvConfigRMUL.field_width), meters_to_pixels(EnvConfigRMUL.field_height)))
            else:
                raise ValueError(f"Invalid render mode: {self._render_mode}")
        if self._renderer is None:
            self._renderer = Renderer(EnvConfigRMUL)

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
            # 使用自定义渲染器
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
