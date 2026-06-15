import pygame
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any

from utils.config.game_config import GameTeam, GameState
from utils.config.robot_config import ROBOT_ID
from utils.observation import linear_norm
from utils.utils import meters_to_pixels, timer
from visualization.renderer import Renderer

from rules.rmul.config.env_config import EnvConfigRMUL
from rules.rmul.config.action_config import ActionRMUL
from rules.rmul.config.observation_config import ObsRMULEnv, ObsRMULRobot, ObsRMULGame, RMUL_ROBOT_TYPE_OBS
from rules.rmul.config.robot_config import RMUL_ROBOT_TYPE_ACTION, RMUL_ROBOT_CONFIGS
from rules.rmul.curriculum import *
from rules.rmul.environment import EnvironmentRMUL


class GameRMUL(gym.Env):
   
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        curriculum_list: List[Dict[type[CurriculumRMUL], Tuple[float, Tuple[bool, bool, bool]]]] = [{CurriculumRMUL: (1.0, (False, False, False))}],
        curriculum_stage: int = 0,
        robot_type_obs: Dict = RMUL_ROBOT_TYPE_OBS,
    ):
        super().__init__()

        # 课程学习
        self._curriculum_list = curriculum_list
        self.set_curriculum(curriculum_stage)
        env_config, obstacle_configs, robot_configs = self._curriculum.random_start(
            if_env=self._curriculum_random_env,
            if_obstacles=self._curriculum_random_obstacles,
            if_robots=self._curriculum_random_robots
        )
        # 随机配置
        self._env_config = env_config
        self._obstacle_configs = obstacle_configs
        self._robot_configs = robot_configs

        # 动作空间和观测空间
        self._robot_type_obs = robot_type_obs
        self.action_space = self.get_action_space(self._robot_configs)
        self.observation_space = self.get_observation_space(self._robot_type_obs, self._robot_configs)

        # 创建底层环境
        self.env = EnvironmentRMUL(self._env_config, self._obstacle_configs, self._robot_configs)
        self.dt = 1 / env_config.fps

        # 渲染
        self._render_mode = render_mode
        self._screen = None
        self._frame_start_time = time.perf_counter()
        self._renderer = None
        if self._render_mode:
            self._init_render()

        # 状态记录
        self._last_observation = self._get_obs()
        self._last_action = None

        self._time_stats = defaultdict(list)

    def set_curriculum(self, curriculum_stage: int):
        self._curriculum = random.choices(
            list(self._curriculum_list[curriculum_stage].keys()),
            weights=[v[0] for v in self._curriculum_list[curriculum_stage].values()]
        )[0]()
        random_env, random_obstacles, random_robots = self._curriculum_list[curriculum_stage][type(self._curriculum)][1]
        self._curriculum_random_env = random_env
        self._curriculum_random_obstacles = random_obstacles
        self._curriculum_random_robots = random_robots

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
        
        # 重新课程随机
        env_config, obstacle_configs, robot_configs = self._curriculum.random_start(
            if_env=self._curriculum_random_env,
            if_obstacles=self._curriculum_random_obstacles,
            if_robots=self._curriculum_random_robots
        )
        self._env_config = env_config
        self._obstacle_configs = obstacle_configs
        self._robot_configs = robot_configs

        # 重置底层环境
        self.env.reset(self._env_config, self._obstacle_configs, self._robot_configs)
        self.action_space = self.get_action_space(self._robot_configs)
        self.observation_space = self.get_observation_space(self._robot_type_obs, self._robot_configs)

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
            red_action: Dict[str, ActionRMUL],
            blue_action: Dict[str, ActionRMUL] = None,
            control_steps: int = 1
        ):
        """执行一步动作"""
        render_images = []
        for _ in range(control_steps):
            # 记录帧开始时间
            self._frame_start_time = time.perf_counter()

            # 执行环境步进
            with timer(self._time_stats, 'env_step'):
                if not blue_action:
                    blue_action = self._curriculum.get_enemy_action(self.env._remaining_time, self.env.robots)
                self.env.step(red_action, blue_action)

            # 获取观察
            with timer(self._time_stats, 'get_obs'):
                observation = self._get_obs()

            # 判断是否结束
            terminated = self._is_terminated()
            truncated = self._is_truncated()

            # 渲染
            with timer(self._time_stats, 'render'):
                if self._render_mode:
                    render_images.append(self.render())

            if terminated or truncated:
                break

        # 计算奖励（以红队视角）
        with timer(self._time_stats, 'get_reward'):
            reward = self._curriculum.reward(self.env.robots, red_action)

        # 信息
        with timer(self._time_stats, 'make_info'):
            info = {
                'game_state': self.env.game_state,
                'remaining_time': self.env._remaining_time,
                'red_vic_prog': self.env._victory_progress[GameTeam.RED],
                'blue_vic_prog': self.env._victory_progress[GameTeam.RED],
                'red_eco': self.env._economics[GameTeam.RED],
                'blue_eco': self.env._economics[GameTeam.BLUE],
                'render_images': render_images,
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
            remaining_time_norm=(self.env._remaining_time / EnvConfigRMUL.game_time_limit) * 2 - 1,
            victory_progress_red_norm=(self.env._victory_progress[GameTeam.RED] / EnvConfigRMUL.victory_target) * 2 - 1,
            victory_progress_blue_norm=(self.env._victory_progress[GameTeam.BLUE] / EnvConfigRMUL.victory_target) * 2 - 1,
            economics_red_norm=linear_norm(self.env._economics[GameTeam.RED], 0, 400),
            economics_blue_norm=linear_norm(self.env._economics[GameTeam.BLUE], 0, 400),
        )

        # 机器人状态向量
        robots_env = {}
        for robot_id, robot in self.env.robots.items():
            obs_cls = ObsRMULGame._get_storage_robot_obs_cls(self._robot_type_obs, robot.robot_type)
            robots_env[robot_id] = obs_cls.from_robot(robot)
        
        return ObsRMULGame(env_obs, robots_env, self._robot_type_obs)

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
