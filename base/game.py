import pygame
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Optional, Tuple, Any

from utils.config.exp_prop_config import LEVEL_NEED_EXP
from utils.config.game_config import GameTeam, GameState
from utils.config.robot_config import RobotType
from utils.grid_map import world_to_grid
from utils.utils import meters_to_pixels, calc_distance, opposite_team
from visualization.renderer import Renderer

from base.config import env_config
from base.config.robot_config import BASE_ROBOT_CONFIGS
from base.environment import Action, GameObs, RobotObs, Observation, Environment

class Game(gym.Env):
   
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

        self.dt = 1 / env_config.FPS
        
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
        """设置动作空间"""
        # 为每个机器人定义动作空间
        robot_action_spaces = {}
        
        for robot_id, robot in self.env.robots.items():
            # 速度动作：x, y方向速度
            velocity_space = spaces.Box(
                low=np.array([0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
            
            # 目标动作：攻击目标类型
            attack_target_space = spaces.Discrete(len(RobotType))  # 所有机器人类型
            
            # 组合动作空间
            robot_action_spaces[robot_id] = spaces.Dict({
                'velocity': velocity_space,
                'attack_target': attack_target_space,
            })
        
        self.action_space = spaces.Dict(robot_action_spaces)
    
    def _setup_observation_space(self):
        """设置观察空间"""
        # 游戏状态：剩余时间
        game_state_remaining_time_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # 机器人状态：位置(2) + 速度(2) + 属性(2) + 等级(1) + 经验(1) + 血量(1) + 热量(1) = 10维
        robot_state_space = spaces.Box(
            low=np.array([0.0, 0.0, -1, -1, 0, 0, 0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1, 1, 2, 2, 10, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # 计算机器人数量
        red_robots = [r for r in self.env.robots.values() if r.team == GameTeam.RED]
        blue_robots = [r for r in self.env.robots.values() if r.team == GameTeam.BLUE]
        
        # 组合观察空间
        observation_low = np.concatenate([
            game_state_remaining_time_space.low,  # 游戏状态
            np.tile(robot_state_space.low, len(red_robots)),  # 红队机器人
            np.tile(robot_state_space.low, len(blue_robots))  # 蓝队机器人
        ])
        
        observation_high = np.concatenate([
            game_state_remaining_time_space.high,  # 游戏状态
            np.tile(robot_state_space.high, len(red_robots)),  # 红队机器人
            np.tile(robot_state_space.high, len(blue_robots))  # 蓝队机器人
        ])
        
        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high,
            dtype=np.float32
        )
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = {}):
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置底层环境
        self.env.reset()

        if "random" in options and options["random"]:
            obs_array = self.observation_space.sample()
            obs = Observation.from_array(obs_array)
            self.apply_observation(obs)
        
        # 渲染
        if self._render_mode:
            render_image = self.render()
        else:
            render_image = None
        
        # 获取初始观察
        observation = self._get_obs()
        info = {
            'render_image': render_image,
        }
        
        return observation, info
    
    def apply_observation(self, observation: Observation):
        """
        【注意！】这是一个非常危险的函数，非特殊情况不要使用！
        直接将指定的观察值赋值到当前环境中
        """
        self.env.apply_observation(observation)
        for robot_id, robot_obs in observation.robot_obs.items():
            robot = self.env.get_robot(robot_id)
            robot.apply_observation(robot_obs)
    
    def step(self, red_action: Dict[str, Action], blue_action: Dict[str, Action], control_steps: int = 1):
        """执行一步动作"""
        reward = 0
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
                render_image = self.render()
            else:
                render_image = None

            if terminated or truncated:
                break
        
        # 信息
        info = {
            'game_state': self.env.game_state,
            'remaining_time': self.env._remaining_time,
            'red_hp': {robot_id: robot.hp for robot_id, robot in self.env.robots.items() if robot.team == GameTeam.RED},
            'blue_hp': {robot_id: robot.hp for robot_id, robot in self.env.robots.items() if robot.team == GameTeam.BLUE},
            'render_image': render_image,
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self) -> Observation:
        """获取观察"""
        # 全局状态向量
        game_state = GameObs(
            remaining_time_norm=self.env._remaining_time / env_config.GAME_TIME_LIMIT,
        )

        # 机器人状态向量
        robot_state = {}
        for robot_id, robot in self.env.robots.items():
            robot_state[robot_id] = RobotObs.from_robot(robot)
        
        return Observation(game_state, robot_state)
    
    def _get_reward(self, team: GameTeam, action: Dict[str, Action]) -> float:
        """获取奖励"""
        reward_list = []
        reward_weight = []

        # 时间消耗惩罚
        reward_time = - self.dt * 1
        reward_list.append(reward_time)
        reward_weight.append(5)

        # # 不可行导航点惩罚
        # if action["RED_3_STANDARD"].navigation_set == 1:
        #     col, row = world_to_grid(action["RED_3_STANDARD"].navigation_target)
        #     if self.env.get_robot("RED_3_STANDARD").grid_map.is_blocked(col, row):
        #         reward_navigation_unmovable = -1.0
        #     else:
        #         reward_navigation_unmovable = 1.0
        # else:
        #     reward_navigation_unmovable = 0.0
        # reward_list.append(reward_navigation_unmovable)
        # reward_weight.append(10)

        # # 导航点差异惩罚
        # try:
        #     last_navigation = self._last_action["RED_3_STANDARD"]["navigation_target"]
        # except:
        #     last_navigation = self.env.robots["RED_3_STANDARD"].position
        # current_navigation = action["RED_3_STANDARD"]["navigation_target"]
        # navigation_diff = calc_distance(last_navigation, current_navigation)
        # reward_navigation_diff = - (navigation_diff ** 2) / (1 + navigation_diff ** 2)
        # reward_list.append(reward_navigation_diff)
        # reward_weight.append(10)

        # 导航代价
        # if action["RED_3_STANDARD"].navigation_set == 1:
        #     reward_navigation_cost = -1.0
        # else:
        #     reward_navigation_cost = 0.0
        # reward_list.append(reward_navigation_cost)
        # reward_weight.append(5)

        # 血量奖励
        our_last_hp = sum([self._last_observation.robot_obs["RED_3_STANDARD"].hp_norm])
        our_hp = sum([robot.hp / robot.max_hp for robot in self.env.robots.values() if robot.team == team])
        enemy_last_hp = sum([self._last_observation.robot_obs["BLUE_3_STANDARD"].hp_norm])
        enemy_hp = sum([robot.hp / robot.max_hp for robot in self.env.robots.values() if robot.team == opposite_team(team)])        
        reward_hp = np.sign((enemy_last_hp - enemy_hp) - (our_last_hp - our_hp))
        reward_list.append(reward_hp)
        reward_weight.append(10)

        # 距离奖励
        our_last_position = np.array(self._last_observation.robot_obs["RED_3_STANDARD"].position)
        our_position = self.env.get_robot("RED_3_STANDARD").get_position()
        enemy_last_position = np.array(self._last_observation.robot_obs["BLUE_3_STANDARD"].position)
        enemy_position = self.env.get_robot("BLUE_3_STANDARD").get_position()
        last_distance = calc_distance(
            our_last_position,
            enemy_last_position
        )
        current_distance = calc_distance(
            our_position,
            enemy_last_position
        )
        # reward_distance = np.sign(last_distance - current_distance)  # 距离减小给予正奖励，距离增加给予负奖励
        reward_distance = (last_distance - current_distance) * 100  # 距离减小给予正奖励，距离增加给予负奖励
        reward_list.append(reward_distance)
        reward_weight.append(10)

        # 撞墙惩罚
        if self.env.physics_engine.shape_query(self.env.robots["RED_3_STANDARD"]._shape):
            reward_collision = -1
        else:
            reward_collision = 1
        reward_list.append(reward_collision)
        reward_weight.append(5)
        
        # 游戏结束奖励
        if self.env.game_state == GameState.RED_TEAM_WIN:
            reward_win = 10.0
        elif self.env.game_state == GameState.BLUE_TEAM_WIN:
            reward_win = -10.0
        else:
            reward_win = 0.0
        
        # 更新状态记录
        self._last_observation = self._get_obs()
        self._last_action = action

        reward = np.average(reward_list, weights=reward_weight)
        # print(reward_list, reward_win, reward)
        return reward + reward_win

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
                    (meters_to_pixels(env_config.FIELD_WIDTH), meters_to_pixels(env_config.FIELD_HEIGHT))
                )
            elif self._render_mode == "rgb_array":
                self._screen = pygame.Surface((meters_to_pixels(env_config.FIELD_WIDTH), meters_to_pixels(env_config.FIELD_HEIGHT)))
            else:
                raise ValueError(f"Invalid render mode: {self._render_mode}")
        if self._renderer is None:
            self._renderer = Renderer(env_config)

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
        if hasattr(self, 'renderer') and self._renderer is not None:
            # 关闭pygame显示
            import pygame
            pygame.display.quit()
            pygame.quit()
