import math
import random
import numpy as np
from copy import deepcopy
from typing import List, Dict

from rules.rmul.config.action_config import ActionRMUL
from rules.rmul.config.env_config import EnvConfigRMUL
from rules.rmul.config.obstacle_config import RMUL_OBSTACLE_CONFIGS
from rules.rmul.config.robot_config import RMUL_ROBOT_TYPE_ACTION, RMUL_ROBOT_CONFIGS
from rules.rmul.config.zone_config import RMUL_ZONES

from utils.config.exp_prop_config import *
from utils.config.game_config import GameTeam, GameState
from utils.config.robot_config import RobotConfig, RobotType, ROBOT_ID
from utils.obstacle import Obstacle
from utils.robot import Robot
from utils.utils import pos_norm2real, pos_real2norm, point_in_polygon, opposite_team, calc_distance, attack_sight_clear


class EnemyScriptControllerRMUL():
    """脚本化行为控制器，用于控制简单的敌方行动"""
    def __init__(
            self,
            auto_nav: bool = False,
            auto_supply: bool = False,
            attack_weight: List[int] = [0, 0, 0, 1],
            spin_mode: int = 0
        ):
        """
        Args:
            auto_nav: 是否自动导航（一直切换不同导航目标）
            attack_weight: 自动攻击权重，按 ROBOT_TYPE + NONE 排序
            spin: 0 不自旋，1 常态自旋，2 随机
        """
        self.auto_nav = auto_nav
        self.auto_supply = auto_supply
        self.attack_weight = attack_weight
        self.spin_mode = spin_mode

        self.attack_target_list = list(RMUL_ROBOT_TYPE_ACTION) + [RobotType.NONE]
        assert len(self.attack_weight) == len(self.attack_target_list), \
            f"attack_weight({self.attack_weight}) must has the same length as self.attack_target_list({self.attack_target_list})"

        self.switch_time_interval = (5, 10)         # 动作切换时间
        self.next_switch_times = {}

        # 记录各机器人的行动
        self.action_records = {}
        self.last_go_home = {}

    def take_action(self, time: float, robots: Dict[str, Robot]):
        actions = {}
        for id, robot in robots.items():
            if robot.team == GameTeam.BLUE:
                # 更新随机行动
                if time < self.next_switch_times.get(id, math.inf):
                    self.update(robot)
                    # 重新随机下次更新时间
                    self.next_switch_times[id] = time - random.uniform(*self.switch_time_interval)

                # 是否满足回家条件
                needs_heal = int(self.auto_supply and (
                    (robot.hp / robot.max_hp < 0.2 or self.last_go_home.get(id) and robot.hp != robot.max_hp)
                    or robot.gun_locked
                ))
                needs_supply = robot.ammo_allowed < robot.bullet.PURCHASE_NUM
                go_home = needs_heal or needs_supply
                # 如果原先没有设定回家
                if self.last_go_home.get(id) != go_home:
                    self.action_records[id].navigation_target_norm = self._random_nav_target_norm(robot, go_home=go_home)
                    self.action_records[id].purchase = needs_supply     # 补给仅限一次
                self.last_go_home[id] = go_home

                actions[id] = self.action_records[id]

        return actions

    def update(self, robot):
        self.action_records[robot.id] = ActionRMUL(
            navigation_target_norm=self._random_nav_target_norm(robot, self.last_go_home.get(id, False)),
            navigation_set=int(self.auto_nav),
            attack_target=self._random_attack_target(),
            spin=self._random_spin(),
        )

    def _sample_point_in_polygon(self, robot: Robot, vertices: List):
        xs = [p[0] for p in vertices]
        ys = [p[1] for p in vertices]
        for _ in range(1000):
            position = (random.uniform(min(xs), max(xs)), random.uniform(min(ys), max(ys)))
            is_valid_target = robot.is_valid_target(position) if robot else True
            if is_valid_target and point_in_polygon(position, vertices):
                return position
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _random_nav_target_norm(self, robot: Robot = None, go_home: bool = False):
        if go_home:
            return pos_real2norm(self._sample_point_in_polygon(robot, RMUL_ZONES["blue_boot"].vertices), EnvConfigRMUL.field_size())
        if random.getrandbits(1):
            return pos_real2norm(self._sample_point_in_polygon(robot, RMUL_ZONES["center"].vertices), EnvConfigRMUL.field_size())
        return np.random.uniform(-1, 1, 2)

    def _random_attack_target(self):
        return random.choices(self.attack_target_list, self.attack_weight)[0].value

    def _random_spin(self):
        if self.spin_mode == 2:
            return random.choice([0, 1])
        else:
            return self.spin_mode


class CurriculumRMUL():
    """
    该课程作为课程基类，调用时仅用于测试
    对手脚本随机移动、不攻击
    """
    def __init__(self):
        self.env_config = EnvConfigRMUL()
        self.obstacle_configs = deepcopy(RMUL_OBSTACLE_CONFIGS)
        self.robot_configs = deepcopy(RMUL_ROBOT_CONFIGS)

        self._enemy_controller = EnemyScriptControllerRMUL(auto_nav=True)

    def random_start(self, if_env: bool, if_obstacles: bool, if_robots: bool):
        if if_env:
            self._random_env_config()
        if if_obstacles:
            self._random_obstacle_configs()
        if if_robots:
            self._random_robot_configs()
        return self.env_config, self.obstacle_configs, self.robot_configs

    def _random_env_config(self):
        self.env_config = EnvConfigRMUL(
            game_remaining_time=random.uniform(30, EnvConfigRMUL().game_time_limit),
            victory_progress_red=random.randint(0, EnvConfigRMUL().victory_target - 20),
            victory_progress_blue=random.randint(0, EnvConfigRMUL().victory_target - 20),
            economics_red=random.randint(0, 400),
            economics_blue=random.randint(0, 400)
        )

    def _random_obstacle_configs(self, std: float = 0.1):
        for obstacle in deepcopy(RMUL_OBSTACLE_CONFIGS):
            obstacle["p1"] = (obstacle["p1"][0] + random.gauss(0, std), obstacle["p1"][1] + random.gauss(0, std))
            obstacle["p2"] = (obstacle["p2"][0] + random.gauss(0, std), obstacle["p2"][1] + random.gauss(0, std))
            obstacle["thickness"] += random.gauss(0, std)

    def _random_robot_configs(self):
        self.robot_configs = []
        for team in GameTeam:
            for robot_type in RMUL_ROBOT_TYPE_ACTION:
                init_pos = (random.uniform(0.5, EnvConfigRMUL.field_width - 0.5), random.uniform(0.5, EnvConfigRMUL.field_height - 0.5))
                # 根据不同机器人类型分别随机属性
                if robot_type == RobotType.HERO:
                    level = random.randint(1, 10)
                    chassis_property_type = random.choice([CHASSIS_PROPERTY_TYPE.POWER, CHASSIS_PROPERTY_TYPE.HP])
                    gimbal_property_type = GIMBAL_PROPERTY_TYPE.DEFAULT
                    max_hp = CHASSIS_PROPERTY_HERO[chassis_property_type][level]["HP"]
                    max_heat = GIMBAL_PROPERTY_42[gimbal_property_type][level]["HEAT"]
                    max_ammo = random.randint(50, 80)
                    ammo_allowed = random.randint(0, 20)
                elif robot_type == RobotType.STANDARD_3:
                    level = random.randint(1, 10)
                    chassis_property_type = random.choice([CHASSIS_PROPERTY_TYPE.POWER, CHASSIS_PROPERTY_TYPE.HP])
                    gimbal_property_type = random.choice([GIMBAL_PROPERTY_TYPE.HEAT, GIMBAL_PROPERTY_TYPE.COOLDOWN])
                    max_hp = CHASSIS_PROPERTY_STANDARD[chassis_property_type][level]["HP"]
                    max_heat = GIMBAL_PROPERTY_17[gimbal_property_type][level]["HEAT"]
                    max_ammo = random.randint(300, 500)
                    ammo_allowed = random.randint(0, 200)
                elif robot_type == RobotType.SENTRY:
                    level = 10
                    chassis_property_type = CHASSIS_PROPERTY_TYPE.HP
                    gimbal_property_type = GIMBAL_PROPERTY_TYPE.COOLDOWN
                    max_hp = CHASSIS_PROPERTY_STANDARD[chassis_property_type][level]["HP"]
                    max_heat = GIMBAL_PROPERTY_17[gimbal_property_type][level]["HEAT"]
                    max_ammo = 750
                    ammo_allowed = random.randint(0, 750)

                self.robot_configs.append(RobotConfig(
                    team=team,
                    robot_type=robot_type,
                    init_pos=init_pos,
                    chassis_property_type=chassis_property_type,
                    gimbal_property_type=gimbal_property_type,
                    max_ammo=max_ammo,
                    ammo_allowed=ammo_allowed,
                    level=level,
                    hp=random.randint(int(0.1 * max_hp), max_hp),
                    heat=random.uniform(0, max_heat),
                    enable_exp=True,
                ))

    def get_enemy_action(self, time: float, robots: Dict[str, Robot]):
        """
        Args:
            time: 倒计时
            robots: 环境中的所有机器人
        """
        return self._enemy_controller.take_action(time, robots)

    def reward(self, game_state: GameState, robots: Dict[str, Robot], actions: Dict[str, ActionRMUL]):
        return 0


class CurriculumRMULMedium(CurriculumRMUL):
    """
    该课程用于训练模型的完整能力
    对手脚本随机移动、随机少量攻击
    """
    def __init__(self):
        super().__init__()

        self._enemy_controller = EnemyScriptControllerRMUL(
            auto_nav=True,
            auto_supply=True,
            attack_weight=[1, 1, 1, 7],
            spin_mode=2
        )

