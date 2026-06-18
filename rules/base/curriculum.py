import math
import random
import numpy as np
from copy import deepcopy
from typing import List, Dict, Tuple

from rules.base.config.action_config import ActionBase
from rules.base.config.env_config import EnvConfigBase
from rules.base.config.obstacle_config import BASE_OBSTACLE_CONFIGS
from rules.base.config.robot_config import BASE_ROBOT_TYPE_ACTION, BASE_ROBOT_CONFIGS

from utils.config.exp_prop_config import *
from utils.config.game_config import GameTeam, GameState
from utils.config.robot_config import RobotConfig, RobotType, ROBOT_ID
from utils.obstacle import Obstacle
from utils.robot import Robot
from utils.utils import pos_norm2real, pos_real2norm, opposite_team, calc_distance, attack_sight_clear


class EnemyScriptControllerBase():
    """脚本化行为控制器，用于控制简单的敌方行动"""
    def __init__(
            self,
            auto_nav: bool = False,
            attack_weight: List[int] = [0, 1],
            spin_mode: int = 0
        ):
        """
        Args:
            auto_nav: 是否自动导航（一直切换不同导航目标）
            attack_weight: 自动攻击权重，按 ROBOT_TYPE + NONE 排序
            spin: 0 不自旋，1 常态自旋，2 随机
        """
        self.auto_nav = auto_nav
        self.attack_weight = attack_weight
        self.spin_mode = spin_mode

        self.attack_target_list = list(BASE_ROBOT_TYPE_ACTION) + [RobotType.NONE]
        assert len(self.attack_weight) == len(self.attack_target_list), \
            f"attack_weight({self.attack_weight}) must has the same length as self.attack_target_list({self.attack_target_list})"

        self.switch_time_interval = (1, 5)         # 动作切换时间
        self.next_switch_time = math.inf
        self.update()

    def take_action(self, time: float, robots: Dict[str, Robot]):
        actions = {}
        for id, robot in robots.items():
            if robot.team == GameTeam.BLUE:
                actions[id] = ActionBase(
                    navigation_target_norm=self.nav_target,
                    navigation_set=int(self.auto_nav),
                    attack_target=self.attack_target,
                    spin=self.spin
                )
        if time < self.next_switch_time:
            self.update()
            # 重新随机下次更新时间
            self.next_switch_time = time - random.uniform(*self.switch_time_interval)
        return actions

    def update(self):
        self.nav_target = self._random_nav_target_norm()
        self.attack_target = self._random_attack_target()
        self.spin = self._random_spin()

    def _random_nav_target_norm(self):
        return np.random.uniform(-1, 1, 2)

    def _random_attack_target(self):
        return random.choices(self.attack_target_list, self.attack_weight)[0].value

    def _random_spin(self):
        if self.spin_mode == 2:
            return random.choice([0, 1])
        else:
            return self.spin_mode


class EnemyScriptControllerBasePeek():
    def __init__(self, obstacles):
        self.obstacles = obstacles
        self.last_position = {}
        self.last_peek = {}
        self.last_attack_target = {}
        self.nav_target = {}
        self.overheat_threshold = 0.8
        self.cooldown_threshold = 0.0

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    def take_action(self, time: float, robots: Dict[str, Robot]):
        actions = {}
        robots_red = {
            id: robot for id, robot in robots.items()
            if robot.team == GameTeam.RED and robot.is_alive
        }
        robots_blue = {
            id: robot for id, robot in robots.items()
            if robot.team == GameTeam.BLUE and robot.is_alive
        }

        for id, robot_blue in robots_blue.items():
            target = self._find_visible_target(robot_blue, robots_red, robots)
            if target:
                self.last_attack_target[id] = target.robot_type
                self.last_position[id] = target.get_position()

            is_peeking = self.last_peek.get(id, False)
            if robot_blue.heat > robot_blue.max_heat * self.overheat_threshold:
                is_peeking = True
            elif is_peeking and robot_blue.heat <= robot_blue.max_heat * self.cooldown_threshold:
                is_peeking = False
            self.last_peek[id] = is_peeking

            if is_peeking:
                cover_target_norm = self._cover_target_norm(id, robot_blue, target, robots)
                actions[id] = ActionBase(
                    navigation_target_norm=cover_target_norm,
                    navigation_set=1,
                    attack_target=RobotType.NONE.value,
                    spin=1,
                )
            elif target:
                self._stop_navigation(robot_blue)
                actions[id] = ActionBase(
                    navigation_target_norm=self._current_position_norm(robot_blue),
                    navigation_set=0,
                    attack_target=target.robot_type.value,
                    spin=1,
                )
            else:
                nearest_target = self._find_nearest_target(robot_blue, robots_red)
                if nearest_target:
                    navigation_target_norm = self._pursuit_target_norm(id, robot_blue, nearest_target)
                else:
                    navigation_target_norm = self._current_position_norm(robot_blue)

                actions[id] = ActionBase(
                    navigation_target_norm=navigation_target_norm,
                    navigation_set=1,
                    attack_target=RobotType.NONE.value,
                    spin=1,
                )
        return actions

    def _find_visible_target(
            self,
            robot_blue: Robot,
            robots_red: Dict[str, Robot],
            robots: Dict[str, Robot],
        ):
        visible_targets = []
        robot_blue_pos = robot_blue.get_position()
        for robot_red in robots_red.values():
            if attack_sight_clear(
                robot_blue_pos,
                robot_red.get_position(),
                robot_red.radius,
                self.obstacles,
                robots,
            ):
                visible_targets.append(robot_red)
        if not visible_targets:
            return None
        locked_target_type = self.last_attack_target.get(robot_blue.id)
        for robot in visible_targets:
            if robot.robot_type == locked_target_type:
                return robot
        return min(visible_targets, key=lambda robot: calc_distance(robot_blue_pos, robot.get_position()))

    def _find_nearest_target(self, robot_blue: Robot, robots_red: Dict[str, Robot]):
        if not robots_red:
            return None
        robot_blue_pos = robot_blue.get_position()
        return min(robots_red.values(), key=lambda robot: calc_distance(robot_blue_pos, robot.get_position()))

    def _stop_navigation(self, robot: Robot):
        robot.target_pos = robot.get_position()
        robot.path_points = []
        robot.current_path_idx = 0
        if robot._body is not None:
            robot._body.velocity = (0, 0)

    def _pursuit_target_norm(self, id: str, robot_blue: Robot, target: Robot):
        pursuit_target = self._clip_to_field(target.get_position())
        self.nav_target[id] = pos_real2norm(pursuit_target, EnvConfigBase.field_size())
        return self.nav_target[id]

    def _current_position_norm(self, robot: Robot):
        return pos_real2norm(robot.get_position(), EnvConfigBase.field_size())

    def _clip_to_field(self, position: Tuple[float, float]):
        field_width, field_height = EnvConfigBase.field_size()
        return (
            min(field_width, max(0, position[0])),
            min(field_height, max(0, position[1])),
        )

    def _cover_target_norm(self, id: str, robot_blue: Robot, target: Robot, robots: Dict[str, Robot]):
        robot_pos = robot_blue.get_position()
        target_pos = target.get_position() if target else self.last_position.get(id)
        if target_pos is None:
            return self._fallback_retreat_norm(robot_pos, (EnvConfigBase.field_width / 2, EnvConfigBase.field_height / 2))

        candidates = self._cover_candidates(robot_blue, target_pos, robots)
        if candidates:
            best_candidate = min(candidates, key=lambda candidate: self._cover_score(robot_pos, target_pos, candidate))
            return pos_real2norm(best_candidate, EnvConfigBase.field_size())

        return self._fallback_retreat_norm(robot_pos, target_pos)

    def _cover_candidates(self, robot_blue: Robot, target_pos: Tuple[float, float], robots: Dict[str, Robot]):
        candidates = []
        fallback_candidates = []
        field_width, field_height = EnvConfigBase.field_size()

        for obstacle in self.obstacles:
            dx = obstacle.p2[0] - obstacle.p1[0]
            dy = obstacle.p2[1] - obstacle.p1[1]
            length = math.hypot(dx, dy)
            if length == 0:
                continue

            dir_x, dir_y = dx / length, dy / length
            normal_x, normal_y = -dir_y, dir_x
            to_enemy_x = target_pos[0] - obstacle.center[0]
            to_enemy_y = target_pos[1] - obstacle.center[1]
            enemy_side = 1 if to_enemy_x * normal_x + to_enemy_y * normal_y >= 0 else -1
            cover_normal = (-enemy_side * normal_x, -enemy_side * normal_y)
            side_offset = obstacle.thickness / 2 + robot_blue.radius + 0.08

            for t in (0.15, 0.35, 0.5, 0.65, 0.85):
                base = (
                    obstacle.p1[0] + dx * t,
                    obstacle.p1[1] + dy * t,
                )
                candidate = (
                    base[0] + cover_normal[0] * side_offset,
                    base[1] + cover_normal[1] * side_offset,
                )
                if not (0 <= candidate[0] <= field_width and 0 <= candidate[1] <= field_height):
                    continue
                if not robot_blue.is_valid_target(candidate):
                    continue

                fallback_candidates.append(candidate)
                if not attack_sight_clear(
                    target_pos,
                    candidate,
                    robot_blue.radius,
                    self.obstacles,
                    robots,
                    ignore_robot_blocked=True,
                ):
                    candidates.append(candidate)

        return candidates or fallback_candidates

    def _cover_score(
            self,
            robot_pos: Tuple[float, float],
            target_pos: Tuple[float, float],
            candidate: Tuple[float, float],
        ):
        return (
            calc_distance(robot_pos, candidate)
            + 0.2 * self._point_to_segment_distance(candidate, robot_pos, target_pos)
        )

    def _fallback_retreat_norm(self, robot_pos: Tuple[float, float], target_pos: Tuple[float, float]):
        dx = robot_pos[0] - target_pos[0]
        dy = robot_pos[1] - target_pos[1]
        length = math.hypot(dx, dy)
        if length == 0:
            dx, dy, length = 1, 0, 1

        retreat_distance = 0.8
        field_width, field_height = EnvConfigBase.field_size()
        retreat_pos = (
            min(field_width, max(0, robot_pos[0] + dx / length * retreat_distance)),
            min(field_height, max(0, robot_pos[1] + dy / length * retreat_distance)),
        )
        return pos_real2norm(retreat_pos, EnvConfigBase.field_size())

    def _point_to_segment_distance(
            self,
            point: Tuple[float, float],
            segment_start: Tuple[float, float],
            segment_end: Tuple[float, float],
        ):
        px, py = point
        x1, y1 = segment_start
        x2, y2 = segment_end
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return calc_distance(point, segment_start)

        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        projection = (x1 + t * dx, y1 + t * dy)
        return calc_distance(point, projection)


class CurriculumBase():
    """
    该课程作为课程基类，调用时仅用于测试
    对手脚本随机移动、不攻击
    """
    def __init__(self):
        self.env_config = EnvConfigBase()
        self.obstacle_configs = deepcopy(BASE_OBSTACLE_CONFIGS)
        self.robot_configs = deepcopy(BASE_ROBOT_CONFIGS)

        self._enemy_controller = EnemyScriptControllerBase()

    def random_start(self, if_env: bool, if_obstacles: bool, if_robots: bool):
        if if_env:
            self._random_env_config()
        if if_obstacles:
            self._random_obstacle_configs()
        if if_robots:
            self._random_robot_configs()
        return self.env_config, self.obstacle_configs, self.robot_configs

    def _random_env_config(self):
        self.env_config = EnvConfigBase(
            game_remaining_time=random.uniform(30, EnvConfigBase().game_time_limit)
        )

    def _random_obstacle_configs(self, std: float = 0.1):
        for obstacle in deepcopy(BASE_OBSTACLE_CONFIGS):
            obstacle["p1"] = (obstacle["p1"][0] + random.gauss(0, std), obstacle["p1"][1] + random.gauss(0, std))
            obstacle["p2"] = (obstacle["p2"][0] + random.gauss(0, std), obstacle["p2"][1] + random.gauss(0, std))
            obstacle["thickness"] += random.gauss(0, std)

    def _random_robot_configs(self):
        self.robot_configs = []
        for team in GameTeam:
            for robot_type in BASE_ROBOT_TYPE_ACTION:
                init_pos = pos_norm2real(np.random.uniform(-0.9, 0.9, 2), EnvConfigBase.field_size())       # 为了防止被随机围墙挤到地图外，此处随机初始化位置范围为 [-0.9, 0.9]
                chassis_property_type = random.choice([CHASSIS_PROPERTY_TYPE.POWER, CHASSIS_PROPERTY_TYPE.HP])
                gimbal_property_type = random.choice([GIMBAL_PROPERTY_TYPE.HEAT, GIMBAL_PROPERTY_TYPE.COOLDOWN])
                max_hp = CHASSIS_PROPERTY_STANDARD[chassis_property_type][1]["HP"]
                max_heat = GIMBAL_PROPERTY_17[gimbal_property_type][1]["HEAT"]
                self.robot_configs.append(RobotConfig(
                    team=team,
                    robot_type=robot_type,
                    init_pos=init_pos,
                    chassis_property_type=chassis_property_type,
                    gimbal_property_type=gimbal_property_type,
                    max_ammo=200,
                    ammo_allowed=200,
                    hp=random.randint(int(0.1 * max_hp), max_hp),
                    heat=random.uniform(0, max_heat),
                    enable_exp=False,
                ))

    def get_enemy_action(self, time: float, robots: Dict[str, Robot]):
        """
        Args:
            time: 倒计时
            robots: 环境中的所有机器人
        """
        return self._enemy_controller.take_action(time, robots)

    def reward(self, game_state: GameState, robots: Dict[str, Robot], actions: Dict[str, ActionBase]):
        return 0


class CurriculumBaseMovement(CurriculumBase):
    """
    该课程用于训练模型的基本移动能力，包括设置导航点、靠近敌人等
    对手脚本随机移动、不攻击
    """
    def __init__(self):
        super().__init__()

        self._enemy_controller = EnemyScriptControllerBase(auto_nav=True)

    def reward(self, game_state: GameState, robots: Dict[str, Robot], actions: Dict[str, ActionBase]):
        """计算红方奖励"""
        reward = 0
        for id, action in actions.items():
            if action.navigation_set != 1:
                reward += 0.01
                continue

            # 鼓励导航点合法
            navigation_target = pos_norm2real(action.navigation_target_norm, EnvConfigBase.field_size())
            reward += 0.05 if robots[id].is_valid_target(navigation_target) else -0.1

            # 鼓励导航点接近蓝方坐标
            blue_robot_pos_list = [robot.get_position() for robot in robots.values() if robot.team == opposite_team(robots[id].team)]
            min_dist = min([calc_distance(navigation_target, pos) for pos in blue_robot_pos_list])
            reward += min(1, math.exp(-2 * (min_dist - 1)))
        return reward


class CurriculumBaseBattle(CurriculumBase):
    """
    该课程用于训练模型的战斗能力，包括选择攻击目标、自旋防御等
    对手脚本随机移动、少量攻击
    """
    def __init__(self):
        super().__init__()

        self._enemy_controller = EnemyScriptControllerBase(auto_nav=True)

    def reward(self, game_state: GameState, robots: Dict[str, Robot], actions: Dict[str, ActionBase]):
        """计算红方奖励"""
        reward = 0
        for id, action in actions.items():
            robot_attacker = robots[id]
            # 奖励攻击目标
            if RobotType(action.attack_target) in list(BASE_ROBOT_TYPE_ACTION) + [RobotType.NONE]:
                reward += 0.05

                if action.attack_target != RobotType.NONE.value:
                    robot_target = robots[ROBOT_ID[opposite_team(robot_attacker.team)][RobotType(action.attack_target)]]
                    # 奖励攻击视野
                    if attack_sight_clear(
                        robot_attacker.get_position(),
                        robot_target.get_position(),
                        robot_target.radius,
                        [Obstacle(obstacle_config) for obstacle_config in self.obstacle_configs],
                        robots,
                    ):
                        reward += 0.05
                    else:
                        reward -= 0.05

            # 奖励自旋防御
            robot_enemy = robots[ROBOT_ID[opposite_team(robot_attacker.team)][robot_attacker.robot_type]]
            if attack_sight_clear(
                robot_enemy.get_position(),
                robot_attacker.get_position(),
                robot_attacker.radius,
                [Obstacle(obstacle_config) for obstacle_config in self.obstacle_configs],
                robots,
            ):
                if action.spin == 1:
                    reward += 0.05
                else:
                    reward -= 0.05
        return reward


class CurriculumBaseEasy(CurriculumBase):
    """
    该课程用于训练模型的完整能力
    对手脚本随机移动、少量攻击
    """
    def __init__(self):
        super().__init__()

        self._enemy_controller = EnemyScriptControllerBase(auto_nav=True, attack_weight=[1, 1], spin_mode=2)
        self.hp_recorder = {}

    def reward(self, game_state: GameState, robots: Dict[str, Robot], actions: Dict[str, ActionBase]):
        # 时间惩罚
        reward = -0.01
        for id, action in actions.items():
            # 血量奖励
            if self.hp_recorder:
                robot_attacker = robots[id]
                # 被攻击惩罚
                reward -= 0.005 * (self.hp_recorder[id] - robot_attacker.hp)
                # 如果攻击目标合法
                if RobotType(action.attack_target) in BASE_ROBOT_TYPE_ACTION:
                    robot_target = robots[ROBOT_ID[opposite_team(robot_attacker.team)][RobotType(action.attack_target)]]
                    # 攻击奖励
                    reward += 0.005 * (self.hp_recorder[robot_target.id] - robot_target.hp)

        # 胜利奖励
        if game_state == GameState.RED_TEAM_WIN:
            reward += 10
        elif game_state == GameState.BLUE_TEAM_WIN:
            reward -= 10

        self.hp_recorder = {id: robot.hp for id, robot in robots.items()}
        return reward

    def _random_robot_configs(self):
        self.robot_configs = []
        for team in GameTeam:
            for robot_type in BASE_ROBOT_TYPE_ACTION:
                init_pos = pos_norm2real(np.random.uniform(-0.9, 0.9, 2), EnvConfigBase.field_size())       # 为了防止被随机围墙挤到地图外，此处随机初始化位置范围为 [-0.9, 0.9]
                chassis_property_type = CHASSIS_PROPERTY_TYPE.POWER
                gimbal_property_type = GIMBAL_PROPERTY_TYPE.COOLDOWN
                max_hp = CHASSIS_PROPERTY_STANDARD[chassis_property_type][1]["HP"]
                max_heat = GIMBAL_PROPERTY_17[gimbal_property_type][1]["HEAT"]
                self.robot_configs.append(RobotConfig(
                    team=team,
                    robot_type=robot_type,
                    init_pos=init_pos,
                    chassis_property_type=chassis_property_type,
                    gimbal_property_type=gimbal_property_type,
                    max_ammo=200,
                    ammo_allowed=200,
                    hp=random.randint(int(0.8 * max_hp), max_hp),
                    heat=random.uniform(0, max_heat),
                    enable_exp=False,
                ))


class CurriculumBaseMedium(CurriculumBaseEasy):
    """
    该课程用于训练模型的完整能力
    对手脚本随机移动、少量攻击
    """
    def __init__(self):
        super().__init__()

        self._enemy_controller = EnemyScriptControllerBase(auto_nav=True, attack_weight=[1, 0], spin_mode=2)


class CurriculumBaseHard(CurriculumBaseEasy):
    """
    该课程用于训练模型的完整能力
    对手脚本随机移动、全程攻击
    """
    def __init__(self):
        super().__init__()

        self._enemy_controller = EnemyScriptControllerBase(auto_nav=True, attack_weight=[1, 0], spin_mode=1)


class CurriculumBasePeek(CurriculumBaseEasy):
    def __init__(self):
        super().__init__()

        self._enemy_controller = EnemyScriptControllerBasePeek(
            [Obstacle(obstacle_config) for obstacle_config in self.obstacle_configs]
        )

    def random_start(self, if_env: bool, if_obstacles: bool, if_robots: bool):
        env_config, obstacle_configs, robot_configs = super().random_start(if_env, if_obstacles, if_robots)
        self._enemy_controller.set_obstacles([Obstacle(obstacle_config) for obstacle_config in obstacle_configs])
        return env_config, obstacle_configs, robot_configs
