import random
from enum import Enum
from typing import Union, Tuple

from utils.buff import Buff
from utils.robot import Robot
from utils.utils import point_in_polygon, opposite_team, opposite_position
from utils.config.game_config import GameTeam
from utils.config.robot_config import RobotType

class Zone():
    def __init__(
            self,
            team: Union[GameTeam, None],            # 所属队伍，或者中立
            vertices: list[tuple[float, float]],    # 区域的顶点列表，格式为 [(x1, y1), (x2, y2), ...]
            allow_co_occupation: bool,              # 是否允许双方机器人同时占领
            friend_buff: Buff,
            enemy_buff: Buff,
            robot_type_restrict = [t.value for t in RobotType], # 可获得增益的机器人类型
            if_render: bool = True                              # 是否渲染
        ):
        self.team = team
        self.vertices = vertices
        self.allow_co_occupation = allow_co_occupation
        self.friend_buff = friend_buff
        self.enemy_buff = enemy_buff
        self.robot_type_restrict = robot_type_restrict
        self.if_render = if_render

        self.occupation_team = {GameTeam.RED: False, GameTeam.BLUE: False}
        self.occupation_robots = {GameTeam.RED: [], GameTeam.BLUE: []}

    def update(self, robots: dict[str, Robot]):
        self._update_occupation(robots)
        self._update_robot_buff(robots)

    def _update_occupation(self, robots: dict[str, Robot]):
        # TODO：限制机器人类型
        # 计算红蓝方所有机器人对此增益点的占领情况
        self.occupation_robots = {GameTeam.RED: [], GameTeam.BLUE: []}
        for robot in robots.values():
            if point_in_polygon(robot.get_position(), self.vertices):
                self.occupation_robots[robot.team].append(robot.id)
        # 当前双方占领情况
        cur_occupation_team = {
            GameTeam.RED: bool(self.occupation_robots[GameTeam.RED]),
            GameTeam.BLUE: bool(self.occupation_robots[GameTeam.BLUE])
        }

        # 如果允许同时占领
        if self.allow_co_occupation:
            self.occupation_team = cur_occupation_team
        # 如果不允许同时占领
        else:
            # 如果之前红方已占领
            if self.occupation_team[GameTeam.RED]:
                self.occupation_team[GameTeam.RED] = cur_occupation_team[GameTeam.RED]
            # 如果之前蓝方已占领
            elif self.occupation_team[GameTeam.BLUE]:
                self.occupation_team[GameTeam.BLUE] = cur_occupation_team[GameTeam.BLUE]
            # 之前双方均未占领
            else:
                # 如果恰巧此时同时占领，随机选择一方成功占领
                if cur_occupation_team[GameTeam.RED] and cur_occupation_team[GameTeam.BLUE]:
                    team = random.choice([GameTeam.RED, GameTeam.BLUE])
                    self.occupation_team[team] = True
                # 否则直接继承当前占领情况即可
                else:
                    self.occupation_team = cur_occupation_team
            assert not (self.occupation_team[GameTeam.RED] and self.occupation_team[GameTeam.BLUE]), "该增益点不可同时占领"

    def _update_robot_buff(self, robots: dict[str, Robot]):
        for robot in robots.values():
            if robot.id in self.occupation_robots[robot.team]:
                if robot.team == self.team or self.team == None:
                    robot.add_buff(self.friend_buff)
                else:
                    robot.add_buff(self.enemy_buff)
            else:
                if robot.team == self.team or self.team == None:
                    robot.remove_buff(self.friend_buff)
                else:
                    robot.remove_buff(self.enemy_buff)

    def check_robot_in_zone(self, robot):
        return robot.id in self.occupation_robots[robot.team]

def make_opposite_zone(zone: Zone, field_size: Tuple[float, float]):
    return Zone(
        None if zone.team == None else opposite_team(zone.team),
        [opposite_position(v, field_size) for v in zone.vertices],
        zone.allow_co_occupation,
        zone.friend_buff,
        zone.enemy_buff
    )