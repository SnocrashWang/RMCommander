
import random
from typing import Dict, Tuple

from rules.rmul.environment import ActionRMUL
from rules.rmul.game import GameRMUL as Game
from rules.rmul.config import env_config
from rules.rmul.config.robot_config import RMUL_ROBOT_TYPE_ACTION
from rules.rmul.config.zone_config import RMUL_ZONES

from utils.grid_map import world_to_grid
from utils.utils import point_in_polygon, pos_real2norm
from utils.config.game_config import GameState, GameTeam
from utils.config.robot_config import ROBOT_ID, RobotType


SCRIPT_BLUE_SWITCH_INTERVAL = 1.0


def team_robot_ids(team: GameTeam):
    return [ROBOT_ID[team][robot_type] for robot_type in RMUL_ROBOT_TYPE_ACTION]


def sample_point_in_polygon(vertices):
    xs = [p[0] for p in vertices]
    ys = [p[1] for p in vertices]
    for _ in range(1000):
        position = (random.uniform(min(xs), max(xs)), random.uniform(min(ys), max(ys)))
        if point_in_polygon(position, vertices):
            return position
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def is_valid_position(game: Game, position: Tuple[float, float], robot_id: str) -> bool:
    try:
        col, row = world_to_grid(position)
        return not game.env.get_robot(robot_id)._grid_map.is_blocked(col, row)
    except ValueError:
        return False


def sample_valid_map_position(game: Game, robot_id: str) -> Tuple[float, float]:
    for _ in range(1000):
        position = (
            random.uniform(0.3, env_config.FIELD_WIDTH - 0.3),
            random.uniform(0.3, env_config.FIELD_HEIGHT - 0.3),
        )
        if is_valid_position(game, position, robot_id):
            return position
    return game.env.get_robot(robot_id).get_position()


class ScriptControllerRMUL:
    def __init__(self, game: Game):
        self.game = game
        self.next_switch_time = 0.0
        self.attack_targets = {}
        self.last_needs_supply = {}
        self.nav_targets = {}

    def make_action(self, elapsed_time: float) -> Dict[str, ActionRMUL]:
        if elapsed_time >= self.next_switch_time:
            self._resample(elapsed_time)

        actions = {}
        for robot_id in team_robot_ids(GameTeam.BLUE):
            robot = self.game.env.get_robot(robot_id)
            needs_supply = (
                (robot.hp / robot.max_hp < 0.2 or self.last_needs_supply.get(robot_id) and robot.hp != robot.max_hp)
                or robot.ammo_allowed < robot.bullet.PURCHASE_NUM
                or robot.gun_locked
            )
            if (
                robot_id not in self.nav_targets
                or self.last_needs_supply.get(robot_id) != needs_supply
            ):
                self.nav_targets[robot_id] = self._sample_navigation_target(robot_id, needs_supply)
                self.last_needs_supply[robot_id] = needs_supply

            if needs_supply:
                purchase = 1
            else:
                purchase = 0
            target_world = self.nav_targets[robot_id]

            actions[robot_id] = ActionRMUL(
                navigation_target_norm=pos_real2norm(target_world, (env_config.FIELD_WIDTH, env_config.FIELD_HEIGHT)),
                navigation_set=1,
                attack_target=self.attack_targets.get(robot_id, RobotType.STANDARD_3.value),
                spin=1,
                purchase=purchase,
            )
        return actions

    def _resample(self, elapsed_time: float):
        for robot_id in team_robot_ids(GameTeam.BLUE):
            # self.attack_targets[robot_id] = random.choices(list(RMUL_ROBOT_TYPE_ACTION) + [RobotType.NONE], weights=[1, 1, 1, 5])[0].value
            self.attack_targets[robot_id] = RobotType.NONE.value
        self.next_switch_time = elapsed_time + SCRIPT_BLUE_SWITCH_INTERVAL

    def _sample_navigation_target(self, robot_id: str, needs_supply: bool) -> Tuple[float, float]:
        if needs_supply:
            return sample_point_in_polygon(RMUL_ZONES["blue_boot"].vertices)
        if random.getrandbits(1):
            return sample_point_in_polygon(RMUL_ZONES["center"].vertices)
        return sample_valid_map_position(self.game, robot_id)
