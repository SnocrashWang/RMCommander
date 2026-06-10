import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

from agents.ppo_agent import PPOAgent
from rules.rmul.config import env_config
from rules.rmul.config.robot_config import RMUL_ROBOT_TYPE_LIST
from rules.rmul.config.zone_config import RMUL_ZONES
from rules.rmul.environment import ActionRMUL
from rules.rmul.game import GameRMUL as Game
from utils.config.game_config import GameState, GameTeam
from utils.config.robot_config import ROBOT_ID, RobotType
from utils.buff import Buff
from utils.grid_map import world_to_grid
from utils.utils import point_in_polygon


SCRIPT_BLUE_SWITCH_INTERVAL = 1.0


def world_to_navigation_norm(position: Tuple[float, float]) -> Tuple[float, float]:
    return (
        position[0] * 2 / env_config.FIELD_WIDTH - 1,
        position[1] * 2 / env_config.FIELD_HEIGHT - 1,
    )


def team_robot_ids(team: GameTeam):
    return [ROBOT_ID[team][robot_type] for robot_type in RMUL_ROBOT_TYPE_LIST]


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
        return not game.env.get_robot(robot_id).grid_map.is_blocked(col, row)
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


def sample_blue_supply_position() -> Tuple[float, float]:
    return sample_point_in_polygon(RMUL_ZONES["blue_boot"].vertices)


def sample_center_position() -> Tuple[float, float]:
    return sample_point_in_polygon(RMUL_ZONES["center"].vertices)


class ScriptBlueController:
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
                robot.hp / robot.max_hp < 0.2
                or robot.ammo_allowed < robot.bullet.PURCHASE_NUM
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
                navigation_target_norm=world_to_navigation_norm(target_world),
                navigation_set=1,
                attack_target=self.attack_targets.get(robot_id, RobotType.STANDARD_3.value),
                spin=1,
                purchase=purchase,
            )
        return actions

    def _resample(self, elapsed_time: float):
        for robot_id in team_robot_ids(GameTeam.BLUE):
            # self.attack_targets[robot_id] = random.choices(RMUL_ROBOT_TYPE_LIST + [RobotType.NONE], weights=[1, 1, 1, 5])[0].value
            self.attack_targets[robot_id] = RobotType.NONE.value
        self.next_switch_time = elapsed_time + SCRIPT_BLUE_SWITCH_INTERVAL

    def _sample_navigation_target(self, robot_id: str, needs_supply: bool) -> Tuple[float, float]:
        if needs_supply:
            return sample_blue_supply_position()
        if random.getrandbits(1):
            return sample_center_position()
        return sample_valid_map_position(self.game, robot_id)


def robot_action_array(action: Dict[str, ActionRMUL]) -> np.ndarray:
    return np.concatenate([
        action[robot_id].to_array()
        for robot_id in team_robot_ids(GameTeam.RED)
    ])


def basic_action_reward(actions: Dict[str, ActionRMUL]):
    reward = 0
    for id, action in actions.items():
        # if action.navigation_set == 1:
        #     reward += 0.1
        # else:
        #     reward -= 0.1
        if action.attack_target in [robot_type.value for robot_type in RMUL_ROBOT_TYPE_LIST] + [RobotType.NONE]:
            reward += 0.1
        else:
            reward -= 0.1
    return reward


def nav_reward(info: Dict, actions: Dict[str, ActionRMUL]):
    reward = 0
    for id, robot in info["robots"].items():
        if id not in actions:
            continue
        if robot.hp / robot.max_hp > 0.9:
            reward += 0.05 * np.linalg.norm(actions[id].navigation_target_norm - np.array([-1, -1], dtype=np.float32))
            if actions[id].navigation_set == 1:
                reward += 0.02
        else:
            if Buff(name="boot", healing=0.25) in robot._buff_manager._buff_list:
                reward += 0.1
    return reward


def economic_reward(info: Dict, action: Dict[str, ActionRMUL]):
    reward = 0
    economics = info["economics"][GameTeam.RED]
    for id, robot in info["robots"].items():
        if id not in action:
            continue
        if action[id].purchase == 1:
            # 在补给区且钱足够
            if Buff(name="boot", healing=0.25) in robot._buff_manager._buff_list and economics >= robot.bullet.PRICE * robot.bullet.PURCHASE_NUM:
                reward += 0.05
                if robot.ammo_allowed < robot.bullet.PURCHASE_NUM:
                    reward += 0.1
                if robot.ammo_allowed >= robot.bullet.PURCHASE_NUM * 2:
                    reward -= 0.1
            else:
                reward -= 0.05
    return reward


def game_reward(info: Dict, last_info: Dict) -> float:
    reward = 0

    # 胜利进度奖励
    cur_red_progress = info["victory_progress"][GameTeam.RED]
    cur_blue_progress = info["victory_progress"][GameTeam.BLUE]
    pre_red_progress = last_info["victory_progress"][GameTeam.RED]
    pre_blue_progress = last_info["victory_progress"][GameTeam.BLUE]
    progress_gain = (cur_red_progress - pre_red_progress) - (cur_blue_progress - pre_blue_progress)
    reward += 1.0 * progress_gain

    # 终局奖励
    if info["game_state"] == GameState.RED_TEAM_WIN:
        reward += 100.0
    elif info["game_state"] == GameState.BLUE_TEAM_WIN:
        reward -= 100.0
    elif info["game_state"] == GameState.DRAW:
        reward -= 10.0

    return reward


def rollout(
    game: Game,
    agent: PPOAgent,
    control_steps: int,
    blue_mode: str,
    deterministic: bool = False,
    train: bool = False,
):
    obs, info = game.reset()
    script_blue_controller = ScriptBlueController(game) if blue_mode == "script" else None
    last_info = info
    state = obs.to_array(GameTeam.RED)
    transition_dict = {"states": [], "actions": [], "next_states": [], "rewards": [], "dones": []}
    total_reward = 0.0
    steps = 0

    max_decision_steps = env_config.GAME_TIME_LIMIT * env_config.FPS // control_steps + 1
    for _ in range(max_decision_steps):
        red_action = agent.take_action(state, GameTeam.RED, deterministic=deterministic)
        if blue_mode == "self":
            blue_action = agent.take_action(obs.to_array(GameTeam.BLUE), GameTeam.BLUE, deterministic=deterministic)
        elif blue_mode == "script":
            elapsed_time = game.env.total_time - game.env._remaining_time
            blue_action = script_blue_controller.make_action(elapsed_time)
        else:
            raise ValueError(f"Unsupported blue_mode: {blue_mode}")

        next_obs, _, terminated, truncated, info = game.step(red_action, blue_action, control_steps)
        next_state = next_obs.to_array(GameTeam.RED)
        reward = basic_action_reward(red_action) + \
                 nav_reward(info, red_action) + \
                 economic_reward(info, red_action) + \
                 game_reward(info, last_info)
        reward = float(reward)
        done = terminated or truncated
        last_info = info

        if train:
            transition_dict["states"].append(state)
            transition_dict["actions"].append(robot_action_array(red_action))
            transition_dict["next_states"].append(next_state)
            transition_dict["rewards"].append(reward)
            transition_dict["dones"].append(done)

        obs = next_obs
        state = next_state
        total_reward += reward
        steps += 1
        if done:
            break

    return transition_dict, {
        "game_state": game.env.game_state.name,
        "elapsed_time": game.env.total_time - game.env._remaining_time,
        "red_progress": game.env._victory_progress[GameTeam.RED],
        "blue_progress": game.env._victory_progress[GameTeam.BLUE],
        "reward": total_reward,
        "steps": steps,
    }


def evaluate(
    game: Game,
    agent: PPOAgent,
    episodes: int,
    control_steps: int,
    blue_mode: str,
):
    results = []
    for _ in range(episodes):
        _, result = rollout(
            game,
            agent,
            control_steps=control_steps,
            blue_mode=blue_mode,
            deterministic=True,
            train=False,
        )
        results.append(result)
    wins = Counter(result["game_state"] for result in results)
    return {
        "episodes": episodes,
        "wins": dict(sorted(wins.items())),
        "avg_elapsed_time": float(np.mean([r["elapsed_time"] for r in results])),
        "avg_red_progress": float(np.mean([r["red_progress"] for r in results])),
        "avg_blue_progress": float(np.mean([r["blue_progress"] for r in results])),
        "avg_reward": float(np.mean([r["reward"] for r in results])),
    }


def train(args):
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_agent_{time_tag}"

    game = Game()
    agent = PPOAgent(
        state_dim=game.observation_space.shape[0],
        device=args.device,
        robot_type_list=RMUL_ROBOT_TYPE_LIST,
        action_cls=ActionRMUL,
    )
    if args.base_model:
        agent.load(args.base_model)

    control_steps = max(1, int(Game.metadata["render_fps"] // args.control_frequency))
    history = []

    for episode in tqdm(range(1, args.episodes + 1), dynamic_ncols=True):
        blue_mode = args.blue_mode
        if blue_mode == "auto":
            blue_mode = random.choice(["self", "script"])

        transition_dict, result = rollout(
            game,
            agent,
            control_steps=control_steps,
            blue_mode=blue_mode,
            deterministic=False,
            train=True,
        )
        if transition_dict["states"]:
            agent.update(transition_dict)
        history.append({"episode": episode, "blue_mode": blue_mode, **result})

        if (args.base_model and episode == 1) or episode % args.eval_interval == 0:
            eval_result = evaluate(
                game,
                agent,
                args.eval_episodes,
                control_steps,
                blue_mode,
            )
            tqdm.write(
                f"episode={episode}\t"
                f"blue_mode={blue_mode}\t"
                f"eval_results={eval_result['wins']}\t"
                f"avg_time={eval_result['avg_elapsed_time']:.2f}s\t"
                f"avg_progress={eval_result['avg_red_progress']:.2f}/{eval_result['avg_blue_progress']:.2f}\t"
                f"avg_reward={eval_result['avg_reward']:.2f}"
            )
            history.append({"episode": episode, "blue_mode": blue_mode, "eval": eval_result})

        if args.save_interval > 0 and episode % args.save_interval == 0:
            checkpoint_path = os.path.join(args.model_dir, f"{run_name}_episode_{episode}.pt")
            agent.save(checkpoint_path)

    model_path = os.path.join(args.model_dir, f"{run_name}.pt")
    log_path = os.path.join(args.log_dir, f"rmul_{time_tag}.json")
    agent.save(model_path)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "model_path": model_path,
                "history": history,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    final_eval = evaluate(
        game,
        agent,
        args.eval_episodes,
        control_steps,
        args.blue_mode if args.blue_mode != "auto" else "script",
    )
    game.close()
    print(json.dumps({"model_path": model_path, "log_path": log_path, "final_eval": final_eval}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="PPO training for RMUL rule.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--control-frequency", type=float, default=2)
    parser.add_argument("--blue-mode", choices=["auto", "self", "script"], default="auto")
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default="models/rmul")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--device", type=str, default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
