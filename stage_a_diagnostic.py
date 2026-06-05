import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from rules.base.config import env_config
from rules.base.environment import Action
from rules.base.game import Game
from utils.config.game_config import GameState, GameTeam
from utils.config.robot_config import RobotType
from utils.grid_map import world_to_grid
from utils.utils import attack_sight_clear, calc_distance


RED_ID = "RED_3_STANDARD"
BLUE_ID = "BLUE_3_STANDARD"


def force_stage_a_action(action: Dict[str, Action]) -> Dict[str, Action]:
    robot_action = action[RED_ID]
    robot_action.navigation_set = 1
    robot_action.attack_target = RobotType.STANDARD_3.value
    return action


def make_blue_action(mode: str) -> Dict[str, Action]:
    if mode == "attack":
        return {
            BLUE_ID: Action(
                navigation_set=0,
                attack_target=RobotType.STANDARD_3.value,
            )
        }
    return {BLUE_ID: Action()}


def robot_snapshot(game: Game) -> Tuple[Tuple[float, float], Tuple[float, float], int, int]:
    red = game.env.get_robot(RED_ID)
    blue = game.env.get_robot(BLUE_ID)
    return red.get_position(), blue.get_position(), red.hp, blue.hp


def is_valid_position(game: Game, position: Tuple[float, float]) -> bool:
    try:
        col, row = world_to_grid(position)
        return not game.env.get_robot(RED_ID).grid_map.is_blocked(col, row)
    except ValueError:
        return False


def sample_position(game: Game, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> Tuple[float, float]:
    for _ in range(1000):
        position = (random.uniform(*x_range), random.uniform(*y_range))
        if is_valid_position(game, position):
            return position
    raise RuntimeError("failed to sample a valid start position")


def apply_random_start(game: Game):
    red = game.env.get_robot(RED_ID)
    blue = game.env.get_robot(BLUE_ID)

    red_pos = sample_position(game, (0.4, 2.2), (0.4, 4.6))
    blue_pos = sample_position(game, (2.8, 4.6), (0.4, 4.6))
    while calc_distance(red_pos, blue_pos) < 2.0:
        blue_pos = sample_position(game, (2.8, 4.6), (0.4, 4.6))

    red._body.position = red_pos
    blue._body.position = blue_pos
    red._body.velocity = (0, 0)
    blue._body.velocity = (0, 0)
    red.target_pos = red_pos
    blue.target_pos = blue_pos
    red.path_points = []
    blue.path_points = []
    red.current_path_idx = 0
    blue.current_path_idx = 0


def stage_a_reward(game: Game, before, after) -> float:
    red_pos_before, blue_pos_before, red_hp_before, blue_hp_before = before
    red_pos_after, blue_pos_after, red_hp_after, blue_hp_after = after

    enemy_damage = max(0, blue_hp_before - blue_hp_after)
    self_damage = max(0, red_hp_before - red_hp_after)
    distance_before = calc_distance(red_pos_before, blue_pos_before)
    distance_after = calc_distance(red_pos_after, blue_pos_after)
    distance_gain = distance_before - distance_after

    red = game.env.get_robot(RED_ID)
    blue = game.env.get_robot(BLUE_ID)
    has_los = attack_sight_clear(
        red.get_position(),
        blue.get_position(),
        blue.radius,
        game.env.obstacles,
        game.env.robots.values(),
    )

    reward = 0.0
    reward += 1.0 * distance_gain
    reward += 0.10 * enemy_damage
    reward -= 0.10 * self_damage
    reward += 0.03 if has_los else 0.0
    reward -= 0.01

    if game.env.game_state == GameState.RED_TEAM_WIN:
        reward += 20.0
    elif game.env.game_state == GameState.BLUE_TEAM_WIN:
        reward -= 20.0
    return reward


def rollout(
    agent: PPOAgent,
    control_steps: int,
    blue_mode: str,
    random_start: bool,
    deterministic: bool = False,
    train: bool = False,
):
    game = Game()
    obs, _ = game.reset()
    if random_start:
        apply_random_start(game)
        obs = game._get_obs()
    state = obs.to_array(GameTeam.RED)
    transition_dict = {"states": [], "actions": [], "next_states": [], "rewards": [], "dones": []}
    target_counts = Counter()
    nav_counts = Counter()
    total_reward = 0.0
    steps = 0
    damage_done = 0

    max_decision_steps = env_config.GAME_TIME_LIMIT * env_config.FPS // control_steps + 1
    for _ in range(max_decision_steps):
        red_action = agent.take_action(state, GameTeam.RED, deterministic=deterministic)
        red_action = force_stage_a_action(red_action)
        blue_action = make_blue_action(blue_mode)

        before = robot_snapshot(game)
        next_obs, _, terminated, truncated, info = game.step(red_action, blue_action, control_steps)
        after = robot_snapshot(game)
        reward = stage_a_reward(game, before, after)
        next_state = next_obs.to_array(GameTeam.RED)
        done = terminated or truncated

        executed_action = red_action[RED_ID].to_array()
        target_counts[red_action[RED_ID].attack_target] += 1
        nav_counts[red_action[RED_ID].navigation_set] += 1
        damage_done += max(0, before[3] - after[3])

        if train:
            transition_dict["states"].append(state)
            transition_dict["actions"].append(executed_action)
            transition_dict["next_states"].append(next_state)
            transition_dict["rewards"].append(reward)
            transition_dict["dones"].append(done)

        obs = next_obs
        state = next_state
        total_reward += reward
        steps += 1
        if done:
            break

    result = {
        "game_state": game.env.game_state.name,
        "remaining_time": game.env._remaining_time,
        "red_hp": game.env.get_robot(RED_ID).hp,
        "blue_hp": game.env.get_robot(BLUE_ID).hp,
        "damage_done": damage_done,
        "steps": steps,
        "reward": total_reward,
        "target_counts": dict(target_counts),
        "nav_counts": dict(nav_counts),
    }
    game.close()
    return transition_dict, result


def evaluate(agent: PPOAgent, episodes: int, control_steps: int, blue_mode: str, random_start: bool):
    results = []
    for _ in range(episodes):
        _, result = rollout(
            agent,
            control_steps=control_steps,
            blue_mode=blue_mode,
            random_start=random_start,
            deterministic=True,
            train=False,
        )
        results.append(result)
    wins = Counter(result["game_state"] for result in results)
    return {
        "episodes": episodes,
        "wins": dict(wins),
        "avg_reward": float(np.mean([r["reward"] for r in results])),
        "avg_damage_done": float(np.mean([r["damage_done"] for r in results])),
        "avg_steps": float(np.mean([r["steps"] for r in results])),
        "avg_red_hp": float(np.mean([r["red_hp"] for r in results])),
        "avg_blue_hp": float(np.mean([r["blue_hp"] for r in results])),
        "samples": results[:5],
    }


def train(args):
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    game = Game()
    agent = PPOAgent(state_dim=game.observation_space.shape[0], device=args.device)
    game.close()
    if args.base_model:
        agent.load(args.base_model)

    control_steps = int(Game.metadata["render_fps"] // args.control_frequency)
    history = []

    for episode in range(1, args.episodes + 1):
        transition_dict, result = rollout(
            agent,
            control_steps=control_steps,
            blue_mode=args.blue_mode,
            random_start=args.random_start,
            deterministic=False,
            train=True,
        )
        if transition_dict["states"]:
            agent.update(transition_dict)
        history.append(result)

        if episode == 1 or episode % args.eval_interval == 0:
            eval_result = evaluate(agent, args.eval_episodes, control_steps, args.blue_mode, args.random_start)
            print(
                f"episode={episode} "
                f"train_state={result['game_state']} "
                f"train_damage={result['damage_done']} "
                f"eval_wins={eval_result['wins']} "
                f"eval_damage={eval_result['avg_damage_done']:.2f} "
                f"eval_blue_hp={eval_result['avg_blue_hp']:.2f}"
            )
            history.append({"episode": episode, "eval": eval_result})

    model_path = os.path.join(args.model_dir, f"stage_a_agent_{time_tag}.pt")
    log_path = os.path.join(args.log_dir, f"stage_a_{time_tag}.json")
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
    final_eval = evaluate(agent, args.eval_episodes, control_steps, args.blue_mode, args.random_start)
    print(json.dumps({"model_path": model_path, "log_path": log_path, "final_eval": final_eval}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Stage A diagnostic PPO training for base 1v1.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--control-frequency", type=float, default=2)
    parser.add_argument("--blue-mode", choices=["idle", "attack"], default="idle")
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--device", type=str, default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
