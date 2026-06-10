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
from rules.rmul.environment import ActionRMUL
from rules.rmul.game import GameRMUL as Game
from utils.config.game_config import GameState, GameTeam
from utils.config.robot_config import ROBOT_ID, RobotType


CENTER_WORLD = (env_config.FIELD_WIDTH / 2, env_config.FIELD_HEIGHT / 2)


def world_to_navigation_norm(position: Tuple[float, float]) -> Tuple[float, float]:
    return (
        position[0] * 2 / env_config.FIELD_WIDTH - 1,
        position[1] * 2 / env_config.FIELD_HEIGHT - 1,
    )


def team_robot_ids(team: GameTeam):
    return [ROBOT_ID[team][robot_type] for robot_type in RMUL_ROBOT_TYPE_LIST]


def mirror_world_position(position: Tuple[float, float]) -> Tuple[float, float]:
    return env_config.FIELD_WIDTH - position[0], env_config.FIELD_HEIGHT - position[1]


def make_hold_action(team: GameTeam, attack_target: RobotType = RobotType.NONE) -> Dict[str, ActionRMUL]:
    return {
        robot_id: ActionRMUL(
            navigation_set=0,
            attack_target=attack_target.value,
            spin=0,
            purchase=1,
        )
        for robot_id in team_robot_ids(team)
    }


def make_group_nav_action(
    team: GameTeam,
    target_world: Tuple[float, float],
    attack_target: RobotType = RobotType.STANDARD_3,
) -> Dict[str, ActionRMUL]:
    return {
        robot_id: ActionRMUL(
            navigation_target_norm=world_to_navigation_norm(target_world),
            navigation_set=1,
            attack_target=attack_target.value,
            spin=0,
            purchase=1,
        )
        for robot_id in team_robot_ids(team)
    }


def make_blue_action(mode: str) -> Dict[str, ActionRMUL]:
    if mode == "idle":
        return make_hold_action(GameTeam.BLUE, RobotType.NONE)
    if mode == "attack":
        return make_hold_action(GameTeam.BLUE, RobotType.STANDARD_3)
    if mode == "center":
        return make_group_nav_action(
            GameTeam.BLUE,
            mirror_world_position(CENTER_WORLD),
            RobotType.STANDARD_3,
        )
    return make_hold_action(GameTeam.BLUE, RobotType.NONE)


def robot_action_array(action: Dict[str, ActionRMUL]) -> np.ndarray:
    return np.concatenate([
        action[robot_id].to_array()
        for robot_id in team_robot_ids(GameTeam.RED)
    ])


def team_hp(game: Game, team: GameTeam) -> int:
    return sum(robot.hp for robot in game.env.robots.values() if robot.team == team)


def compute_reward(game: Game, before_progress: float, before_enemy_progress: float) -> float:
    red_progress = game.env._victory_progress[GameTeam.RED]
    blue_progress = game.env._victory_progress[GameTeam.BLUE]
    progress_gain = (red_progress - before_progress) - (blue_progress - before_enemy_progress)

    reward = 1.0 * progress_gain - 0.03
    if game.env.game_state == GameState.RED_TEAM_WIN:
        reward += 100.0
    elif game.env.game_state == GameState.BLUE_TEAM_WIN:
        reward -= 100.0
    elif game.env.game_state == GameState.DRAW:
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
    obs, _ = game.reset()
    state = obs.to_array(GameTeam.RED)
    transition_dict = {"states": [], "actions": [], "next_states": [], "rewards": [], "dones": []}
    total_reward = 0.0
    steps = 0

    max_decision_steps = env_config.GAME_TIME_LIMIT * env_config.FPS // control_steps + 1
    for _ in range(max_decision_steps):
        red_action = agent.take_action(state, GameTeam.RED, deterministic=deterministic)
        if blue_mode == "self":
            blue_action = agent.take_action(obs.to_array(GameTeam.BLUE), GameTeam.BLUE, deterministic=deterministic)
        else:
            blue_action = make_blue_action(blue_mode)

        before_progress = game.env._victory_progress[GameTeam.RED]
        before_enemy_progress = game.env._victory_progress[GameTeam.BLUE]
        next_obs, _, terminated, truncated, _ = game.step(red_action, blue_action, control_steps)
        next_state = next_obs.to_array(GameTeam.RED)
        reward = compute_reward(game, before_progress, before_enemy_progress)
        done = terminated or truncated

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
    run_name = f"rmul_agent_{time_tag}"

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
            blue_mode = random.choice(["center", "attack", "self"])

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

        if episode == 1 or episode % args.eval_interval == 0:
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
        args.blue_mode if args.blue_mode != "auto" else "center",
    )
    game.close()
    print(json.dumps({"model_path": model_path, "log_path": log_path, "final_eval": final_eval}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="PPO training for RMUL rule.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--control-frequency", type=float, default=2)
    parser.add_argument("--blue-mode", choices=["auto", "self", "idle", "attack", "center"], default="auto")
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default="models/rmul")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
