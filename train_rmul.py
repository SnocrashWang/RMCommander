import argparse
import json
import os
import math
import random
from collections import Counter
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

from agents.ppo_agent import PPOAgent
from agents.script_agent_rmul import ScriptControllerRMUL
from rules.rmul.config import env_config
from rules.rmul.config.robot_config import RMUL_ROBOT_TYPE_ACTION
from rules.rmul.environment import ActionRMUL
from rules.rmul.game import GameRMUL as Game
from utils.config.game_config import GameState, GameTeam
from utils.config.robot_config import RobotType
from utils.buff import Buff



def basic_action_reward(actions: Dict[str, ActionRMUL]):
    reward = 0
    for id, action in actions.items():
        # if action.navigation_set == 1:
        #     reward += 0.1
        # else:
        #     reward -= 0.1
        if action.attack_target in [robot_type.value for robot_type in RMUL_ROBOT_TYPE_ACTION] + [RobotType.NONE.value]:
            reward += 0.1
        else:
            reward -= 0.1
    return reward


def nav_reward(info: Dict, actions: Dict[str, ActionRMUL]):
    reward = 0
    for id in actions:
        robot = info["robots"][id]
        if robot.hp / robot.max_hp > 0.9:
            dist = np.linalg.norm(actions[id].navigation_target_norm - np.array([-1, -1], dtype=np.float32))
            reward += 0.02 * (1 - math.exp(-dist))
        else:
            if robot.has_buff(Buff(name="boot", healing=0.25)):
                reward += 0.1
    return reward


def center_reward(info: Dict, actions: Dict[str, ActionRMUL]):
    reward = 0
    for id in actions:
        robot = info["robots"][id]
        if robot.has_buff(Buff(name="center")):
            reward += 0.05
    return reward


def economic_reward(info: Dict, actions: Dict[str, ActionRMUL]):
    reward = 0
    economics = info["economics"][GameTeam.RED]
    for id in actions:
        robot = info["robots"][id]
        if actions[id].purchase == 1:
            # 在补给区且钱足够
            if robot.has_buff(Buff(name="boot", healing=0.25)) and economics >= robot.bullet.PRICE * robot.bullet.PURCHASE_NUM:
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
    reward += 0.2 * progress_gain

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
    blue_agent = ScriptControllerRMUL(game) if blue_mode == "script" else None
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
            blue_action = blue_agent.make_action(elapsed_time)
        else:
            raise ValueError(f"Unsupported blue_mode: {blue_mode}")

        next_obs, _, terminated, truncated, info = game.step(red_action, blue_action, control_steps)
        next_state = next_obs.to_array(GameTeam.RED)
        reward = basic_action_reward(red_action) \
               + nav_reward(info, red_action) \
               + economic_reward(info, red_action) \
               # + game_reward(info, last_info)
        reward = float(reward)
        done = terminated or truncated
        last_info = info

        if train:
            transition_dict["states"].append(state)
            transition_dict["actions"].append(agent.action_to_array(red_action))
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
    deterministic_eval: bool
):
    results = []
    for _ in range(episodes):
        _, result = rollout(
            game,
            agent,
            control_steps=control_steps,
            blue_mode=blue_mode,
            deterministic=deterministic_eval,
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
        robot_type_action=RMUL_ROBOT_TYPE_ACTION,
        device=args.device,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        lmbda=args.lmbda,
        epochs=args.epochs,
        eps=args.eps,
        batch_size=args.ppo_batch_size
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
                args.deterministic_eval
            )
            tqdm.write(
                "[EVAL]\t"
                f"episode={episode}\t"
                f"results={eval_result['wins']}\t"
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


def parse_args():
    parser = argparse.ArgumentParser(description="并行 PPO 训练 RMUL 规则。")

    file_group = parser.add_argument_group('文件配置')
    file_group.add_argument("--base-model", type=str, default=None, help="基础模型路径；不指定时从头训练。")
    file_group.add_argument("--model-dir", type=str, default="models/rmul", help="模型保存目录。")
    file_group.add_argument("--log-dir", type=str, default="logs", help="训练日志保存目录。")

    ctrl_group = parser.add_argument_group('控制配置')
    ctrl_group.add_argument("--control-frequency", type=float, default=2, help="控制频率，用于计算每个决策对应的仿真步数。")
    ctrl_group.add_argument("--blue-mode", choices=["auto", "self", "script"], default="auto", help="蓝方控制模式：auto 根据阶段自动选择，self 使用智能体，script 使用逻辑脚本。")

    train_group = parser.add_argument_group('训练配置')
    train_group.add_argument("--episodes", type=int, default=1000, help="训练总回合数。")
    train_group.add_argument("--save-interval", type=int, default=100, help="模型保存间隔，按训练回合数计算。")
    train_group.add_argument("--eval-interval", type=int, default=100, help="评估间隔，按训练回合数计算。")
    train_group.add_argument("--eval-episodes", type=int, default=20, help="每次评估运行的回合数。")
    train_group.add_argument("--timing-interval", type=int, default=1, help="耗时统计打印间隔，按训练回合数计算；设为 0 则只在结束时打印。")
    train_group.add_argument("--parallel-eval", action="store_true", help="启用并行评估。")
    train_group.add_argument("--deterministic-eval", action="store_true", help="启用确定性策略评估。")

    agent_group = parser.add_argument_group('PPO配置')
    agent_group.add_argument("--device", type=str, default=None, help="训练设备，例如 cpu、cuda 或 cuda:0；不指定时自动选择。")
    agent_group.add_argument("--actor-lr", type=float, default=5e-5, help="Actor 网络学习率。")
    agent_group.add_argument("--critic-lr", type=float, default=5e-4, help="Critic 网络学习率。")
    agent_group.add_argument("--gamma", type=float, default=0.98, help="奖励折扣因子。")
    agent_group.add_argument("--lmbda", type=float, default=0.95, help="GAE 优势估计的 lambda 参数。")
    agent_group.add_argument("--epochs", type=int, default=4, help="每批采样数据上的 PPO 训练轮数。")
    agent_group.add_argument("--eps", type=float, default=0.1, help="PPO clipping 的 epsilon 参数。")
    agent_group.add_argument("--ppo-batch-size", type=int, default=16, help="PPO 更新时的小批量大小。")

    return parser.parse_args()


if __name__ == "__main__":
    torch.set_num_threads(1)
    args = parse_args()
    train(args)
