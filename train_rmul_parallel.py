import argparse
import json
import multiprocessing
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from agents.ppo_agent import PPOAgent
from rules.rmul.config.robot_config import RMUL_ROBOT_TYPE_ACTION
from rules.rmul.game import GameRMUL as Game
from train_rmul import evaluate, rollout


def resolve_blue_mode(blue_mode: str) -> str:
    if blue_mode == "auto":
        return random.choice(["self", "script"])
    return blue_mode


def rollout_worker(agent: PPOAgent, control_steps: int, blue_mode: str, seed: int = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    game = Game()
    try:
        worker_blue_mode = resolve_blue_mode(blue_mode)
        transition_dict, result = rollout(
            game=game,
            agent=agent,
            control_steps=control_steps,
            blue_mode=worker_blue_mode,
            deterministic=False,
            train=True,
        )
        return transition_dict, result
    finally:
        game.close()


def eval_worker(agent: PPOAgent, control_steps: int, blue_mode: str, seed: int = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    game = Game()
    try:
        _, result = rollout(
            game=game,
            agent=agent,
            control_steps=control_steps,
            blue_mode=blue_mode,
            deterministic=True,
            train=False,
        )
        return result
    finally:
        game.close()


def evaluate_parallel(agent: PPOAgent, episodes: int, control_steps: int, blue_mode: str, num_workers: int):
    results = []
    if episodes <= 0:
        return {
            "episodes": 0,
            "wins": {},
            "avg_elapsed_time": 0.0,
            "avg_red_progress": 0.0,
            "avg_blue_progress": 0.0,
            "avg_reward": 0.0,
        }

    eval_blue_mode = "script" if blue_mode == "auto" else blue_mode
    with ProcessPoolExecutor(max_workers=max(1, num_workers)) as executor:
        futures = [
            executor.submit(
                eval_worker,
                agent,
                control_steps,
                eval_blue_mode,
                random.randrange(2**31),
            )
            for _ in range(episodes)
        ]
        for future in as_completed(futures):
            results.append(future.result())

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
    run_name = f"ppo_agent_parallel_{time_tag}"

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
    game.close()

    if args.base_model:
        agent.load(args.base_model)

    control_steps = max(1, int(Game.metadata["render_fps"] // args.control_frequency))
    history = []

    episode_bar = tqdm(
        range(1, args.episodes + 1),
        desc="episodes",
        position=0,
        dynamic_ncols=True,
    )
    for episode in episode_bar:
        transition_list = []
        result_list = []

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(
                    rollout_worker,
                    agent,
                    control_steps,
                    args.blue_mode,
                    random.randrange(2**31),
                )
                for _ in range(args.rollout_batch_size)
            ]

            batch_bar = tqdm(
                total=len(futures),
                desc=f"rollouts",
                position=1,
                leave=False,
                dynamic_ncols=True,
            )
            with batch_bar:
                for future in as_completed(futures):
                    try:
                        transition_dict, result = future.result()
                        if transition_dict["states"]:
                            transition_list.append(transition_dict)
                        result_list.append(result)
                    except Exception:
                        import traceback

                        traceback.print_exc()
                    finally:
                        batch_bar.update(1)

        wins = Counter(result["game_state"] for result in result_list)
        avg_reward = float(np.mean([result["reward"] for result in result_list])) if result_list else 0.0
        avg_elapsed_time = float(np.mean([result["elapsed_time"] for result in result_list])) if result_list else 0.0
        avg_red_progress = float(np.mean([result["red_progress"] for result in result_list])) if result_list else 0.0
        avg_blue_progress = float(np.mean([result["blue_progress"] for result in result_list])) if result_list else 0.0

        episode_record = {
            "episode": episode,
            "wins": dict(sorted(wins.items())),
            "avg_elapsed_time": avg_elapsed_time,
            "avg_red_progress": avg_red_progress,
            "avg_blue_progress": avg_blue_progress,
            "avg_reward": avg_reward,
            "rollouts": result_list,
        }
        history.append(episode_record)

        tqdm.write(
            "[ROLLOUT]\t"
            f"episode={episode}\t"
            f"results={episode_record['wins']}\t"
            f"avg_time={avg_elapsed_time:.2f}s\t"
            f"avg_progress={avg_red_progress:.2f}/{avg_blue_progress:.2f}\t"
            f"avg_reward={avg_reward:.2f}"
        )

        if transition_list:
            agent.update_multi_rollout(transition_list)

        if (args.base_model and episode == 1) or episode % args.eval_interval == 0:
            if args.parallel_eval:
                eval_result = evaluate_parallel(
                    agent,
                    args.eval_episodes,
                    control_steps,
                    args.blue_mode,
                    args.num_workers,
                )
            else:
                eval_game = Game()
                try:
                    eval_result = evaluate(
                        eval_game,
                        agent,
                        args.eval_episodes,
                        control_steps,
                        args.blue_mode if args.blue_mode != "auto" else "script",
                    )
                finally:
                    eval_game.close()
            tqdm.write(
                "[EVAL]\t"
                f"episode={episode}\t"
                f"results={eval_result['wins']}\t"
                f"avg_time={eval_result['avg_elapsed_time']:.2f}s\t"
                f"avg_progress={eval_result['avg_red_progress']:.2f}/{eval_result['avg_blue_progress']:.2f}\t"
                f"avg_reward={eval_result['avg_reward']:.2f}"
            )
            history.append({"episode": episode, "eval": eval_result})

        if args.save_interval > 0 and episode % args.save_interval == 0:
            checkpoint_path = os.path.join(args.model_dir, f"{run_name}_episode_{episode}.pt")
            agent.save(checkpoint_path)

    model_path = os.path.join(args.model_dir, f"{run_name}.pt")
    log_path = os.path.join(args.log_dir, f"rmul_parallel_{time_tag}.json")
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

    final_game = Game()
    try:
        final_eval = evaluate(
            final_game,
            agent,
            args.eval_episodes,
            control_steps,
            args.blue_mode if args.blue_mode != "auto" else "script",
        )
    finally:
        final_game.close()
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
    train_group.add_argument("--episodes", type=int, default=100, help="训练总回合数。")
    train_group.add_argument("--rollout-batch-size", type=int, default=4, help="每次 PPO 更新前收集的 rollout 数量。")
    train_group.add_argument("--num-workers", type=int, default=2, help="并行采样的工作进程数量。")
    train_group.add_argument("--save-interval", type=int, default=5, help="模型保存间隔，按训练回合数计算。")
    train_group.add_argument("--eval-interval", type=int, default=5, help="评估间隔，按训练回合数计算。")
    train_group.add_argument("--eval-episodes", type=int, default=20, help="每次评估运行的回合数。")
    train_group.add_argument("--parallel-eval", action="store_true", help="启用并行评估。")

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
    multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()
    train(args)
