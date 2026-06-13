import os
import json
import math
import torch
import random
import argparse
import statistics
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple
from datetime import datetime
from collections import Counter

from config import CURRENT_GAME
from agents.ppo_agent import PPOAgent
from utils.config.game_config import GameState, GameTeam, GameType
from utils.config.robot_config import RobotType
from utils.buff import Buff

if CURRENT_GAME == GameType.BASE:
    from rules.base.game import Game
    from rules.base.curriculum import *
    from rules.base.environment import ActionBase as Action
    from rules.base.config.env_config import EnvConfigBase as EnvConfig
    from rules.base.config.robot_config import BASE_ROBOT_TYPE_ACTION as ROBOT_TYPE_ACTION
elif CURRENT_GAME == GameType.RMUL:
    from rules.rmul.game import GameRMUL as Game
    from rules.rmul.environment import ActionRMUL as Action
    from rules.rmul.config.env_config import EnvConfigRMUL as EnvConfig
    from rules.rmul.config.robot_config import RMUL_ROBOT_TYPE_ACTION as ROBOT_TYPE_ACTION


def statistical_analysis(data):
    if not data:
        return {}

    field_names = data[0].keys()
    field_data = {field: [] for field in field_names}
    
    for record in data:
        for field in field_names:
            if field in record:
                field_data[field].append(record[field])

    result = {}
    for field, values in field_data.items():
        if not values:
            continue
        
        # 检查第一个值的类型来决定统计方式
        first_value = values[0]
        # 特殊枚举类统计
        if isinstance(first_value, GameState):
            enum_counts = {}
            for v in values:
                if isinstance(v, GameState):
                    enum_counts[v.name] = enum_counts.get(v.name, 0) + 1
            # 按枚举值排序
            sorted_counts = dict(sorted(enum_counts.items(), key=lambda x: x[0]))
            result[f"{field}_cnt"] = sorted_counts
        # 数值类型统计（int或float）
        elif isinstance(first_value, (int, float)):
            result[f"{field}_mean"] = statistics.mean(values)

    return result


def rollout(
    game: Game,
    agent: PPOAgent,
    control_frequency: int,
    curriculum_stage: int,
    deterministic: bool = False,
    train: bool = False,
):
    game.set_curriculum(curriculum_stage, random_env=True, random_obstacles=True, random_robots=True)
    obs, info = game.reset()
    state = obs.to_array(GameTeam.RED)
    transition_dict = {"states": [], "actions": [], "next_states": [], "rewards": [], "dones": []}
    total_reward = 0.0
    steps = 0

    control_steps = max(1, EnvConfig.fps // control_frequency)
    max_decision_steps = EnvConfig.game_time_limit * control_frequency + 1
    for _ in range(max_decision_steps):
        red_action = agent.take_action(state, GameTeam.RED, deterministic=deterministic)

        next_obs, reward, terminated, truncated, info = game.step(red_action, control_steps=control_steps)
        next_state = next_obs.to_array(GameTeam.RED)
        done = terminated or truncated

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

    return transition_dict, info


def evaluate(
    game: Game,
    agent: PPOAgent,
    episodes: int,
    control_frequency: int,
    curriculum_stage: int,
    deterministic_eval: bool
):
    results = []
    for _ in range(episodes):
        _, result = rollout(
            game,
            agent,
            control_frequency=control_frequency,
            curriculum_stage=curriculum_stage,
            deterministic=deterministic_eval,
            train=False,
        )
        results.append(result)
    return statistical_analysis(results)


def train(args):
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_agent_{time_tag}"

    game = Game(curriculum_list=[CurriculumBaseMovement()])
    agent = PPOAgent(
        state_dim=game.observation_space.shape[0],
        robot_type_action=ROBOT_TYPE_ACTION,
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

    history = []
    for episode in tqdm(range(1, args.episodes + 1), dynamic_ncols=True):
        transition_dict, result = rollout(
            game,
            agent,
            control_frequency=args.control_frequency,
            curriculum_stage=args.curriculum_stage,
            deterministic=False,
            train=True,
        )
        if transition_dict["states"]:
            agent.update(transition_dict)
        history.append({"episode": episode, **result})

        if (args.base_model and episode == 1) or episode % args.eval_interval == 0:
            eval_result = evaluate(
                game,
                agent,
                args.eval_episodes,
                args.control_frequency,
                args.curriculum_stage,
                args.deterministic_eval
            )
            eval_result_info = ",\t".join([
                f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}={v}" 
                for k, v in eval_result.items()
            ])
            tqdm.write(
                "[EVAL]\t"
                f"episode={episode}\t"
                f"{eval_result_info}"
            )
            history.append({"episode": episode, "eval": eval_result})

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
        args.control_frequency,
        args.curriculum_stage,
        args.deterministic_eval
    )
    game.close()
    print(json.dumps({"model_path": model_path, "log_path": log_path, "final_eval": final_eval}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="并行 PPO 训练 RMUL 规则。")

    file_group = parser.add_argument_group('文件配置')
    file_group.add_argument("--base-model", type=str, default=None, help="基础模型路径；不指定时从头训练。")
    file_group.add_argument("--model-dir", type=str, default="models/rmul", help="模型保存目录。")
    file_group.add_argument("--log-dir", type=str, default="logs", help="训练日志保存目录。")

    env_group = parser.add_argument_group('环境配置')
    env_group.add_argument("--control-frequency", type=float, default=2, help="控制频率，用于计算每个决策对应的仿真步数。")
    env_group.add_argument("--curriculum-stage", type=int, default=0, help="课程学习阶段")

    train_group = parser.add_argument_group('训练配置')
    train_group.add_argument("--episodes", type=int, default=1000, help="训练总回合数。")
    train_group.add_argument("--save-interval", type=int, default=100, help="模型保存间隔，按训练回合数计算。")
    train_group.add_argument("--eval-interval", type=int, default=100, help="评估间隔，按训练回合数计算。")
    train_group.add_argument("--eval-episodes", type=int, default=20, help="每次评估运行的回合数。")
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
