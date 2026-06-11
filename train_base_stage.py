import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime
from typing import Dict, Tuple
from tqdm import tqdm

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from rules.base.config import env_config
from rules.base.environment import ActionBase
from rules.base.game import Game
from rules.base.config.robot_config import BASE_ROBOT_TYPE_ACTION
from utils.config.game_config import GameState, GameTeam
from utils.config.robot_config import RobotType
from utils.grid_map import world_to_grid
from utils.utils import attack_sight_clear, calc_distance, pos_real2norm


RED_ID = "RED_3_STANDARD"
BLUE_ID = "BLUE_3_STANDARD"
SCRIPT_SWITCH_INTERVAL_RANGE = (3.0, 5.0)


def force_stage_a_action(action: Dict[str, ActionBase]) -> Dict[str, ActionBase]:
    """Stage A 中用于隔离导航学习的动作钳制。"""
    robot_action = action[RED_ID]
    robot_action.navigation_set = 1
    robot_action.attack_target = RobotType.STANDARD_3.value
    robot_action.spin = 0
    return action


class ScriptBlueController:
    """Stage C 脚本蓝方：周期性随机切换自旋和导航点。"""

    def __init__(self, game: Game):
        self.game = game
        self.spin_enabled = False
        self.nav_world = self._sample_blue_nav_world()
        self.next_spin_switch_time = 0.0
        self.next_nav_switch_time = 0.0
        self.spin_switches = 0
        self.nav_switches = 0

    def _sample_next_switch_time(self, elapsed_time: float) -> float:
        return elapsed_time + random.uniform(*SCRIPT_SWITCH_INTERVAL_RANGE)

    def _sample_blue_nav_world(self) -> Tuple[float, float]:
        return sample_position(self.game, (0.5, 4.5), (0.5, 4.5))

    def make_action(self, elapsed_time: float) -> Dict[str, ActionBase]:
        if elapsed_time >= self.next_spin_switch_time:
            self.spin_enabled = bool(random.getrandbits(1))
            self.next_spin_switch_time = self._sample_next_switch_time(elapsed_time)
            self.spin_switches += 1

        if elapsed_time >= self.next_nav_switch_time:
            self.nav_world = self._sample_blue_nav_world()
            # self.next_nav_switch_time = self._sample_next_switch_time(elapsed_time)
            self.next_nav_switch_time = elapsed_time + 1000.0   # 导航点保持不变，直到 episode 结束
            self.nav_switches += 1

        return {
            BLUE_ID: ActionBase(
                navigation_target_norm=pos_real2norm(self.nav_world, (env_config.FIELD_WIDTH, env_config.FIELD_HEIGHT)),
                navigation_set=1,
                attack_target=RobotType.STANDARD_3.value,
                spin=int(self.spin_enabled),
            )
        }


def make_blue_action(
    mode: str,
    script_blue_controller: ScriptBlueController = None,
    elapsed_time: float = 0.0,
) -> Dict[str, ActionBase]:
    """脚本蓝方：静止或只进行固定攻击。"""
    if mode == "idle":
        return {
            BLUE_ID: ActionBase(
                navigation_set=0,
                attack_target=RobotType.NONE.value,
                spin=0,
            )
        }
    elif mode == "attack":
        return {
            BLUE_ID: ActionBase(
                navigation_set=0,
                attack_target=RobotType.STANDARD_3.value,
                spin=1,
            )
        }
    elif mode == "nav_attack":
        if script_blue_controller is None:
            return {BLUE_ID: ActionBase()}
        return script_blue_controller.make_action(elapsed_time)
    return {BLUE_ID: ActionBase()}


def robot_snapshot(game: Game) -> Tuple[Tuple[float, float], Tuple[float, float], int, int]:
    red = game.env.get_robot(RED_ID)
    blue = game.env.get_robot(BLUE_ID)
    return red.get_position(), blue.get_position(), red.hp, blue.hp


def is_valid_position(game: Game, position: Tuple[float, float]) -> bool:
    try:
        col, row = world_to_grid(position)
        return not game.env.get_robot(RED_ID)._grid_map.is_blocked(col, row)
    except ValueError:
        return False


def sample_position(game: Game, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> Tuple[float, float]:
    for _ in range(1000):
        position = (random.uniform(*x_range), random.uniform(*y_range))
        if game is not None:
            if is_valid_position(game, position):
                return position
        else:
            return position
    raise RuntimeError("failed to sample a valid start position")


def apply_random_start(game: Game):
    """在可行区域内随机放置双方，用来减少固定开局导致的过拟合。"""
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


def reward(game: Game, stage, action: ActionBase, before, after) -> float:
    """共享诊断奖励：只描述接近、伤害、视野和胜负，不追求最终战术完备。"""
    red_pos_before, blue_pos_before, red_hp_before, blue_hp_before = before
    red_pos_after, blue_pos_after, red_hp_after, blue_hp_after = after

    self_damage = max(0, blue_hp_before - blue_hp_after)
    enemy_damage = max(0, red_hp_before - red_hp_after)
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
    if stage == "a":
        reward += 1.0 * distance_gain
        reward += 0.03 if has_los else 0.0
        reward += 0.10 * self_damage
        reward -= 0.10 * enemy_damage
        reward -= 0.05   # 时间惩罚
    if stage == "b":
        reward += 0.1 * self_damage
        # 鼓励自旋与受到攻击相绑定
        if has_los:
            reward += 0.02
            if enemy_damage > 0:
                if action.spin == 1:
                    reward += 0.1
                    # 受击时鼓励原地自旋
                    if action.navigation_set == 0:
                        reward += 0.05
                else:
                    reward -= 0.1
                    # 或者不自旋直接逃跑
                    if action.navigation_set == 1:
                        reward += 0.12
        else:
            if action.spin == 0:
                reward += 0.05
        reward -= 0.1
    if stage == "c":
        reward -= 0.05
    if stage == "d":
        reward -= 0.05

    if game.env.game_state == GameState.RED_TEAM_WIN:
        reward += 100.0
    elif game.env.game_state == GameState.BLUE_TEAM_WIN:
        reward -= 100.0
    return reward


def navigation_validity_reward(game: Game, action: ActionBase) -> float:
    """复用 base 奖励中的导航点可行性判断：可行给奖，不可行惩罚。"""
    if action.navigation_set != 1:
        return 0.01

    navigation_target = (np.array(action.navigation_target_norm) + 1) * np.array([
        env_config.FIELD_WIDTH,
        env_config.FIELD_HEIGHT,
    ]) / 2

    try:
        col, row = world_to_grid(navigation_target)
        is_blocked = game.env.get_robot(RED_ID)._grid_map.is_blocked(col, row)
    except ValueError:
        is_blocked = True

    # 合法导航点只给很小奖励，非法点仍然明显惩罚。
    # 否则模型可以靠每步输出任意合法点获得很高回报，却完全不接敌。
    return -1.0 if is_blocked else 0.05


def action_head_reward(action: ActionBase) -> float:
    """轻量鼓励基础动作头，避免奖励过大压过真正的战斗结果。"""
    reward = 0.0
    reward += 0.10 if action.navigation_set == 1 else -0.10
    reward += 0.30 if action.attack_target == RobotType.STANDARD_3.value else -0.30
    return reward


def rollout(
    agent: PPOAgent,
    control_steps: int,
    stage: str,
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
    spin_counts = Counter()
    total_reward = 0.0
    steps = 0
    damage_done = 0
    script_blue_controller = ScriptBlueController(game) if blue_mode == "nav_attack" else None

    max_decision_steps = env_config.GAME_TIME_LIMIT * env_config.FPS // control_steps + 1
    for _ in range(max_decision_steps):
        elapsed_time = env_config.GAME_TIME_LIMIT - game.env._remaining_time
        red_action = agent.take_action(state, GameTeam.RED, deterministic=deterministic)
        if blue_mode == "self":
            # 蓝方也使用当前 agent，但输入蓝方视角观测。
            blue_action = agent.take_action(obs.to_array(GameTeam.BLUE), GameTeam.BLUE, deterministic=deterministic)
            # blue_action = mirror_navigation_target_actions(blue_action)
        else:
            blue_action = make_blue_action(blue_mode, script_blue_controller, elapsed_time)

        before = robot_snapshot(game)
        next_obs, _, terminated, truncated, info = game.step(red_action, blue_action, control_steps)
        after = robot_snapshot(game)
        step_reward = reward(game, stage, red_action[RED_ID], before, after)
        step_reward += navigation_validity_reward(game, red_action[RED_ID])
        if stage == "a":
            step_reward += action_head_reward(red_action[RED_ID])
        elif stage == "b":
            pass
        elif stage == "c":
            pass
        elif stage == "d":
            pass
        next_state = next_obs.to_array(GameTeam.RED)
        done = terminated or truncated

        executed_action = agent.action_to_array(red_action)
        target_counts[red_action[RED_ID].attack_target] += 1
        nav_counts[red_action[RED_ID].navigation_set] += 1
        spin_counts[red_action[RED_ID].spin] += 1
        damage_done += max(0, before[3] - after[3])

        if train:
            transition_dict["states"].append(state)
            transition_dict["actions"].append(executed_action)
            transition_dict["next_states"].append(next_state)
            transition_dict["rewards"].append(step_reward)
            transition_dict["dones"].append(done)

        obs = next_obs
        state = next_state
        total_reward += step_reward
        steps += 1
        if done:
            break

    result = {
        "game_state": game.env.game_state.name,
        "remaining_time": game.env._remaining_time,
        "elapsed_time": game.env.total_time - game.env._remaining_time,
        "red_hp": game.env.get_robot(RED_ID).hp,
        "blue_hp": game.env.get_robot(BLUE_ID).hp,
        "damage_done": damage_done,
        "steps": steps,
        "reward": total_reward,
        "red_action_statics": {
            "target_counts": dict(target_counts),
            "nav_counts": dict(nav_counts),
            "spin_counts": dict(spin_counts),
        }
    }
    game.close()
    return transition_dict, result


def evaluate(
    agent: PPOAgent,
    episodes: int,
    control_steps: int,
    blue_mode: str,
    random_start: bool,
):
    results = []
    for _ in range(episodes):
        _, result = rollout(
            agent,
            control_steps=control_steps,
            stage="",
            blue_mode=blue_mode,
            random_start=random_start,
            deterministic=True,
            train=False,
        )
        results.append(result)
    wins = Counter(result["game_state"] for result in results)
    return {
        "episodes": episodes,
        "wins": dict(sorted(wins.items())),
        "avg_reward": float(np.mean([r["reward"] for r in results])),
        "avg_damage_done": float(np.mean([r["damage_done"] for r in results])),
        "avg_steps": float(np.mean([r["steps"] for r in results])),
        "avg_elapsed_time": float(np.mean([r["elapsed_time"] for r in results])),
        "avg_red_hp": float(np.mean([r["red_hp"] for r in results])),
        "avg_blue_hp": float(np.mean([r["blue_hp"] for r in results])),
    }


def resolve_stage_defaults(args):
    """根据阶段补默认配置；命令行显式传入的选项仍可覆盖。"""
    if args.random_start is None:
        args.random_start = args.stage == "c"
    if args.stage == "a":
        if args.blue_mode == "auto":
            args.blue_mode = "idle"
    elif args.stage == "b":
        if args.blue_mode == "auto":
            # args.blue_mode = "nav_attack"
            args.blue_mode = random.choice(["attack", "nav_attack", "self"])
    elif args.stage == "c":
        if args.blue_mode == "auto":
            args.blue_mode = random.choice(["attack", "nav_attack", "self"])
    elif args.stage == "d":
        if args.blue_mode == "auto":
            args.blue_mode = "self"
    else:
        raise ValueError(f"unknown stage: {args.stage}")


def train(args):
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_agent_{time_tag}_stage_{args.stage}"

    game = Game()
    agent = PPOAgent(
        state_dim=game.observation_space.shape[0],
        robot_type_action=BASE_ROBOT_TYPE_ACTION,
        device=args.device,
    )
    game.close()
    if args.base_model:
        agent.load(args.base_model)

    control_steps = int(Game.metadata["render_fps"] // args.control_frequency)
    history = []

    for episode in tqdm(range(1, args.episodes + 1), dynamic_ncols=True):
        resolve_stage_defaults(args)
        transition_dict, result = rollout(
            agent,
            control_steps=control_steps,
            stage=args.stage,
            blue_mode=args.blue_mode,
            random_start=args.random_start,
            deterministic=False,
            train=True,
        )
        if transition_dict["states"]:
            agent.update(transition_dict)
        history.append(result)

        if episode == 1 or episode % args.eval_interval == 0:
            eval_result = evaluate(
                agent,
                args.eval_episodes,
                control_steps,
                args.blue_mode,
                args.random_start,
            )
            tqdm.write(
                f"episode={episode}\t"
                f"eval_results={eval_result['wins']}\t"
                f"avg_damage={eval_result['avg_damage_done']:.2f}\t"
                f"avg_time={eval_result['avg_elapsed_time']:.2f}s\t"
                f"avg_blue_hp={eval_result['avg_blue_hp']:.2f}"
            )
            history.append({"episode": episode, "eval": eval_result})

        if args.save_interval > 0 and episode % args.save_interval == 0:
            # 诊断训练不只看最后模型，中间 checkpoint 往往更有参考价值。
            checkpoint_path = os.path.join(args.model_dir, f"{run_name}_episode_{episode}.pt")
            agent.save(checkpoint_path)

    model_path = os.path.join(args.model_dir, f"{run_name}.pt")
    log_path = os.path.join(args.log_dir, f"stage_{args.stage}_{time_tag}.json")
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
        agent,
        args.eval_episodes,
        control_steps,
        args.blue_mode,
        args.random_start,
    )
    print(json.dumps({"model_path": model_path, "log_path": log_path, "final_eval": final_eval}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Staged PPO training for base rule.")
    parser.add_argument("--stage", choices=["a", "b", "c", "d"], default="a")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--control-frequency", type=float, default=2)
    parser.add_argument("--blue-mode", choices=["auto", "self", "idle", "attack", "nav_attack"], default="auto")
    parser.add_argument("--random-start", action=argparse.BooleanOptionalAction, default=None, help="是否在每局开始时随机放置双方")
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default="models/base")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--device", type=str, default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
