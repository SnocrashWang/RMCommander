import os
import json
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import random

from agents.ppo_agent import PPOAgent
from base.config import env_config
from base.environment import Action
from base.game import Game
from utils.config.game_config import GameTeam
from utils.config.robot_config import RobotType, ROBOT_ID
from utils.grid_map import world_to_grid
from utils.utils import timer, opposite_position

def train(
    base_model: str = None,
    rival_model: str = None,
    adversarial: bool = False,
    device: str = None,
    model_dir: str = "models",
    log_dir: str = "logs",
    num_episodes: int = 2000,
    max_steps: int = env_config.GAME_TIME_LIMIT * env_config.FPS,
    control_frequency: int = 2,
    save_interval: int = 100,
):
    """
    训练PPO智能体
    
    Args:
        load_model: 预训练模型路径，None表示从头开始训练
        device: 训练设备，None表示自动选择
        model_dir: 模型保存目录
        log_dir: 日志保存目录
        num_episodes: 训练回合数
        max_steps: 每回合最大步数
        save_interval: 模型保存间隔
    """
    # 创建保存目录
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建环境和智能体
    game = Game()
    state_size = game.observation_space.shape[0]
    agent_train = PPOAgent(
        state_dim=state_size,
        device=device
    )
    if adversarial:
        print("对抗训练")
        rival_model = None
    if rival_model is not None:
        agent_test = PPOAgent(
            state_dim=state_size,
            device=device
        )
    # 如果指定了预训练模型，则加载它
    if base_model is not None:
        if os.path.exists(base_model):
            agent_train.load(base_model)
            print(f"已加载预训练模型: {base_model}")
        else:
            print(f"警告: 预训练模型 {base_model} 不存在，将从头开始训练")
    # 如果指定了对手模型，则加载它
    if rival_model is not None:
        if os.path.exists(rival_model):
            agent_test.load(rival_model)
            print(f"已加载对手模型: {rival_model}")
        else:
            print(f"警告: 对手模型 {rival_model} 不存在，将采用随机策略")

    # 计算控制步数
    control_steps = int(game.metadata['render_fps'] // control_frequency)
    
    # 训练循环
    for episode in tqdm(range(num_episodes), dynamic_ncols=True):
        # 性能统计
        time_stats = defaultdict(list)

        obs, info = game.reset(options={"random": True})
        state = obs.to_array()
        init_state = state

        episode_actions = [] # 记录当前回合的动作序列
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}

        for step in range(max_steps // control_steps):
            # 选择动作
            with timer(time_stats, 'act'):
                red_action = agent_train.take_action(obs.to_array(GameTeam.RED), GameTeam.RED)
                if adversarial:
                    # 对抗训练，蓝方使用红方模型
                    blue_action = agent_train.take_action(obs.to_array(GameTeam.BLUE), GameTeam.BLUE)
                elif rival_model is not None:
                    # 蓝方使用对手模型
                    blue_action = agent_test.take_action(obs.to_array(GameTeam.BLUE), GameTeam.BLUE)
                else:
                    blue_action = {id: Action(**action) for id, action in game.action_space.sample().items() if id in ROBOT_ID[GameTeam.BLUE].values()}
                # 翻转蓝方导航点
                for id in blue_action.keys():
                    blue_action[id].navigation_target = opposite_position(blue_action[id].navigation_target, env_config.FIELD_WIDTH, env_config.FIELD_HEIGHT)
            
            # 记录动作
            frame_action = {
                'red_action': {
                    robot_id: {
                        'navigation_target': list(action.navigation_target),
                        'navigation_set': action.navigation_set,
                        'attack_target': action.attack_target
                    } for robot_id, action in red_action.items()
                },
                'blue_action': {
                    robot_id: {
                        'navigation_target': list(action.navigation_target),
                        'navigation_set': action.navigation_set,
                        'attack_target': action.attack_target
                    } for robot_id, action in blue_action.items()
                }
            }
            episode_actions.extend([frame_action] * control_steps)
            
            # 执行动作
            with timer(time_stats, 'env_step'):
                next_obs, reward, terminated, truncated, info = game.step(red_action, blue_action, control_steps)
                next_state = next_obs.to_array()
                done = terminated or truncated

                transition_dict['states'].append(state)
                transition_dict['actions'].append(red_action["RED_3_STANDARD"].to_array())
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)

                obs = next_obs
                state = next_state
            
            # 检查是否结束
            if done:
                break
        
        # 保存当前回合的动作序列
        if (episode + 1) % (save_interval) == 0 or episode == 0:
            episode_log = {
                'episode': episode,
                'length': len(transition_dict['states']),
                'reward': sum(transition_dict['rewards']),
                'game_state': game.env.game_state.value,
                'init_state': init_state.tolist(),
                'actions': episode_actions
            }
            with open(os.path.join(log_dir, f'episode_{time_tag}_{episode+1}.json'), 'w') as f:
                json.dump(episode_log, f, indent=2)
        
        # 更新策略
        with timer(time_stats, 'update'):
            agent_train.update([transition_dict])
        
        # 打印训练进度
        tqdm.write(f"回合 {episode + 1}/{num_episodes}")
        tqdm.write(f"回合长度: {len(transition_dict['states'])}")
        tqdm.write(f"剩余时间: {info['remaining_time']:.3f}s")
        tqdm.write(f"平均奖励: {sum(transition_dict['rewards'])/len(transition_dict['rewards']):.3f}")
        tqdm.write(f"比赛结果: {game.env.game_state}")
        
        # # 打印性能统计
        # tqdm.write("\n性能统计:")
        # for key, times in time_stats.items():
        #     if times:  # 确保有数据
        #         avg_time = sum(times)
        #         tqdm.write(f"{key}: {avg_time:.6f}s")

        tqdm.write("=" * 50)
        
        # 保存模型
        if (episode + 1) % save_interval == 0:
            agent_train.save(os.path.join(model_dir, f"ppo_agent_{time_tag}_episode_{episode+1}.pt"))


if __name__ == "__main__":
    # 设置预训练模型路径（如果需要从预训练模型继续训练）
    # BASE_MODEL = "models/ppo_agent_20250709_160428_episode_200.pt"
    BASE_MODEL = None
    # RIVAL_MODEL = "models/ppo_agent_20250709_160428_episode_200.pt"
    RIVAL_MODEL = None
    
    # 对抗训练
    ADVERSARIAL = True

    # 设置训练设备（None表示自动选择，'cuda'表示使用GPU，'cpu'表示使用CPU）
    DEVICE = None
    
    train(
        base_model=BASE_MODEL,
        rival_model=RIVAL_MODEL,
        adversarial=ADVERSARIAL,
        device=DEVICE
    )
