import pygame
import os
import time
import json
from tqdm import tqdm
from collections import defaultdict

from agents.ppo_agent import PPOAgent
from base.config import env_config
from base.environment import Environment, Action
from visualization.renderer import Renderer
from utils.config.game_config import GameTeam, GameState
from utils.utils import timer

def train(
    num_episodes: int = 1000,
    max_steps: int = env_config.GAME_TIME_LIMIT * env_config.FPS,
    save_interval: int = 50,
    model_dir: str = "models",
    log_dir: str = "logs",
    visualize: bool = False,
):
    """
    训练PPO智能体
    
    Args:
        num_episodes: 训练回合数
        max_steps: 每回合最大步数
        save_interval: 模型保存间隔
        model_dir: 模型保存目录
        log_dir: 日志保存目录
        visualize: 是否启用可视化模式
    """
    # 创建保存目录
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建环境和智能体
    env = Environment()
    state_size = len(env._get_team_state(GameTeam.RED))
    agent = PPOAgent(
        team=GameTeam.RED,
        state_size=state_size,
        field_width=env_config.FIELD_WIDTH,
        field_height=env_config.FIELD_HEIGHT
    )
    
    if visualize:
        # 初始化pygame
        pygame.init()
        renderer = Renderer(env_config)
    
    # 训练记录
    episode_rewards = []
    episode_lengths = []
    
    # 性能统计
    time_stats = defaultdict(list)
    
    # 训练循环
    for episode in tqdm(range(num_episodes), dynamic_ncols=True):
        with timer(time_stats, 'env_reset'):
            env.reset()
        episode_reward = 0
        episode_length = 0
        
        # 记录当前回合的动作序列
        episode_actions = []
        
        for step in range(max_steps):
            with timer(time_stats, 'total_step'):
                # 获取状态
                with timer(time_stats, 'get_state'):
                    state = env._get_team_state(GameTeam.RED)
                
                # 选择动作
                with timer(time_stats, 'act'):
                    red_action = agent.act(state)
                    blue_action = {"BLUE_3_STANDARD": Action(navigation=None, attack=False, target=None)}
                
                # 记录动作
                with timer(time_stats, 'record_action'):
                    frame_action = {
                        'red_action': {
                            robot_id: {
                                'navigation': action.navigation,
                                'attack': action.attack,
                                'target': action.target.value if action.target is not None else None
                            } for robot_id, action in red_action.items()
                        },
                        'blue_action': {
                            robot_id: {
                                'navigation': action.navigation,
                                'attack': action.attack,
                                'target': action.target.value if action.target is not None else None
                            } for robot_id, action in blue_action.items()
                        }
                    }
                    episode_actions.append(frame_action)
                
                # 执行动作
                with timer(time_stats, 'env_step'):
                    env.step(1/env_config.FPS, red_action, blue_action)
                
                # 计算奖励
                with timer(time_stats, 'calculate_reward'):
                    reward = env.calculate_reward(GameTeam.RED, red_action)
                    episode_reward += reward
                
                # 存储轨迹
                with timer(time_stats, 'store_reward'):
                    agent.store_reward(reward, env.game_state_manager.state != GameState.PLAYING)
                
                # 更新步数
                episode_length += 1
                
                # 检查是否结束
                if env.game_state_manager.state != GameState.PLAYING:
                    break
                
                # 可视化模式
                if visualize:
                    renderer.render(env, show_grid=False)
                    pygame.display.flip()
                    time.sleep(0.5)  # 控制渲染速度
        
        # 保存当前回合的动作序列
        if episode % save_interval == 0:
            episode_log = {
                'episode': episode,
                'reward': episode_reward,
                'length': episode_length,
                'game_state': env.game_state_manager.state.value,
                'actions': episode_actions
            }
            with open(os.path.join(log_dir, f'episode_{episode+1}.json'), 'w') as f:
                json.dump(episode_log, f, indent=2)
        
        # 更新策略
        with timer(time_stats, 'update'):
            agent.update()
        
        # 记录训练数据
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 打印训练进度
        tqdm.write(f"回合 {episode + 1}/{num_episodes}")
        tqdm.write(f"总奖励: {episode_reward:.2f}")
        tqdm.write(f"回合长度: {episode_length}")
        tqdm.write(f"剩余时间: {env.game_state_manager.remaining_time:.2f}")
        tqdm.write(f"比赛结果: {env.game_state_manager.state}")
        
        # 打印性能统计
        tqdm.write("\n性能统计 (平均耗时，单位：秒):")
        for key, times in time_stats.items():
            if times:  # 确保有数据
                avg_time = sum(times)
                tqdm.write(f"{key}: {avg_time:.6f}")
        tqdm.write("=" * 50)
        
        # 保存模型
        if (episode + 1) % save_interval == 0:
            agent.save(os.path.join(model_dir, f"ppo_agent_episode_{episode+1}.pt"))
    
    # 保存最终模型
    agent.save(os.path.join(model_dir, "ppo_agent_final.pt"))
    
    if visualize:
        pygame.quit()

if __name__ == "__main__":
    # 设置可视化模式
    VISUALIZE = False  # 设置为True启用可视化
    
    train(num_episodes=1000, visualize=VISUALIZE)
