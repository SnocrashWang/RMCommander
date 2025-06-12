import os
import time
import pygame
import numpy as np
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from agents.ppo_agent import PPOAgent
from base.config import env_config
from base.environment import Environment, Action
from visualization.renderer import Renderer
from utils.config.game_config import GameTeam, GameState

def train(
    num_episodes: int = 1000,
    max_steps: int = 1000,
    save_interval: int = 100,
    model_dir: str = "models",
    visualize: bool = False,
    render_delay: float = 0.5  # 渲染延迟时间（秒）
):
    """
    训练PPO智能体
    
    Args:
        num_episodes: 训练回合数
        max_steps: 每回合最大步数
        save_interval: 模型保存间隔
        model_dir: 模型保存目录
        visualize: 是否启用可视化模式
        render_delay: 渲染延迟时间（秒）
    """
    # 创建保存模型的目录
    os.makedirs(model_dir, exist_ok=True)
    
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
    win_rates = []
    wins = 0
    
    # 训练循环
    for episode in tqdm(range(num_episodes)):
        env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # 获取状态
            state = env._get_team_state(GameTeam.RED)
            
            # 选择动作
            red_action = agent.act(state)
            blue_action = {"BLUE_3_STANDARD": Action(navigation=None, attack=False, target=None)}
            
            # 执行动作
            env.step(1/env_config.FPS, red_action, blue_action)
            
            # 计算奖励
            reward = env.calculate_reward(GameTeam.RED)
            episode_reward += reward
            
            # 存储轨迹
            agent.store_reward(reward, env.game_state_manager.state != GameState.PLAYING)
            
            # 更新步数
            episode_length += 1
            
            # 检查是否结束
            if env.game_state_manager.state != GameState.PLAYING:
                if env.game_state_manager.state == GameState.RED_WIN:
                    wins += 1
                break
            
            # 可视化模式
            if visualize:
                renderer.render(env, show_grid=False)
                time.sleep(render_delay)  # 控制渲染速度
            
            time.sleep(0.5)
            print(red_action)
            print(reward)
            print("-" * 50)
        
        # 更新策略
        agent.update()
        
        # 记录训练数据
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        win_rates.append(wins / (episode + 1))
        
        # 打印训练进度
        print(f"回合 {episode + 1}/{num_episodes}")
        print(f"总奖励: {episode_reward:.2f}")
        print(f"回合长度: {episode_length}")
        print(f"胜率: {win_rates[-1]:.2%}")
        print("-" * 50)
        
        # 保存模型
        if (episode + 1) % save_interval == 0:
            agent.save(os.path.join(model_dir, f"ppo_agent_episode_{episode+1}.pt"))
    
    # 保存最终模型
    agent.save(os.path.join(model_dir, "ppo_agent_final.pt"))
    
    # # 绘制训练曲线
    # plt.figure(figsize=(12, 4))
    
    # plt.subplot(131)
    # plt.plot(episode_rewards)
    # plt.title('Episode Rewards')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    
    # plt.subplot(132)
    # plt.plot(episode_lengths)
    # plt.title('Episode Lengths')
    # plt.xlabel('Episode')
    # plt.ylabel('Length')
    
    # plt.subplot(133)
    # plt.plot(win_rates)
    # plt.title('Win Rates')
    # plt.xlabel('Episode')
    # plt.ylabel('Win Rate')
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(model_dir, 'training_curves.png'))
    # plt.close()

if __name__ == "__main__":
    # 设置可视化模式
    VISUALIZE = True  # 设置为True启用可视化
    RENDER_DELAY = 0.5  # 渲染延迟时间（秒）
    
    train(num_episodes=1000, visualize=VISUALIZE, render_delay=RENDER_DELAY)
