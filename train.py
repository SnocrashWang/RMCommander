import os
import time
import pygame
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from rl.rl_env import RLEnvironment
from rl.agents.ppo_agent import PPOAgent
from visualization.renderer import Renderer
from base.config import env_config

def train(
    episodes: int = 1000,
    max_steps: int = 1000,
    save_interval: int = 100,
    model_dir: str = "models",
    visualize: bool = False,
    render_delay: float = 0.5  # 渲染延迟时间（秒）
):
    """
    训练PPO智能体
    
    Args:
        episodes: 训练回合数
        max_steps: 每回合最大步数
        save_interval: 模型保存间隔
        model_dir: 模型保存目录
        visualize: 是否启用可视化模式
        render_delay: 渲染延迟时间（秒）
    """
    # 创建保存模型的目录
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建环境和智能体
    env = RLEnvironment()
    state_size = env.state_size
    action_space = env.action_space
    agent = PPOAgent(state_size, action_space)
    
    if visualize:
        # 初始化pygame
        pygame.init()
        # 计算屏幕尺寸
        screen_width = int(env_config.FIELD_WIDTH * env_config.SCALE)
        screen_height = int(env_config.FIELD_HEIGHT * env_config.SCALE)
        renderer = Renderer(screen_width, screen_height)
    
    # 训练记录
    episode_rewards = []
    episode_lengths = []
    win_rates = []
    wins = 0
    
    for episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            print(f"训练循环收到奖励: {reward}")  # 调试信息
            
            # 存储奖励
            agent.store_reward(reward, done)
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
            print(f"当前回合累计奖励: {episode_reward}")  # 调试信息
            steps += 1
            
            # 可视化模式
            if visualize:
                env_state = env.get_game_state()
                renderer.render(env_state, show_grid=False)
                time.sleep(render_delay)  # 控制渲染速度
            
            # 如果回合结束，更新策略
            if done:
                if info['game_state'].game_state.value == 2:  # 红方胜利
                    wins += 1
                agent.update()
                break

            time.sleep(0.5)
            print("-" * 50)
        
        # 记录训练数据
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        win_rates.append(wins / (episode + 1))
        
        # 打印训练进度
        print(f"回合 {episode + 1}/{episodes}")
        print(f"总奖励: {episode_reward:.2f}")
        print(f"回合长度: {steps}")
        print(f"胜率: {win_rates[-1]:.2%}")
        print("-" * 50)
        
        # 定期保存模型
        if (episode + 1) % save_interval == 0:
            agent.save(os.path.join(model_dir, f"ppo_agent_episode_{episode + 1}.pth"))
    
    # 保存最终模型
    agent.save(os.path.join(model_dir, "ppo_agent_final.pth"))
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    # 绘制奖励曲线
    plt.subplot(131)
    plt.plot(episode_rewards)
    plt.title('回合奖励')
    plt.xlabel('回合')
    plt.ylabel('奖励')
    
    # 绘制回合长度曲线
    plt.subplot(132)
    plt.plot(episode_lengths)
    plt.title('回合长度')
    plt.xlabel('回合')
    plt.ylabel('步数')
    
    # 绘制胜率曲线
    plt.subplot(133)
    plt.plot(win_rates)
    plt.title('胜率')
    plt.xlabel('回合')
    plt.ylabel('胜率')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "training_curves.png"))
    plt.close()

if __name__ == "__main__":
    # 设置可视化模式
    VISUALIZE = False  # 设置为True启用可视化
    RENDER_DELAY = 0.5  # 渲染延迟时间（秒）
    
    train(visualize=VISUALIZE, render_delay=RENDER_DELAY)
