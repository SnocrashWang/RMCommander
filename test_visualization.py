import pygame
import sys
import time
import torch
from rl.rl_env import RLEnvironment
from rl.agents.dqn_agent import DQNAgent
from visualization.renderer import Renderer
from base.config import env_config

def main():
    # 初始化pygame
    pygame.init()

    # 计算屏幕尺寸
    screen_width = int(env_config.FIELD_WIDTH * env_config.SCALE)
    screen_height = int(env_config.FIELD_HEIGHT * env_config.SCALE)

    # 创建环境和渲染器
    env = RLEnvironment()
    renderer = Renderer(screen_width, screen_height)

    # 创建智能体并加载模型
    state_size = 11
    agent = DQNAgent(
        state_size=state_size,
        action_space=env.action_space,
        epsilon=0.0  # 设置为0，使用确定性策略
    )
    agent.load("models/dqn_agent_final.pth")  # 加载训练好的模型

    # 创建时钟对象
    clock = pygame.time.Clock()

    # 重置环境
    state, info = env.reset()
    running = True

    while running:
        frame_start = time.perf_counter()

        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    state, info = env.reset()
                elif event.key == pygame.K_ESCAPE:
                    running = False

        # 智能体选择动作
        action = agent.act(state)

        # 执行动作
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state

        # 获取环境状态
        env_state = env.env.get_game_state()

        # 渲染环境
        renderer.render(env_state, env.env.obstacles)

        # 更新显示
        pygame.display.flip()

        # 控制帧率
        dt = 1 / env_config.FPS
        time_cost = time.perf_counter() - frame_start
        wait_time = max(0, dt - time_cost)
        if wait_time > 0:
            time.sleep(wait_time)

        # 如果游戏结束，等待一段时间后重置
        if done:
            time.sleep(2)  # 等待2秒
            state, info = env.reset()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 