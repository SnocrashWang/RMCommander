import pygame
import os
import sys
import time
import argparse
import cv2
import numpy as np
import random

from config import CURRENT_GAME
from utils.config.game_config import GameTeam, GameType
from visualization.renderer import Renderer
from agents.ppo_agent import PPOAgent
from utils.config.robot_config import RobotType
from utils.grid_map import world_to_grid
from utils.utils import opposite_position

if CURRENT_GAME == GameType.BASE:
    from base.environment import Environment, Action
    from base.config import env_config
elif CURRENT_GAME == GameType.RMUL:
    from RMUL.environment import EnvironmentRMUL as Environment, Action
    from RMUL.config import env_config
# elif CURRENT_GAME == GameType.RMUC:
#     from RMUC.environment import EnvironmentRMUC, Action
#     from RMUC.config import env_config

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--model_file', type=str, default=None, help='模型文件')
    parser.add_argument('-d', '--delay', type=float, default=None, help='渲染延迟时间（秒）')
    parser.add_argument('-v', '--video', action='store_true', help='是否保存为视频')
    parser.add_argument('--video_dir', type=str, default='videos', help='视频保存目录')
    args = parser.parse_args()

    # 初始化pygame
    pygame.init()

    # 创建环境和渲染器
    env = Environment()
    renderer = Renderer(env_config)
    
    # 如果保存视频，初始化视频写入器
    video_writer = None
    if args.video:
        os.makedirs(args.video_dir, exist_ok=True)
        # 获取第一帧来确定视频尺寸
        renderer.render(env, {"show_grid": False})
        pygame.display.flip()
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = frame.transpose([1, 0, 2])  # 转置以匹配cv2的格式
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(args.video_dir, f'agent_control_{time.strftime("%Y%m%d_%H%M%S")}.mp4')
        video_writer = cv2.VideoWriter(video_path, fourcc, int(1/env.dt), (width, height))
        print(f"视频将保存到: {video_path}")
    
    # 创建PPO agent
    state_size = len(env._get_team_state(GameTeam.RED))
    agent_red = PPOAgent(
        team=GameTeam.RED,
        state_size=state_size,
        field_width=env_config.FIELD_WIDTH,
        field_height=env_config.FIELD_HEIGHT,
        device="cpu"
    )
    agent_blue = PPOAgent(
        team=GameTeam.BLUE,
        state_size=state_size,
        field_width=env_config.FIELD_WIDTH,
        field_height=env_config.FIELD_HEIGHT,
        device="cpu"
    )

    blue_navigation = (random.randint(0, int(env_config.FIELD_WIDTH)), random.randint(0, int(env_config.FIELD_HEIGHT)))
    while env.get_robot("BLUE_3_STANDARD").grid_map.is_blocked(*world_to_grid(blue_navigation)):
        blue_navigation = (random.randint(0, int(env_config.FIELD_WIDTH)), random.randint(0, int(env_config.FIELD_HEIGHT)))
    blue_action = {"BLUE_3_STANDARD": Action(navigation=blue_navigation, attack=True, target=RobotType.STANDARD_3)}
    
    # 加载训练好的模型
    try:
        agent_red.load(args.model_file)
        agent_blue.load(args.model_file)
        print("成功加载模型")
    except:
        print("未找到模型文件，使用随机策略")

    show_grid = False  # 控制是否显示可移动栅格

    running = True
    while running:
        frame_start = time.perf_counter()

        if args.delay is None:
            args.delay = env.dt

        # 获取当前状态
        state = env._get_team_state(GameTeam.RED)
        
        # 红方动作
        red_action = agent_red.act(state)
        
        # # 蓝方动作
        # blue_action = {
        #     robot_id: Action(
        #         navigation=None,
        #         attack=False,
        #         target=None,
        #     ) for robot_id, robot in env.robots.items() if robot.team == GameTeam.BLUE
        # }
        # blue_action = agent_blue.act(state)
        # blue_action["BLUE_3_STANDARD"].navigation = opposite_position(blue_action["BLUE_3_STANDARD"].navigation, env_config.FIELD_WIDTH, env_config.FIELD_HEIGHT)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # 按键事件
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
                    print("Environment reset")
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_g:  # 切换显示栅格
                    show_grid = not show_grid

        # 更新环境
        env.step(env.dt, red_action, blue_action)

        # 渲染环境
        renderer.render(env, {"show_grid": show_grid})

        # 如果保存视频，保存当前帧
        if video_writer:
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = frame.transpose([1, 0, 2])  # 转置以匹配cv2的格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转换颜色空间
            video_writer.write(frame)

        # 更新显示
        pygame.display.flip()

        # 计算本帧消耗的时间
        time_cost = time.perf_counter() - frame_start
        wait_time = max(0, args.delay - time_cost)
        if wait_time > 0:
            time.sleep(wait_time)

    if video_writer:
        video_writer.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
