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
from agents.ppo_agent import PPOAgent
from utils.config.robot_config import RobotType
from utils.grid_map import world_to_grid
from utils.utils import calc_distance

if CURRENT_GAME == GameType.BASE:
    from rules.base.game import Game
    from rules.base.environment import ActionBase as Action
    from rules.base.config import env_config
    from rules.base.config.robot_config import BASE_ROBOT_TYPE_ACTION as ROBOT_TYPE_ACTION
elif CURRENT_GAME == GameType.RMUL:
    from rules.rmul.game import GameRMUL as Game
    from rules.rmul.environment import ActionRMUL as Action
    from rules.rmul.config import env_config
    from rules.rmul.config.robot_config import RMUL_ROBOT_TYPE_ACTION as ROBOT_TYPE_ACTION
# elif CURRENT_GAME == GameType.RMUC:
#     from RMUC.environment import EnvironmentRMUC, Action
#     from RMUC.config import env_config


np.set_printoptions(precision=3, suppress=True)

def is_valid_base_position(game: Game, position):
    try:
        col, row = world_to_grid(position)
        return not game.env.get_robot("RED_3_STANDARD")._grid_map.is_blocked(col, row)
    except ValueError:
        return False


def sample_base_position(game: Game, x_range, y_range):
    for _ in range(1000):
        position = (random.uniform(*x_range), random.uniform(*y_range))
        if is_valid_base_position(game, position):
            return position
    raise RuntimeError("failed to sample a valid start position")

def agent_control(
    model_file: str,
    delay: float,
    control_frequency: float,
    save_video: bool,
    video_path: str,
    deterministic: bool,
):
    # 初始化
    if save_video:
        render_mode = "rgb_array"
        if video_path is None:
            raise ValueError("视频保存路径不能为空")
    else:
        render_mode = "human"

    # 创建环境和渲染器
    game = Game(render_mode=render_mode)
    obs, info = game.reset()
    
    # 如果保存视频，初始化视频写入器
    video_writer = None
    if save_video:
        # 获取第一帧来确定视频尺寸
        frame = info['render_images'][0]
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(video_path, model_file.replace('\\', '/').split("/")[-1].split('.')[0] + ".mp4")
        video_writer = cv2.VideoWriter(video_path, fourcc, game.metadata['render_fps'], (width, height))
        print(f"视频将保存到: {video_path}")
    
    # 创建PPO agent
    state_size = game.observation_space.shape[0]
    agent = PPOAgent(
        state_dim=state_size,
        device=None,
        robot_type_action=ROBOT_TYPE_ACTION,
    )

    # 加载训练好的模型
    try:
        agent.load(model_file)
        print("成功加载模型")
    except:
        print("未找到模型文件，使用随机权重")

    show_grid = False  # 控制是否显示可移动栅格
    control_steps = int(game.metadata['render_fps'] // control_frequency)

    running = True
    while running:
        frame_start = time.perf_counter()
        
        # 动作
        red_action = agent.take_action(obs.to_array(GameTeam.RED), GameTeam.RED, deterministic=deterministic)
        blue_action = agent.take_action(obs.to_array(GameTeam.BLUE), GameTeam.BLUE, deterministic=deterministic)
        # blue_action = {"BLUE_3_STANDARD": mirror_navigation_target_actions(red_action)["RED_3_STANDARD"]}
        # blue_action = {"BLUE_3_STANDARD": Action(attack_target=3, spin=1)}
        print(obs.to_array(GameTeam.RED))
        print(red_action)
        # print(blue_action)

        # 更新环境
        obs, reward, terminated, truncated, info = game.step(red_action, blue_action, control_steps)
        running = not terminated and not truncated

        # 如果保存视频，保存当前帧
        if video_writer:
            frames = info['render_images']
            for frame in frames:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转换颜色空间
                video_writer.write(frame)
            continue

        # 计算本帧消耗的时间
        time_cost = time.perf_counter() - frame_start
        wait_time = max(0, delay - time_cost)
        if wait_time > 0:
            time.sleep(wait_time)

        # 按键控制
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # 按键事件
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = obs, info = game.reset()
                    print("Environment reset")
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_g:  # 切换显示栅格
                    show_grid = not show_grid

    if video_writer:
        video_writer.release()
    pygame.quit()
    sys.exit()

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--model_file', type=str, default=None, help='模型文件')
    parser.add_argument('-d', '--delay', type=float, default=0, help='渲染延迟时间（秒）')
    parser.add_argument('-v', '--video', action='store_true', help='是否保存为视频')
    parser.add_argument('--control_frequency', type=float, default=2, help='控制频率（Hz）')
    parser.add_argument('--video_dir', type=str, default='videos', help='视频保存目录')
    parser.add_argument('--deterministic', action='store_true', help='使用确定性策略进行评估')
    args = parser.parse_args()

    agent_control(
        args.model_file,
        args.delay,
        args.control_frequency,
        args.video,
        args.video_dir,
        args.deterministic,
    )

if __name__ == "__main__":
    main()
