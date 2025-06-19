import os
import json
import time
import pygame
import argparse
import cv2
import numpy as np
from typing import Dict, Any

from base.environment import Environment, Action
from base.config import env_config
from visualization.renderer import Renderer
from utils.config.game_config import GameTeam
from utils.config.robot_config import RobotType

def load_episode(log_file: str) -> Dict[str, Any]:
    """加载回合日志文件"""
    with open(log_file, 'r') as f:
        return json.load(f)

def replay_episode(episode_data: Dict[str, Any], delay: float, save_video: bool = False, video_path: str = None):
    """回放一个回合的动作序列"""
    # 初始化环境和渲染器
    pygame.init()

    env = Environment()
    env.reset()
    renderer = Renderer(env_config)
    
    # 如果保存视频，初始化视频写入器
    video_writer = None
    if save_video and video_path:
        # 获取第一帧来确定视频尺寸
        renderer.render(env, {"show_grid": True, "robot_id": "RED_3_STANDARD", "target_id": RobotType.STANDARD_3})
        pygame.display.flip()
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = frame.transpose([1, 0, 2])  # 转置以匹配cv2的格式
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, env_config.FPS, (width, height))
    
    # 回放每一帧
    for frame_action in episode_data['actions']:
        frame_start = time.perf_counter()

        # 处理pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                if video_writer:
                    video_writer.release()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    if video_writer:
                        video_writer.release()
                    return

        # 转换动作格式
        red_action = {
            robot_id: Action(
                navigation=tuple(action['navigation']) if action['navigation'] else None,
                attack=action['attack'],
                target=RobotType(action['target']) if action['target'] else RobotType.NONE
            ) for robot_id, action in frame_action['red_action'].items()
        }
        
        blue_action = {
            robot_id: Action(
                navigation=tuple(action['navigation']) if action['navigation'] else None,
                attack=action['attack'],
                target=RobotType(action['target']) if action['target'] else RobotType.NONE
            ) for robot_id, action in frame_action['blue_action'].items()
        }
        
        # 执行动作
        # print(red_action)
        env.step(env.dt, red_action, blue_action)
        
        # 渲染环境
        renderer.render(env, {"show_grid": True, "robot_id": "RED_3_STANDARD", "target_id": RobotType.STANDARD_3})
        pygame.display.flip()
        
        # 如果保存视频，保存当前帧
        if video_writer:
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = frame.transpose([1, 0, 2])  # 转置以匹配cv2的格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转换颜色空间
            video_writer.write(frame)
        
        # 计算本帧消耗的时间
        time_cost = time.perf_counter() - frame_start
        wait_time = max(0, delay - time_cost)
        if wait_time > 0:
            time.sleep(wait_time)
    
    if video_writer:
        video_writer.release()
    pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='回放训练过程中的动作序列')
    parser.add_argument('-l', '--log_file', type=str, default=None, help='日志文件')
    parser.add_argument('-d', '--delay', type=float, default=None, help='渲染延迟时间（秒）')
    parser.add_argument('-v', '--video', action='store_true', help='是否保存为视频')
    parser.add_argument('--video_dir', type=str, default='videos', help='视频保存目录')
    args = parser.parse_args()

    if args.delay is None:
        args.delay = 1 / env_config.FPS
    
    if args.log_file is not None:
        # 回放指定回合
        if not os.path.exists(args.log_file):
            print(f"错误：找不到 {args.log_file}")
            return
        
        episode_data = load_episode(args.log_file)
        print(f"回放回合 {args.log_file}")
        print(f"总奖励: {episode_data['reward']:.2f}")
        print(f"回合长度: {episode_data['length']}")
        
        # 如果保存视频，创建视频保存目录
        video_path = None
        if args.video:
            os.makedirs(args.video_dir, exist_ok=True)
            video_path = os.path.join(args.video_dir, args.log_file.replace('\\', '/').split("/")[-1].split('.')[0] + ".mp4")
            print(f"视频将保存到: {video_path}")
        
        replay_episode(episode_data, args.delay, args.video, video_path)
    else:
        # 列出所有可用的回合
        log_files = [f for f in os.listdir('logs') if f.endswith('.json')]
        if not log_files:
            print(f"错误：在 logs 目录下找不到任何回合日志文件")
            return
        
        print("可用的回合：")
        for log_file in sorted(log_files):
            episode_num = int(log_file.split('_')[1].split('.')[0])
            episode_data = load_episode(os.path.join('logs', log_file))
            print(f"回合 {episode_num}: 奖励={episode_data['reward']:.2f}, 长度={episode_data['length']}")

if __name__ == "__main__":
    main() 