import os
import json
import time
import pygame
import argparse
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

def replay_episode(episode_data: Dict[str, Any], dt: float = 0.5):
    """回放一个回合的动作序列"""
    # 初始化环境和渲染器
    pygame.init()

    env = Environment()
    env.reset()
    renderer = Renderer(env_config)
    
    # 回放每一帧
    for frame_action in episode_data['actions']:
        frame_start = time.perf_counter()

        # 处理pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
        # print(frame_action)
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
        print(red_action)
        env.step(env.dt, red_action, blue_action)
        
        # 渲染环境
        renderer.render(env, show_grid=True)
        pygame.display.flip()
        
        # 计算本帧消耗的时间
        time_cost = time.perf_counter() - frame_start
        wait_time = max(0, dt - time_cost)
        if wait_time > 0:
            time.sleep(wait_time)
    
    pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='回放训练过程中的动作序列')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志文件目录')
    parser.add_argument('--episode', type=int, default=0, help='要回放的回合编号')
    parser.add_argument('--delay', type=float, default=None, help='渲染延迟时间（秒）')
    args = parser.parse_args()

    if args.delay is None:
        args.delay = 1 / env_config.FPS
    
    if args.episode is not None:
        # 回放指定回合
        log_file = os.path.join(args.log_dir, f'episode_{args.episode}.json')
        if not os.path.exists(log_file):
            print(f"错误：找不到回合 {args.episode} 的日志文件")
            return
        
        episode_data = load_episode(log_file)
        print(f"回放回合 {args.episode}")
        print(f"总奖励: {episode_data['reward']:.2f}")
        print(f"回合长度: {episode_data['length']}")
        replay_episode(episode_data, args.delay)
    else:
        # 列出所有可用的回合
        log_files = [f for f in os.listdir(args.log_dir) if f.startswith('episode_') and f.endswith('.json')]
        if not log_files:
            print(f"错误：在 {args.log_dir} 目录下找不到任何回合日志文件")
            return
        
        print("可用的回合：")
        for log_file in sorted(log_files):
            episode_num = int(log_file.split('_')[1].split('.')[0])
            episode_data = load_episode(os.path.join(args.log_dir, log_file))
            print(f"回合 {episode_num}: 奖励={episode_data['reward']:.2f}, 长度={episode_data['length']}")

if __name__ == "__main__":
    main() 