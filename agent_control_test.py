import pygame
import sys
import time

from config import CURRENT_GAME
from utils.config.game_config import GameTeam, GameType
from visualization.renderer import Renderer
from agents.ppo_agent import PPOAgent

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
    # 初始化pygame
    pygame.init()

    # 创建环境和渲染器
    env = Environment()
    renderer = Renderer(env_config)
    
    # 创建PPO agent
    state_size = len(env._get_team_state(GameTeam.RED))
    agent = PPOAgent(
        team=GameTeam.RED,
        state_size=state_size,
        field_width=env_config.FIELD_WIDTH,
        field_height=env_config.FIELD_HEIGHT
    )
    
    # 加载训练好的模型
    try:
        agent.load("models/ppo_agent_episode_1.pt")
        print("成功加载模型")
    except:
        print("未找到模型文件，使用随机策略")

    show_grid = False  # 控制是否显示可移动栅格

    running = True
    while running:
        frame_start = time.perf_counter()

        dt = env.dt

        # 获取当前状态
        state = env._get_team_state(GameTeam.RED)
        
        # 使用agent选择动作
        red_action = agent.act(state)
        
        # 蓝方保持静止
        blue_action = {
            robot_id: Action(
                navigation=None,
                attack=False,
                target=None,
            ) for robot_id, robot in env.robots.items() if robot.team == GameTeam.BLUE
        }

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
        env.step(dt, red_action, blue_action)

        # 渲染环境
        renderer.render(env, show_grid=show_grid)

        # 更新显示
        pygame.display.flip()

        # 计算本帧消耗的时间
        time_cost = time.perf_counter() - frame_start
        wait_time = max(0, dt - time_cost)
        if wait_time > 0:
            time.sleep(wait_time)
            time.sleep(2)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
