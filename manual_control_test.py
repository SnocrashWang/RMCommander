import pygame
import sys
import time

from config import CURRENT_GAME
from utils.config.game_config import GameTeam, GameType
from utils.config.robot_config import RobotType
from visualization.config import render_config
from visualization.renderer import Renderer

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

    show_grid = False  # 控制是否显示可移动栅格

    running = True
    while running:
        frame_start = time.perf_counter()

        dt = env.dt

        red_action = {
            robot_id: Action(
                navigation=None,
                attack=False,
                target=None,
            ) for robot_id, robot in env.robots.items() if robot.team == GameTeam.RED
        }
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

            # 鼠标左键点击，设置第一个机器人目标点
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()  # 屏幕坐标
                # 转换为世界坐标
                world_x = mouse_pos[0] / render_config.SCALE
                world_y = mouse_pos[1] / render_config.SCALE
                if env.robots:  # 兼容机器人列表
                    red_action["RED_3_STANDARD"].navigation = (world_x, world_y)

            # 按键事件
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
                    print("Environment reset")
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_g:  # 切换显示栅格
                    show_grid = not show_grid
                elif event.key == pygame.K_a:  # A键攻击
                    red_action["RED_3_STANDARD"].attack = True
                    red_action["RED_3_STANDARD"].target = RobotType.STANDARD_3

        # 更新环境
        env.step(dt, red_action, blue_action)

        # 渲染环境
        renderer.render(env, show_grid=show_grid, show_control=True)

        # 更新显示
        pygame.display.flip()

        # 计算本帧消耗的时间
        time_cost = time.perf_counter() - frame_start
        wait_time = max(0, dt - time_cost)
        if wait_time > 0:
            time.sleep(wait_time)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
