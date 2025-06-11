import pygame
import sys
import time

from base_game.environment import Environment
from RMUL.environment import EnvironmentRL
from visualization.renderer import Renderer
from base_game.config import env_config as base_env_config
from visualization.config import render_config

def main():
    # 初始化pygame
    pygame.init()

    # 创建环境和渲染器
    env = Environment()
    # env = EnvironmentRL()
    renderer = Renderer(base_env_config)

    # 创建时钟对象
    clock = pygame.time.Clock()

    show_grid = False  # 控制是否显示可移动栅格

    running = True
    while running:
        frame_start = time.perf_counter()

        dt = env.dt

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
                    env.robots["RED_3_STANDARD"].set_target((world_x, world_y))

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
                    if env.robots and len(env.robots) > 1:
                        # 第一个机器人攻击第二个机器人
                        env.robots["RED_3_STANDARD"].attack(env.robots["BLUE_3_STANDARD"], 1)

        # 更新环境
        env.step(dt)

        # 获取环境状态
        env_state = env.get_game_state()

        # 渲染环境
        renderer.render(env_state, show_grid=show_grid)

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
