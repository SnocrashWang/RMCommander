import pygame
import sys
import time
import random
import numpy as np

from config import CURRENT_GAME
from utils.config.game_config import GameTeam, GameType
from utils.config.robot_config import RobotType
from visualization.config import render_config
# from visualization.renderer import Renderer

if CURRENT_GAME == GameType.BASE:
    from base.environment import Action
    # from base.config import env_config
    from base.gym import RoboMasterGym as Environment
    from base.config.robot_config import BASE_ROBOT_TYPE_LIST as ROBOT_TYPE_LIST
elif CURRENT_GAME == GameType.RMUL:
    from RMUL.environment import EnvironmentRMUL as Environment, ActionRMUL as Action
    from RMUL.config import env_config
    from RMUL.config.robot_config import RMUL_ROBOT_TYPE_LIST as ROBOT_TYPE_LIST
# elif CURRENT_GAME == GameType.RMUC:
#     from RMUC.environment import EnvironmentRMUC, Action
#     from RMUC.config import env_config
#     from RMUC.config.robot_config import RMUC_ROBOT_TYPE_LIST as ROBOT_TYPE_LIST

def main():
    # 创建环境和渲染器
    env = Environment(render_mode="human")
    # renderer = Renderer(env_config)

    robot_id_list = list(env.env.robots.keys())
    target_id_list = ROBOT_TYPE_LIST
    control_state = {
        "robot_id": robot_id_list[0],
        "target_id": target_id_list[0],
        "show_grid": False,
    }

    while True:

        default_action = {
            "attack": np.int64(0),
            "navigation": np.array([0.0, 0.0]),
            "target": np.int64(0),
        }

        red_action = {
            robot_id: default_action for robot_id, robot in env.env.robots.items() if robot.team == GameTeam.RED
        }
        blue_action = {
            robot_id: default_action for robot_id, robot in env.env.robots.items() if robot.team == GameTeam.BLUE
        }

        # blue_action["BLUE_3_STANDARD"].navigation = (6.0, 4.0)
        # blue_action["BLUE_3_STANDARD"].target = RobotType.STANDARD_3
        # blue_action["BLUE_3_STANDARD"].attack = True if random.random() < 0.05 else False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit()

            # 鼠标左键点击，设置第一个机器人目标点
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()  # 屏幕坐标
                # 转换为世界坐标
                world_x = mouse_pos[0] / render_config.SCALE
                world_y = mouse_pos[1] / render_config.SCALE
                if env.env.robots[control_state["robot_id"]].team == GameTeam.RED:
                    red_action[control_state["robot_id"]]["navigation"] = (world_x, world_y)
                else:
                    blue_action[control_state["robot_id"]]["navigation"] = (world_x, world_y)

            # 按键事件
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.reset()
                    print("Environment reset")
                elif event.key == pygame.K_ESCAPE:
                    env.close()
                    sys.exit()
                elif event.key == pygame.K_TAB:  # 切换显示栅格
                    control_state["show_grid"] = not control_state["show_grid"]
                elif event.key == pygame.K_q:  # Q键攻击
                    if env.env.robots[control_state["robot_id"]].team == GameTeam.RED:
                        red_action[control_state["robot_id"]]["attack"] = True
                        red_action[control_state["robot_id"]]["target"] = control_state["target_id"]
                    else:
                        blue_action[control_state["robot_id"]]["attack"] = True
                        blue_action[control_state["robot_id"]]["target"] = control_state["target_id"]
                elif event.key == pygame.K_e:  # E键购买子弹
                    if env.env.robots[control_state["robot_id"]].team == GameTeam.RED:
                        red_action[control_state["robot_id"]]["purchase"] = True
                    else:
                        blue_action[control_state["robot_id"]]["purchase"] = True
                elif event.key == pygame.K_w:  # W键切换机器人
                    control_state["robot_id"] = robot_id_list[(robot_id_list.index(control_state["robot_id"]) + 1) % len(robot_id_list)]
                elif event.key == pygame.K_s:  # S键切换机器人
                    control_state["robot_id"] = robot_id_list[(robot_id_list.index(control_state["robot_id"]) - 1) % len(robot_id_list)]
                elif event.key == pygame.K_a:  # A键切换目标
                    control_state["target_id"] = target_id_list[(target_id_list.index(control_state["target_id"]) + 1) % len(target_id_list)]
                elif event.key == pygame.K_d:  # D键切换目标
                    control_state["target_id"] = target_id_list[(target_id_list.index(control_state["target_id"]) - 1) % len(target_id_list)]

        # 更新环境
        observation, reward, terminated, truncated, info = env.step({**red_action, **blue_action})

        # # 渲染环境
        # renderer.render(env, control_state)

        # # 更新显示
        # pygame.display.flip()

        # # 计算本帧消耗的时间
        # time_cost = time.perf_counter() - frame_start
        # wait_time = max(0, dt - time_cost)
        # if wait_time > 0:
        #     time.sleep(wait_time)


if __name__ == "__main__":
    main()
