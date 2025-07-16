import pygame
import sys
import time
import random
import numpy as np

from config import CURRENT_GAME
from utils.config.game_config import GameTeam, GameType
from utils.config.robot_config import RobotType
from utils.grid_map import world_to_grid, grid_to_world, a_star, simplify_path
from visualization.config import render_config

if CURRENT_GAME == GameType.BASE:
    from base.game import Game
    from base.environment import Action
    from base.config.robot_config import BASE_ROBOT_TYPE_LIST as ROBOT_TYPE_LIST
elif CURRENT_GAME == GameType.RMUL:
    from RMUL.game import GameRMUL as Game
    from RMUL.environment import ActionRMUL as Action
    from RMUL.config.robot_config import RMUL_ROBOT_TYPE_LIST as ROBOT_TYPE_LIST
# elif CURRENT_GAME == GameType.RMUC:
    # from RMUC.game import GameRMUC as Game
    # from RMUC.config.robot_config import RMUC_ROBOT_TYPE_LIST as ROBOT_TYPE_LIST

np.set_printoptions(precision=4, floatmode='fixed')

def main():
    # 创建环境
    game = Game(render_mode="human")

    robot_id_list = list(game.env.robots.keys())
    target_id_list = ROBOT_TYPE_LIST
    control_state = {
        "robot_id": robot_id_list[0],
        "target_id": target_id_list[0],
        "show_grid": False,
    }

    while True:
        red_action = {
            robot_id: Action() for robot_id, robot in game.env.robots.items() if robot.team == GameTeam.RED
        }
        blue_action = {
            robot_id: Action() for robot_id, robot in game.env.robots.items() if robot.team == GameTeam.BLUE
        }

        # blue_action["BLUE_3_STANDARD"].navigation = (6.0, 4.0)
        # blue_action["BLUE_3_STANDARD"].target = RobotType.STANDARD_3
        # blue_action["BLUE_3_STANDARD"].attack = True if random.random() < 0.05 else False

        for event in pygame.event.get():
            robot_in_control = game.env.robots[control_state["robot_id"]]
            if event.type == pygame.QUIT:
                game.close()
                sys.exit()

            # 鼠标左键点击，设置第一个机器人目标点
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()  # 屏幕坐标
                # 转换为世界坐标
                world_pos = (mouse_pos[0] / render_config.SCALE, mouse_pos[1] / render_config.SCALE)
                # 如果目标位置与当前位置相同，则不再次计算路径
                if world_pos != robot_in_control.target_pos:
                    robot_in_control.target_pos = world_pos

                    # 如果网格地图为空
                    if robot_in_control.grid_map is None:
                        raise ValueError("网格地图为空，无法计算路径")

                    # 使用A*算法规划路径
                    start_grid = world_to_grid(robot_in_control.get_position())
                    goal_grid = world_to_grid(world_pos)
                    path_grids = a_star(robot_in_control.grid_map, start_grid, goal_grid)
                    path_grids = simplify_path(path_grids, robot_in_control.grid_map)
                
                    # 将栅格坐标转换回世界坐标
                    robot_in_control.path_points = [grid_to_world(gp[0], gp[1]) for gp in path_grids[1:]]
                    robot_in_control.current_path_idx = 0

            # 按键事件
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()
                elif event.key == pygame.K_ESCAPE:
                    game.close()
                    sys.exit()
                elif event.key == pygame.K_TAB:  # 切换显示栅格
                    control_state["show_grid"] = not control_state["show_grid"]
                elif event.key == pygame.K_q:  # Q键攻击
                    if game.env.robots[control_state["robot_id"]].team == GameTeam.RED:
                        red_action[control_state["robot_id"]].attack_target = control_state["target_id"]
                    else:
                        blue_action[control_state["robot_id"]].attack_target = control_state["target_id"]
                elif event.key == pygame.K_e:  # E键购买子弹
                    if "purchase" in Action.__dict__:
                        if game.env.robots[control_state["robot_id"]].team == GameTeam.RED:
                            red_action[control_state["robot_id"]].purchase = 1
                        else:
                            blue_action[control_state["robot_id"]].purchase = 1
                elif event.key == pygame.K_w:  # W键切换机器人
                    control_state["robot_id"] = robot_id_list[(robot_id_list.index(control_state["robot_id"]) + 1) % len(robot_id_list)]
                elif event.key == pygame.K_s:  # S键切换机器人
                    control_state["robot_id"] = robot_id_list[(robot_id_list.index(control_state["robot_id"]) - 1) % len(robot_id_list)]
                elif event.key == pygame.K_a:  # A键切换目标
                    control_state["target_id"] = target_id_list[(target_id_list.index(control_state["target_id"]) + 1) % len(target_id_list)]
                elif event.key == pygame.K_d:  # D键切换目标
                    control_state["target_id"] = target_id_list[(target_id_list.index(control_state["target_id"]) - 1) % len(target_id_list)]

        # 计算速度
        for robot in game.env.robots.values():
            if robot.path_points and robot.current_path_idx < len(robot.path_points):
                next_point = robot.path_points[robot.current_path_idx]
                current_pos = pygame.math.Vector2(robot.get_position())
                direction = pygame.math.Vector2(next_point) - current_pos
                if direction.length() < 0.05:
                    robot.current_path_idx += 1
                else:
                    direction = direction.normalize()
                    velocity = (direction.x, direction.y)
            else:
                velocity = (0, 0)

            if robot.team == GameTeam.RED:
                red_action[robot.id].velocity = velocity
            else:
                blue_action[robot.id].velocity = velocity

        # print(red_action, blue_action)
        # 更新环境
        game.set_render(control_state)
        observation, reward, terminated, truncated, info = game.step(red_action, blue_action)
        state = observation.to_array()


if __name__ == "__main__":
    main()
