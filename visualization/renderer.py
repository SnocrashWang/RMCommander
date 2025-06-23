import math
import pygame
from visualization.config import render_config
from utils.config.exp_prop_config import LEVEL_NEED_EXP
from utils.config.game_config import GameState, GameTeam
from utils.config.robot_config import ROBOT_ID
from utils.utils import meters_to_pixels, draw_dashed_line, second2minute, get_tangent_points, opposite_team

class Renderer:
    def __init__(self, env_config):
        self.env_config = env_config
        self.control_state = {}

        # 计算屏幕尺寸
        self.screen_width = meters_to_pixels(self.env_config.FIELD_WIDTH)
        self.screen_height = meters_to_pixels(self.env_config.FIELD_HEIGHT)

        # self.screen_main = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.screen_field = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        self.screen_robot = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        self.screen_grid = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        self.screen_grid.set_alpha(render_config.ALPHA_GRID)
        self.screen_note = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        # pygame.display.set_caption("Robot Battle Environment")

        # 创建栅格缓存
        self.last_grid_state = None  # 用于跟踪栅格状态

        self.font_large = pygame.font.SysFont("consolas", 36, bold=True)
        self.font_medium = pygame.font.SysFont("consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 16, bold=True)
        self.font_tiny = pygame.font.SysFont("consolas", 9, bold=True)

    def render(self, env) -> pygame.Surface:
        """渲染环境"""
        # 清空屏幕
        screen = pygame.Surface((self.screen_width, self.screen_height))
        screen.fill(render_config.COLOR_BACKGROUND)
        self.screen_field.fill((0, 0, 0, 0))
        self.screen_robot.fill((0, 0, 0, 0))
        self.screen_note.fill((0, 0, 0, 0))

        # 绘制增益区域
        if hasattr(env, "buff_zone"):
            self._draw_buff_zone(env)

        # 绘制四周墙壁
        self._draw_walls()

        # 绘制障碍物
        self._draw_obstacles(env.obstacles)

        # 绘制所有机器人
        for robot in env.robots.values():
            self._draw_robot(robot)

        # 绘制攻击线
        for robot in env.robots.values():
            self._draw_attack_line(robot.get_attack_line())

        # 绘制顶部信息
        self._draw_top_bar(env.get_top_bar_info(), self.env_config.ENV_NAME)

        # 绘制控制提示信息
        if self.control_state:
            self._draw_control_info(self.control_state, self.env_config.ENV_NAME)
            # 绘制可移动栅格
            if self.control_state.get("show_grid", False) and "robot_id" in self.control_state:
                robot = env.robots[self.control_state["robot_id"]]
                self._draw_grid(robot.grid_map)
                self._draw_path(robot)
                self._draw_attack_sight_line(robot, env.get_robot(ROBOT_ID[opposite_team(robot.team)][self.control_state["target_id"]]))

        # 绘制游戏结束信息
        self._draw_game_over(env.game_state)

        # 叠加图层
        screen.blit(self.screen_field, (0, 0))
        screen.blit(self.screen_robot, (0, 0))
        if self.control_state.get("show_grid", False):
            screen.blit(self.screen_grid, (0, 0))
        screen.blit(self.screen_note, (0, 0))

        return screen

    def _draw_buff_zone(self, env):
        """绘制增益区域"""
        # 绘制所有增益区
        for buff_zone in env.buff_zone.values():
            vertices = []
            for x, y in buff_zone:
                px = meters_to_pixels(x)
                py = meters_to_pixels(y)
                vertices.append((px, py))
            
            # 在Surface上绘制增益区
            pygame.draw.polygon(self.screen_field, render_config.COLOR_CENTER_ZONE, vertices)

    def _draw_walls(self):
        """绘制围墙"""
        # 上墙
        pygame.draw.rect(self.screen_field, render_config.COLOR_WALL, (0, 0, meters_to_pixels(self.env_config.FIELD_WIDTH), 10))
        # 下墙
        pygame.draw.rect(self.screen_field, render_config.COLOR_WALL, (0, meters_to_pixels(self.env_config.FIELD_HEIGHT) - 10, meters_to_pixels(self.env_config.FIELD_WIDTH), 10))
        # 左墙
        pygame.draw.rect(self.screen_field, render_config.COLOR_WALL, (0, 0, 10, meters_to_pixels(self.env_config.FIELD_HEIGHT)))
        # 右墙
        pygame.draw.rect(self.screen_field, render_config.COLOR_WALL, (meters_to_pixels(self.env_config.FIELD_WIDTH) - 10, 0, 10, meters_to_pixels(self.env_config.FIELD_HEIGHT)))

    def _draw_obstacles(self, obstacles):
        """绘制所有障碍物为有厚度的矩形"""
        for obs in obstacles:
            x1, y1 = obs.p1
            x2, y2 = obs.p2
            thickness = obs.thickness
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length == 0:
                continue
            nx, ny = -dy / length, dx / length  # 法向量
            half_t = thickness / 2

            # 四个顶点（世界坐标）
            v1 = (meters_to_pixels(x1 + nx * half_t), meters_to_pixels(y1 + ny * half_t))
            v2 = (meters_to_pixels(x1 - nx * half_t), meters_to_pixels(y1 - ny * half_t))
            v3 = (meters_to_pixels(x2 - nx * half_t), meters_to_pixels(y2 - ny * half_t))
            v4 = (meters_to_pixels(x2 + nx * half_t), meters_to_pixels(y2 + ny * half_t))

            pygame.draw.polygon(self.screen_field, render_config.COLOR_OBSTACLE, [v1, v2, v3, v4])

    def _draw_robot(self, robot):
        """绘制机器人"""
        position = robot.get_position()
        x = meters_to_pixels(position[0])
        y = meters_to_pixels(position[1])
        radius = meters_to_pixels(robot.radius)
        scale = 0.25 * render_config.SCALE

        # 绘制机器人主体
        pygame.draw.circle(self.screen_robot, render_config.ROBOT_COLORS[robot.team], (x, y), radius)

        # 绘制炮塔（小方形）
        turret_size = scale * 0.6
        turret_rect = pygame.Rect(x - turret_size/2, y - turret_size/2, turret_size, turret_size)
        pygame.draw.rect(self.screen_robot, (30, 30, 30), turret_rect)

        # 绘制炮管
        angle_rad = math.radians(robot.angle)
        barrel_length = scale * 0.8
        end_x = x + barrel_length * math.cos(angle_rad)
        end_y = y + barrel_length * math.sin(angle_rad)
        pygame.draw.line(self.screen_robot, (30, 30, 30), (x, y), (end_x, end_y), int(2))

        # 绘制机器人标识（使用小字体）
        team_text = self.font_tiny.render(f"{robot.id}", True, render_config.COLOR_TEXT)
        self.screen_note.blit(team_text, (x - team_text.get_width() / 2, y + scale * 1.5))

        # 绘制数据条
        bar_width = render_config.SCALE * 0.5
        bar_height = render_config.SCALE * 0.08
        # 血量条
        if robot.is_alive:
            hp_bar_x = x - bar_width / 2
            hp_bar_y = y - scale * 1.5 - bar_height * 2
            current_hp_width = int(bar_width * robot.hp / robot.max_hp)
            pygame.draw.rect(self.screen_note, render_config.TEAM_COLORS[robot.team] if robot.defense_buff == 0 else render_config.COLOR_GREEN,
                            (hp_bar_x, hp_bar_y, current_hp_width, bar_height))
            # FIXME: 有一个特殊情况会导致hp为小数，暂时还没有复现到
            try:
                hp_text = self.font_tiny.render(f"{robot.hp:>3d}/{robot.max_hp:>3d}", True, render_config.COLOR_TEXT)
                self.screen_note.blit(hp_text, (hp_bar_x + bar_width / 2 - hp_text.get_width() / 2, hp_bar_y + bar_height / 2 - hp_text.get_height() / 2))
            except:
                print(robot.id, robot.hp, robot.max_hp)
                exit()
        else:
            revive_bar_x = x - bar_width / 2
            revive_bar_y = y - scale * 1.5 - bar_height * 2
            current_revive_width = int(bar_width * robot.revive_progress / robot.revive_target)
            pygame.draw.rect(self.screen_note, render_config.COLOR_REVIVE_BAR,
                            (revive_bar_x, revive_bar_y, current_revive_width, bar_height))
            revive_text = self.font_tiny.render(f"{int(robot.revive_progress):>3d}/{robot.revive_target:>3d}", True, render_config.COLOR_TEXT)
            self.screen_note.blit(revive_text, (revive_bar_x + bar_width / 2 - revive_text.get_width() / 2, revive_bar_y + bar_height / 2 - revive_text.get_height() / 2))

        # 热量条
        heat_bar_x = x - bar_width / 2
        heat_bar_y = y - scale * 1.5 - bar_height
        current_heat_width = int(bar_width * robot.heat / robot.max_heat)
        pygame.draw.rect(self.screen_note, render_config.COLOR_HEAT_BAR,
                        (heat_bar_x, heat_bar_y, current_heat_width, bar_height))
        heat_text = self.font_tiny.render(f"{int(robot.heat)}/{int(robot.max_heat)}", True, render_config.COLOR_TEXT)
        self.screen_note.blit(heat_text, (heat_bar_x + bar_width / 2 - heat_text.get_width() / 2, heat_bar_y + bar_height / 2 - heat_text.get_height() / 2))
        # 子弹
        ammo_text = self.font_tiny.render(f"{robot.ammo_allowed}", True, render_config.COLOR_TEXT if not robot.gun_locked else render_config.COLOR_SILVER_GRAY)
        self.screen_note.blit(ammo_text, (heat_bar_x - ammo_text.get_width(), heat_bar_y + bar_height / 2 - ammo_text.get_height() / 2))

        # 经验条
        exp_bar_x = x - bar_width / 2
        exp_bar_y = y - scale * 1.5
        if robot.level < len(LEVEL_NEED_EXP):
            exp_need_to_level_up = LEVEL_NEED_EXP[robot.level + 1] - LEVEL_NEED_EXP[robot.level]
        else:
            exp_need_to_level_up = LEVEL_NEED_EXP[robot.level] - LEVEL_NEED_EXP[robot.level - 1]
        current_exp_width = int(bar_width * min(1, (robot.exp - LEVEL_NEED_EXP[min(robot.level, len(LEVEL_NEED_EXP) - 1)]) / exp_need_to_level_up))
        pygame.draw.rect(self.screen_note, render_config.COLOR_EXP_BAR,
                        (exp_bar_x, exp_bar_y, current_exp_width, bar_height))
        exp_text = self.font_tiny.render(f"{robot.exp - LEVEL_NEED_EXP[min(robot.level, len(LEVEL_NEED_EXP) - 1)]:>3d}/{exp_need_to_level_up:>3d}", True, render_config.COLOR_TEXT)
        self.screen_note.blit(exp_text, (exp_bar_x + bar_width / 2 - exp_text.get_width() / 2, exp_bar_y + bar_height / 2 - exp_text.get_height() / 2))
        # 等级
        level_text = self.font_tiny.render(f"Lv.{robot.level:>2d}", True, render_config.COLOR_TEXT)
        self.screen_note.blit(level_text, (exp_bar_x - level_text.get_width(), exp_bar_y + bar_height / 2 - level_text.get_height() / 2))

    def _draw_attack_line(self, attack_line):
        """绘制攻击线
        Args:
            attack_line: 攻击线，如果为None，则不绘制
        """
        if attack_line:
            draw_dashed_line(
                self.screen_robot, (0, 255, 0), attack_line[0], attack_line[1],
                int(render_config.SCALE * 0.02), int(render_config.SCALE * 0.05), int(render_config.SCALE * 0.1)
            )

    def _draw_attack_sight_line(self, robot, target_robot):
        """绘制视野"""
        p1 = robot.get_position()
        p2 = target_robot.get_position()
        r = robot.radius
        tangents = get_tangent_points(p1, p2, r)
        if len(tangents) < 2:
            return
        for tangent in tangents:
            draw_dashed_line(
                self.screen_note, render_config.COLOR_GREEN, p1, tangent,
                int(render_config.SCALE * 0.01), int(render_config.SCALE * 0.02), int(render_config.SCALE * 0.05)
            )

    def _draw_top_bar(self, top_bar_info, env_name):
        """绘制顶部信息条"""
        # 倒计时
        min, sec = second2minute(int(top_bar_info["remaining_time"]))
        time_text = self.font_medium.render(f"Time: {min:02d}:{sec:02d}", True, render_config.COLOR_TEXT)
        self.screen_note.blit(time_text, ((self.screen_width - time_text.get_width()) // 2, time_text.get_height() // 2))
        return

        if env_name == "RMUL":
            bar_height = meters_to_pixels(self.env_config.FIELD_HEIGHT * 0.02)
            bar_width = self.screen_width * 0.4

            # 红队进度条
            pygame.draw.rect(self.screen_note, render_config.COLOR_PROGRESS_BAR_BG, (10, 10, bar_width, bar_height))
            progress_width = int(bar_width * (game_state.get_victory_progress(GameTeam.RED) / self.env_config.OCCUPATION_TARGET))
            pygame.draw.rect(self.screen_note, render_config.TEAM_COLORS[GameTeam.RED],
                            (10, 10, progress_width, bar_height))
            red_progress_text = self.font_medium.render(
                f"Team RED: {int(game_state.get_victory_progress(GameTeam.RED)):>3d} / {self.env_config.OCCUPATION_TARGET:>3d}",
                True, render_config.COLOR_TEXT
            )
            self.screen_note.blit(red_progress_text, (bar_width - red_progress_text.get_width(), red_progress_text.get_height() // 2))

            # 蓝队进度条
            pygame.draw.rect(self.screen_note, render_config.COLOR_PROGRESS_BAR_BG, (self.screen_width - bar_width - 10, 10, bar_width, bar_height))
            progress_width = int(bar_width * (game_state.get_victory_progress(GameTeam.BLUE) / self.env_config.OCCUPATION_TARGET))
            pygame.draw.rect(self.screen_note, render_config.TEAM_COLORS[GameTeam.BLUE], 
                            (self.screen_width - bar_width - 10 + (bar_width - progress_width), 10, progress_width, bar_height))
            blue_progress_text = self.font_medium.render(
                f"Team BLUE: {int(game_state.get_victory_progress(GameTeam.BLUE)):>3d} / {self.env_config.OCCUPATION_TARGET:>3d}",
                True, render_config.COLOR_TEXT
            )
            self.screen_note.blit(blue_progress_text, (self.screen_width - bar_width, blue_progress_text.get_height() // 2))

            # 经济
            pygame.draw.circle(self.screen_note, render_config.COLOR_YELLOW, (self.screen_width // 2, bar_height * 3), bar_height * 0.5)
            red_economics_text = self.font_small.render(f"{game_state.get_economics(GameTeam.RED):>4d}", True, render_config.COLOR_TEXT)
            self.screen_note.blit(red_economics_text, (self.screen_width // 2 - bar_height - red_economics_text.get_width(), bar_height * 3 - red_economics_text.get_height() // 2))
            blue_economics_text = self.font_small.render(f"{game_state.get_economics(GameTeam.BLUE):<4d}", True, render_config.COLOR_TEXT)
            self.screen_note.blit(blue_economics_text, (self.screen_width // 2 + bar_height, bar_height * 3 - blue_economics_text.get_height() // 2))

    def _draw_control_info(self, control_state, env_name):
        """绘制控制提示"""
        # 绘制控制键位提示
        controls = [
            "Controls:",
            "ESC: Quit",
            "TAB: Toggle movable grid display",
            "Left Mouse Button: Set target",
            "R: Reset Game",
            "W/S: Switch robot",
            "A/D: Switch target",
            "Q: Attack once",
            "E: Purchase ammo",
            f"Movable Grid: {'ON' if control_state.get('show_grid', False) else 'OFF'}",
        ]
        select = {
            "base_game": [0, 1, 2, 3, 4, 7, 9],
            "RMUL": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }

        controls_displayed = [controls[i] for i in select[env_name]]

        for i, text in enumerate(controls_displayed):
            text_control = self.font_medium.render(text, True, render_config.COLOR_TEXT)
            self.screen_note.blit(text_control, (10, meters_to_pixels(self.env_config.FIELD_HEIGHT) - (len(controls_displayed) - i) * 30))

        # 绘制控制目标提示
        if env_name == "base_game":
            return
        
        states = [
            f"Robot: {control_state['robot_id']}",
            f"Target: {control_state['target_id']}",
        ]
        for i, text in enumerate(states):
            text_control = self.font_medium.render(text, True, render_config.COLOR_TEXT)
            self.screen_note.blit(text_control, (self.screen_width / 2 + 10, meters_to_pixels(self.env_config.FIELD_HEIGHT) - (len(states) - i) * 30))

    def _draw_game_over(self, game_state):
        # 绘制游戏结束信息
        if game_state == GameState.RED_TEAM_WIN:
            win_text = self.font_large.render("Team Red Wins!", True, render_config.TEAM_COLORS[GameTeam.RED])
            text_rect = win_text.get_rect(center=(meters_to_pixels(self.env_config.FIELD_WIDTH) // 2, 
                                                 meters_to_pixels(self.env_config.FIELD_HEIGHT) // 2))
            self.screen_note.blit(win_text, text_rect)
        elif game_state == GameState.BLUE_TEAM_WIN:
            win_text = self.font_large.render("Team Blue Wins!", True, render_config.TEAM_COLORS[GameTeam.BLUE])
            text_rect = win_text.get_rect(center=(meters_to_pixels(self.env_config.FIELD_WIDTH) // 2, 
                                                 meters_to_pixels(self.env_config.FIELD_HEIGHT) // 2))
            self.screen_note.blit(win_text, text_rect)
        elif game_state == GameState.DRAW:
            win_text = self.font_large.render("Draw!", True, render_config.COLOR_TEXT)  # 灰色字体
            text_rect = win_text.get_rect(center=(meters_to_pixels(self.env_config.FIELD_WIDTH) // 2, 
                                                 meters_to_pixels(self.env_config.FIELD_HEIGHT) // 2))
            self.screen_note.blit(win_text, text_rect)

    def _draw_grid(self, grid_map):
        """绘制可移动栅格"""
        # 检查栅格状态是否改变
        current_state = (grid_map.grid_cols, grid_map.grid_rows, grid_map.grid_blocked)
        if self.last_grid_state == current_state:
            # 如果状态没变，直接返回
            return

        # 更新缓存状态
        self.last_grid_state = current_state
        
        # 清空surface
        self.screen_grid.fill((0, 0, 0, 0))
        
        # 计算栅格大小（像素）
        cell_size_px = meters_to_pixels(grid_map.cell_size)
        
        # 绘制所有可移动栅格
        for col in range(grid_map.grid_cols):
            for row in range(grid_map.grid_rows):
                if not grid_map.is_blocked(col, row):
                    x = meters_to_pixels(col * grid_map.cell_size)
                    y = meters_to_pixels(row * grid_map.cell_size)
                    pygame.draw.rect(self.screen_grid, render_config.COLOR_GRID, 
                                   (x, y, cell_size_px, cell_size_px), 1)

    def _draw_path(self, robot):
        """绘制路径"""
        point1 = robot.get_position()
        for i in range(robot.current_path_idx, len(robot.path_points) - 1):
            point2 = robot.path_points[i]
            # 将世界坐标转换为像素坐标，并加上半个栅格的大小使其居中
            x1 = meters_to_pixels(point1[0])
            y1 = meters_to_pixels(point1[1])
            x2 = meters_to_pixels(point2[0])
            y2 = meters_to_pixels(point2[1])
            pygame.draw.line(self.screen_note, (0, 120, 120), (x1, y1), (x2, y2), 2)
            point1 = point2