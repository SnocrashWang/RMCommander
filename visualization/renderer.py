import math
import pygame
from visualization.config import render_config
from utils.config.game_config import GameState, GameTeam
from utils.config.exp_prop_config import LEVEL_NEED_EXP
from utils.utils import meters_to_pixels, draw_dashed_line, second2minute

class Renderer:
    def __init__(self, env_config):
        self.env_config = env_config

        # 计算屏幕尺寸
        screen_width = int(self.env_config.FIELD_WIDTH * render_config.SCALE)
        screen_height = int(self.env_config.FIELD_HEIGHT * render_config.SCALE)
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Robot Battle Environment")

        # 创建栅格缓存
        self.grid_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        self.grid_surface.set_alpha(render_config.GRID_ALPHA)
        self.last_grid_state = None  # 用于跟踪栅格状态

        self.large_font = pygame.font.SysFont(None, 48)
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.tiny_font = pygame.font.Font(None, 12)

    def render(self, env, show_grid=False):
        """渲染环境"""
        # 清空屏幕
        self.screen.fill(render_config.COLOR_BACKGROUND)

        # 绘制增益区域
        self._draw_buff_zone()

        # 绘制四周墙壁
        self._draw_walls()

        # 绘制障碍物
        self._draw_obstacles(env.obstacles)

        # 绘制所有机器人
        for robot in env.robots.values():
            self._draw_tank(robot)

        # 绘制攻击线
        for robot in env.robots.values():
            self._draw_attack_line(robot.get_attack_line())

        # 绘制进度条
        if self.env_config.ENV_NAME == "RMUL":
            self._draw_progress_bars(env.game_state_manager)

        # 绘制状态信息
        self._draw_info(env.game_state_manager, show_grid)

        # 绘制可移动栅格（如果启用）
        if show_grid:
            self._draw_grid(env.robots["RED_3_STANDARD"].grid_map)
            self._draw_path(env.robots["RED_3_STANDARD"])

    def _draw_buff_zone(self):
        """绘制增益区域"""
        # TODO: 绘制增益区域

    def _draw_walls(self):
        """绘制围墙"""
        # 上墙
        pygame.draw.rect(self.screen, render_config.COLOR_WALL, (0, 0, render_config.SCALE * self.env_config.FIELD_WIDTH, 10))
        # 下墙
        bottom_y = render_config.SCALE * self.env_config.FIELD_HEIGHT - 10
        pygame.draw.rect(self.screen, render_config.COLOR_WALL, (0, bottom_y, render_config.SCALE * self.env_config.FIELD_WIDTH, 10))
        # 左墙
        pygame.draw.rect(self.screen, render_config.COLOR_WALL, (0, 0, 10, render_config.SCALE * self.env_config.FIELD_HEIGHT))
        # 右墙
        right_x = render_config.SCALE * self.env_config.FIELD_WIDTH - 10
        pygame.draw.rect(self.screen, render_config.COLOR_WALL, (right_x, 0, 10, render_config.SCALE * self.env_config.FIELD_HEIGHT))

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
            v1 = (x1 + nx * half_t, y1 + ny * half_t)
            v2 = (x1 - nx * half_t, y1 - ny * half_t)
            v3 = (x2 - nx * half_t, y2 - ny * half_t)
            v4 = (x2 + nx * half_t, y2 + ny * half_t)

            def to_px(p):
                return int(p[0] * render_config.SCALE), int(p[1] * render_config.SCALE)

            pygame.draw.polygon(self.screen, render_config.COLOR_OBSTACLE, [to_px(v1), to_px(v2), to_px(v3), to_px(v4)])

    def _draw_tank(self, robot):
        """绘制机器人"""
        position = robot.get_position()
        x = meters_to_pixels(position[0])
        y = meters_to_pixels(position[1])
        radius = meters_to_pixels(robot.radius)

        # 绘制机器人主体
        pygame.draw.circle(self.screen, render_config.ROBOT_COLORS[robot.team], (x, y), radius)

        # 绘制炮塔（小方形）
        turret_size = radius * 0.6
        turret_rect = pygame.Rect(x - turret_size/2, y - turret_size/2, turret_size, turret_size)
        pygame.draw.rect(self.screen, (30, 30, 30), turret_rect)

        # 绘制炮管
        angle_rad = math.radians(robot.angle)
        barrel_length = radius * 0.8
        end_x = x + barrel_length * math.cos(angle_rad)
        end_y = y + barrel_length * math.sin(angle_rad)
        pygame.draw.line(self.screen, (30, 30, 30), (x, y), (end_x, end_y), int(2))

        # 绘制机器人标识（使用小字体）
        team_text = self.tiny_font.render(f"{robot.id}", True, render_config.COLOR_TEXT)
        self.screen.blit(team_text, (x - team_text.get_width() / 2, y + radius * 1.5))


        # 绘制数据条
        bar_width = render_config.SCALE * 0.5
        bar_height = render_config.SCALE * 0.08
        # 血量条
        hp_bar_x = x - bar_width / 2
        hp_bar_y = y - radius * 1.5 - bar_height * 2
        current_hp_width = int(bar_width * robot.hp / robot.max_hp)
        pygame.draw.rect(self.screen, render_config.TEAM_COLORS[robot.team],
                        (hp_bar_x, hp_bar_y, current_hp_width, bar_height))
        try:
            hp_text = self.tiny_font.render(f"{robot.hp:>3d}/{robot.max_hp:>3d}", True, render_config.COLOR_TEXT)
        except:
            print(robot.hp, robot.max_hp, robot.id)
            exit()
        self.screen.blit(hp_text, (hp_bar_x + bar_width / 2 - hp_text.get_width() / 2, hp_bar_y + bar_height / 2 - hp_text.get_height() / 2))

        # 热量条
        heat_bar_x = x - bar_width / 2
        heat_bar_y = y - radius * 1.5 - bar_height
        current_heat_width = int(bar_width * robot.heat / robot.max_heat)
        pygame.draw.rect(self.screen, render_config.COLOR_HEAT_BAR,
                        (heat_bar_x, heat_bar_y, current_heat_width, bar_height))
        heat_text = self.tiny_font.render(f"{int(robot.heat)}/{int(robot.max_heat)}", True, render_config.COLOR_TEXT)
        self.screen.blit(heat_text, (heat_bar_x + bar_width / 2 - heat_text.get_width() / 2, heat_bar_y + bar_height / 2 - heat_text.get_height() / 2))

        # 经验条
        exp_bar_x = x - bar_width / 2
        exp_bar_y = y - radius * 1.5
        if robot.level < len(LEVEL_NEED_EXP):
            exp_need_to_level_up = LEVEL_NEED_EXP[robot.level + 1] - LEVEL_NEED_EXP[robot.level]
        else:
            exp_need_to_level_up = LEVEL_NEED_EXP[robot.level] - LEVEL_NEED_EXP[robot.level - 1]
        current_exp_width = int(bar_width * min(1, (robot.exp - LEVEL_NEED_EXP[min(robot.level, len(LEVEL_NEED_EXP) - 1)]) / exp_need_to_level_up))
        pygame.draw.rect(self.screen, render_config.COLOR_EXP_BAR,
                        (exp_bar_x, exp_bar_y, current_exp_width, bar_height))
        exp_text = self.tiny_font.render(f"{robot.exp - LEVEL_NEED_EXP[robot.level]:>3d}/{exp_need_to_level_up:>3d}", True, render_config.COLOR_TEXT)
        self.screen.blit(exp_text, (exp_bar_x + bar_width / 2 - exp_text.get_width() / 2, exp_bar_y + bar_height / 2 - exp_text.get_height() / 2))
        # 等级
        level_text = self.tiny_font.render(f"Lv.{robot.level:>2d}", True, render_config.COLOR_TEXT)
        self.screen.blit(level_text, (exp_bar_x - level_text.get_width(), exp_bar_y + bar_height / 2 - level_text.get_height() / 2))

    def _draw_attack_line(self, attack_line):
        """绘制攻击线
        Args:
            attack_line: 攻击线，如果为None，则不绘制
        """
        if attack_line:
            draw_dashed_line(
                self.screen, (0, 255, 0), attack_line[0], attack_line[1],
                int(render_config.SCALE * 0.02), int(render_config.SCALE * 0.05), int(render_config.SCALE * 0.1)
            )

    def _draw_progress_bars(self, game_state):
        """绘制进度条"""
        bar_height = render_config.SCALE * self.env_config.FIELD_HEIGHT * 0.02
        screen_width = render_config.SCALE * self.env_config.FIELD_WIDTH
        bar_width = screen_width * 0.4

        # 红队进度条
        pygame.draw.rect(self.screen, render_config.COLOR_PROGRESS_BAR_BG, (10, 10, bar_width, bar_height))
        progress_width = int(bar_width * (game_state.center_zone_progress[GameTeam.RED] / self.env_config.OCCUPATION_TARGET))
        pygame.draw.rect(self.screen, render_config.TEAM_COLORS[GameTeam.RED],
                         (10, 10, progress_width, bar_height))

        # 蓝队进度条
        pygame.draw.rect(self.screen, render_config.COLOR_PROGRESS_BAR_BG, (screen_width - bar_width - 10, 10, bar_width, bar_height))
        progress_width = int(bar_width * (game_state.center_zone_progress[GameTeam.BLUE] / self.env_config.OCCUPATION_TARGET))
        pygame.draw.rect(self.screen, render_config.TEAM_COLORS[GameTeam.BLUE], 
                         (screen_width - bar_width - 10 + (bar_width - progress_width), 10, progress_width, bar_height))

        # 进度文本
        text1 = self.font.render(
            f"Team {GameTeam.RED.value}: {int(game_state.center_zone_progress[GameTeam.RED] / self.env_config.OCCUPATION_TARGET * 100):>3d}%",
            True, render_config.COLOR_TEXT
        )
        text2 = self.font.render(
            f"Team {GameTeam.BLUE.value}: {int(game_state.center_zone_progress[GameTeam.BLUE] / self.env_config.OCCUPATION_TARGET * 100):>3d}%",
            True, render_config.COLOR_TEXT
        )
        self.screen.blit(text1, (bar_width - text1.get_width(), text1.get_height() // 2))
        self.screen.blit(text2, (screen_width - bar_width, text2.get_height() // 2))

        # 倒计时
        min, sec = second2minute(game_state.get_remaining_time())
        text3 = self.font.render(f"Time: {min:02d}:{sec:02d}", True, render_config.COLOR_TEXT)
        self.screen.blit(text3, ((screen_width - text3.get_width()) // 2, text3.get_height() // 2))

    def _draw_info(self, game_state, show_grid=False):
        """Draw game info and controls (English)"""
        controls = [
            "Controls:",
            "Left Mouse Button: Set target",
            "R: Reset Game",
            "G: Toggle movable grid display",
            "A: Attack once",
            "ESC: Quit",
            f"Movable Grid: {'ON' if show_grid else 'OFF'}",
        ]

        for i, text in enumerate(controls):
            text_control = self.font.render(text, True, render_config.COLOR_TEXT)
            self.screen.blit(text_control, (10, render_config.SCALE * self.env_config.FIELD_HEIGHT - (len(controls) - i) * 30))

        # Show game state
        if game_state.state == GameState.RED_TEAM_WIN:
            win_text = self.large_font.render("Team Red Wins!", True, render_config.TEAM_COLORS[GameTeam.RED])
            text_rect = win_text.get_rect(center=(render_config.SCALE * self.env_config.FIELD_WIDTH // 2, 
                                                 render_config.SCALE * self.env_config.FIELD_HEIGHT // 2))
            self.screen.blit(win_text, text_rect)
        elif game_state.state == GameState.BLUE_TEAM_WIN:
            win_text = self.large_font.render("Team Blue Wins!", True, render_config.TEAM_COLORS[GameTeam.BLUE])
            text_rect = win_text.get_rect(center=(render_config.SCALE * self.env_config.FIELD_WIDTH // 2, 
                                                 render_config.SCALE * self.env_config.FIELD_HEIGHT // 2))
            self.screen.blit(win_text, text_rect)
        elif game_state.state == GameState.DRAW:
            win_text = self.large_font.render("Draw!", True, render_config.COLOR_TEXT)  # 灰色字体
            text_rect = win_text.get_rect(center=(render_config.SCALE * self.env_config.FIELD_WIDTH // 2, 
                                                 render_config.SCALE * self.env_config.FIELD_HEIGHT // 2))
            self.screen.blit(win_text, text_rect)

    def _draw_grid(self, grid_map):
        """绘制可移动栅格"""
        # 检查栅格状态是否改变
        current_state = (grid_map.grid_cols, grid_map.grid_rows, grid_map.cell_size)
        if self.last_grid_state == current_state:
            # 如果状态没变，直接使用缓存的surface
            self.screen.blit(self.grid_surface, (0, 0))
            return

        # 更新缓存状态
        self.last_grid_state = current_state
        
        # 清空surface
        self.grid_surface.fill((0, 0, 0, 0))
        
        # 计算栅格大小（像素）
        cell_size_px = int(grid_map.cell_size * render_config.SCALE)
        
        # 绘制所有可移动栅格
        for col in range(grid_map.grid_cols):
            for row in range(grid_map.grid_rows):
                if not grid_map.is_blocked(col, row):
                    x = int(col * grid_map.cell_size * render_config.SCALE)
                    y = int(row * grid_map.cell_size * render_config.SCALE)
                    pygame.draw.rect(self.grid_surface, render_config.GRID_COLOR, 
                                   (x, y, cell_size_px, cell_size_px), 1)
        
        # 将栅格绘制到主屏幕
        self.screen.blit(self.grid_surface, (0, 0))

    def _draw_path(self, robot):
        """绘制路径"""
        point1 = robot.get_position()
        for i in range(robot.current_path_idx, len(robot.path_points) - 1):
            point2 = robot.path_points[i]
            # 将世界坐标转换为像素坐标，并加上半个栅格的大小使其居中
            x1 = int((point1[0]) * render_config.SCALE)
            y1 = int((point1[1]) * render_config.SCALE)
            x2 = int((point2[0]) * render_config.SCALE)
            y2 = int((point2[1]) * render_config.SCALE)
            pygame.draw.line(self.screen, (0, 120, 120), (x1, y1), (x2, y2), 2)
            point1 = point2