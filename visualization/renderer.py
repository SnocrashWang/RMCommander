import math
import pygame
from utils import env_config
from utils.game_config import GameState, GameTeam
from utils import robot_config
from utils.utils import meters_to_pixels, create_rect_from_center

class Renderer:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Robot Battle Environment")
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.SysFont(None, 48)
        
        # 创建栅格缓存
        self.grid_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        self.grid_surface.set_alpha(env_config.GRID_ALPHA)
        self.last_grid_state = None  # 用于跟踪栅格状态是否改变

    def render(self, env_state, obstacles, show_grid=False):
        """渲染环境"""
        # 清空屏幕
        self.screen.fill(env_config.BACKGROUND_COLOR)

        # 绘制中心区域
        self._draw_center_zone()

        # 绘制墙壁
        self._draw_walls()

        # 绘制障碍物
        self._draw_obstacles(obstacles)

        # 绘制所有机器人
        for robot in env_state["robots"]:
            self._draw_tank(robot)

        # 绘制攻击线（如果有）
        if len(env_state["robots"]) > 1:
            attack_line = env_state["robots"][0].get_attack_line(env_state["robots"][1])
            if attack_line:
                self._draw_attack_line(*attack_line)

        # 绘制进度条
        self._draw_progress_bars(env_state["center_zone_progress"])

        # 绘制状态信息
        self._draw_info(env_state["state"], show_grid)

        # 绘制可移动栅格（如果启用）
        if show_grid:
            self._draw_grid(env_state["grid_map"])

    def _draw_center_zone(self):
        """绘制中心区域"""
        center_zone_rect = env_config.CENTER_ZONE_RECT
        pygame.draw.rect(self.screen, env_config.CENTER_ZONE_COLOR, center_zone_rect)  # 更浅的绿色

    def _draw_walls(self):
        """绘制围墙"""
        # 上墙
        pygame.draw.rect(self.screen, env_config.WALL_COLOR, (0, 0, env_config.SCALE * env_config.FIELD_WIDTH, 10))
        # 下墙
        bottom_y = env_config.SCALE * env_config.FIELD_HEIGHT - 10
        pygame.draw.rect(self.screen, env_config.WALL_COLOR, (0, bottom_y, env_config.SCALE * env_config.FIELD_WIDTH, 10))
        # 左墙
        pygame.draw.rect(self.screen, env_config.WALL_COLOR, (0, 0, 10, env_config.SCALE * env_config.FIELD_HEIGHT))
        # 右墙
        right_x = env_config.SCALE * env_config.FIELD_WIDTH - 10
        pygame.draw.rect(self.screen, env_config.WALL_COLOR, (right_x, 0, 10, env_config.SCALE * env_config.FIELD_HEIGHT))

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
                return int(p[0] * env_config.SCALE), int(p[1] * env_config.SCALE)

            pygame.draw.polygon(self.screen, env_config.OBSTACLE_COLOR, [to_px(v1), to_px(v2), to_px(v3), to_px(v4)])

    def _draw_tank(self, robot):
        """绘制机器人"""
        position = robot.get_position()
        x = meters_to_pixels(position[0], env_config.SCALE)
        y = meters_to_pixels(position[1], env_config.SCALE)
        radius = meters_to_pixels(robot_config.TANK_RADIUS, env_config.SCALE)

        # 绘制机器人主体
        pygame.draw.circle(self.screen, robot.color, (x, y), radius)

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
        team_text = self.small_font.render(f"{robot.team}", True, (30, 30, 30))
        self.screen.blit(team_text, (x - 50, y - 50))

        # 绘制血量条
        hp_bar_width = radius * 2
        hp_bar_height = radius * 0.2
        hp_bar_x = x - hp_bar_width / 2
        hp_bar_y = y - radius - hp_bar_height - 5

        # 血量条背景
        pygame.draw.rect(self.screen, (60, 60, 60), 
                        (hp_bar_x, hp_bar_y, hp_bar_width, hp_bar_height))
        
        # 当前血量
        hp_percentage = robot.get_hp_percentage()
        current_hp_width = int(hp_bar_width * hp_percentage)
        hp_color = (0, 255, 0) if hp_percentage > 0.5 else (255, 165, 0) if hp_percentage > 0.25 else (255, 0, 0)
        pygame.draw.rect(self.screen, hp_color,
                        (hp_bar_x, hp_bar_y, current_hp_width, hp_bar_height))

    def _draw_attack_line(self, start_pos, end_pos):
        """绘制攻击线
        Args:
            start_pos: 起点坐标 (x, y)
            end_pos: 终点坐标 (x, y)
        """
        start_x = meters_to_pixels(start_pos[0], env_config.SCALE)
        start_y = meters_to_pixels(start_pos[1], env_config.SCALE)
        end_x = meters_to_pixels(end_pos[0], env_config.SCALE)
        end_y = meters_to_pixels(end_pos[1], env_config.SCALE)
        
        # 绘制虚线
        dash_length = 5
        gap_length = 10
        dx = end_x - start_x
        dy = end_y - start_y
        distance = math.hypot(dx, dy)
        if distance > 0:
            dx, dy = dx / distance, dy / distance
            current_pos = (start_x, start_y)
            while distance > 0:
                # 绘制一段虚线
                next_pos = (
                    current_pos[0] + dx * min(dash_length, distance),
                    current_pos[1] + dy * min(dash_length, distance)
                )
                pygame.draw.line(self.screen, (0, 255, 0), current_pos, next_pos, 2)
                
                # 移动到下一段虚线的起点
                current_pos = (
                    next_pos[0] + dx * min(gap_length, distance - dash_length),
                    next_pos[1] + dy * min(gap_length, distance - dash_length)
                )
                distance -= (dash_length + gap_length)

    def _draw_progress_bars(self, progress):
        """绘制进度条"""
        bar_height = 20
        screen_width = env_config.SCALE * env_config.FIELD_WIDTH
        bar_width = screen_width // 2 - 20

        # 红队进度条
        pygame.draw.rect(self.screen, env_config.PROGRESS_BAR_BG, (10, 10, bar_width, bar_height))
        progress_width = int(bar_width * (progress[GameTeam.RED] / 100))
        pygame.draw.rect(self.screen, env_config.PROGRESS_BAR_RED, (10, 10, progress_width, bar_height))

        # 蓝队进度条
        pygame.draw.rect(self.screen, env_config.PROGRESS_BAR_BG, (screen_width - bar_width - 10, 10, bar_width, bar_height))
        progress_width = int(bar_width * (progress[GameTeam.BLUE] / 100))
        pygame.draw.rect(self.screen, env_config.PROGRESS_BAR_BLUE, 
                        (screen_width - bar_width - 10 + (bar_width - progress_width), 10, 
                         progress_width, bar_height))

        # 进度文本
        text1 = self.font.render(f"Team 1: {int(progress[GameTeam.RED])}%", True, env_config.TEXT_COLOR)
        text2 = self.font.render(f"Team 2: {int(progress[GameTeam.BLUE])}%", True, env_config.TEXT_COLOR)
        self.screen.blit(text1, (20, 15))
        self.screen.blit(text2, (screen_width - bar_width - 10 + 20, 15))

    def _draw_info(self, game_state, show_grid=False):
        """Draw game info and controls (English)"""
        controls = [
            "Controls:",
            "Left Mouse Button: Set target for Robot1",
            "R: Reset Game",
            "G: Toggle movable grid display"
            "ESC: Quit",
        ]

        for i, text in enumerate(controls):
            text_surface = self.font.render(text, True, env_config.TEXT_COLOR)
            self.screen.blit(text_surface, (10, env_config.SCALE * env_config.FIELD_HEIGHT - 150 + i * 30))

        # Show grid status
        grid_status = "ON" if show_grid else "OFF"
        grid_text = self.font.render(f"Movable Grid: {grid_status}", True, env_config.TEXT_COLOR)
        self.screen.blit(grid_text, (10, env_config.SCALE * env_config.FIELD_HEIGHT - 150 + len(controls) * 30))

        # Show game state
        if game_state == GameState.RED_TEAM_WIN:
            win_text = self.large_font.render("Team 1 Wins!", True, robot_config.RED_COLOR)
            text_rect = win_text.get_rect(center=(env_config.SCALE * env_config.FIELD_WIDTH // 2, 
                                                 env_config.SCALE * env_config.FIELD_HEIGHT // 2))
            self.screen.blit(win_text, text_rect)
        elif game_state == GameState.BLUE_TEAM_WIN:
            win_text = self.large_font.render("Team 2 Wins!", True, robot_config.BLUE_COLOR)
            text_rect = win_text.get_rect(center=(env_config.SCALE * env_config.FIELD_WIDTH // 2, 
                                                 env_config.SCALE * env_config.FIELD_HEIGHT // 2))
            self.screen.blit(win_text, text_rect)
        elif game_state == GameState.DRAW:
            win_text = self.large_font.render("Draw!", True, (128, 128, 128))  # 灰色字体
            text_rect = win_text.get_rect(center=(env_config.SCALE * env_config.FIELD_WIDTH // 2, 
                                                 env_config.SCALE * env_config.FIELD_HEIGHT // 2))
            self.screen.blit(win_text, text_rect)

    def _draw_grid(self, grid_map):
        """绘制可移动栅格"""
        # 检查栅格状态是否改变
        current_state = (grid_map.rows, grid_map.cols, grid_map.cell_size)
        if self.last_grid_state == current_state:
            # 如果状态没变，直接使用缓存的surface
            self.screen.blit(self.grid_surface, (0, 0))
            return
            
        # 更新缓存状态
        self.last_grid_state = current_state
        
        # 清空surface
        self.grid_surface.fill((0, 0, 0, 0))
        
        # 计算栅格大小（像素）
        cell_size_px = int(grid_map.cell_size * env_config.SCALE)
        
        # 绘制所有可移动栅格
        for row in range(grid_map.rows):
            for col in range(grid_map.cols):
                if not grid_map.is_blocked(col, row):
                    x = int(col * grid_map.cell_size * env_config.SCALE)
                    y = int(row * grid_map.cell_size * env_config.SCALE)
                    pygame.draw.rect(self.grid_surface, env_config.GRID_COLOR, 
                                   (x, y, cell_size_px, cell_size_px), 1)
        
        # 将栅格绘制到主屏幕
        self.screen.blit(self.grid_surface, (0, 0))
