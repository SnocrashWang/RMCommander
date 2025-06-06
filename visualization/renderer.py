import math
import pygame
import utils.env_config as env_config
from utils.game_config import GameState, GameTeam
import utils.robot_config as robot_config
from utils.utils import meters_to_pixels, create_rect_from_center

class Renderer:
    def __init__(self, screen_width, screen_height):
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("RMUL")
        self.font = pygame.font.SysFont(None, 24)
        self.large_font = pygame.font.SysFont(None, 36)
        
    def render(self, env_state, obstacles, show_grid=False):
        """渲染整个环境"""
        # 绘制背景
        self.screen.fill(env_config.BACKGROUND)
        
        # 绘制中心增益区域
        self._draw_center_zone()
        
        # 绘制围墙
        self._draw_walls()
        
        # 绘制障碍物
        self._draw_obstacles(obstacles)
        
        # 绘制所有机器人（适配机器人列表）
        for robot_info in env_state["robots"]:
            self._draw_tank(
                robot_info["pos"],
                robot_info["angle"],
                robot_config.RED_COLOR if robot_info["team"] == GameTeam.RED else robot_config.BLUE_COLOR,
                robot_info["team"]
            )
        
        # 绘制进度条
        self._draw_progress_bars(env_state["progress"])
        
        # 绘制状态信息
        self._draw_info(env_state["state"], show_grid)
        
        # 绘制可移动栅格
        if show_grid:
            self._draw_movable_grid(env_state["grid_map"])
    
    def _draw_center_zone(self):
        """绘制中心区域"""
        center_x = env_config.FIELD_WIDTH / 2
        center_y = env_config.FIELD_HEIGHT / 2
        size = env_config.CENTER_ZONE_SIZE
        
        rect = create_rect_from_center(
            center_x, center_y, 
            size, size, 
            env_config.SCALE
        )
        pygame.draw.rect(self.screen, env_config.CENTER_ZONE_COLOR, rect)
    
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
    
    def _draw_tank(self, position, angle, color, team):
        """绘制机器人"""
        x = meters_to_pixels(position[0], env_config.SCALE)
        y = meters_to_pixels(position[1], env_config.SCALE)
        radius = meters_to_pixels(robot_config.TANK_RADIUS, env_config.SCALE)
        
        # 绘制机器人主体
        pygame.draw.circle(self.screen, color, (x, y), radius)
        
        # 绘制机器人炮管
        angle_rad = pygame.math.Vector2(1, 0).rotate(angle).angle_to(pygame.math.Vector2(1, 0))
        end_x = x + (radius + 15) * pygame.math.Vector2(1, 0).rotate(angle).x
        end_y = y + (radius + 15) * pygame.math.Vector2(1, 0).rotate(angle).y
        pygame.draw.line(self.screen, (30, 30, 30), (x, y), (end_x, end_y), 5)
        
        # 绘制机器人标识
        team_text = self.font.render(f"T{team}", True, (30, 30, 30))
        self.screen.blit(team_text, (x - 10, y - 10))
    
    def _draw_progress_bars(self, progress):
        """绘制进度条"""
        bar_height = 20
        screen_width = env_config.SCALE * env_config.FIELD_WIDTH
        bar_width = screen_width // 2 - 20
        
        # 队伍1进度条
        pygame.draw.rect(self.screen, env_config.PROGRESS_BAR_BG, (10, 10, bar_width, bar_height))
        progress_width = int(bar_width * (progress[1] / 100))
        pygame.draw.rect(self.screen, env_config.PROGRESS_BAR1, (10, 10, progress_width, bar_height))
        
        # 队伍2进度条
        pygame.draw.rect(self.screen, env_config.PROGRESS_BAR_BG, (screen_width - bar_width - 10, 10, bar_width, bar_height))
        progress_width = int(bar_width * (progress[2] / 100))
        pygame.draw.rect(self.screen, env_config.PROGRESS_BAR2, 
                        (screen_width - bar_width - 10 + (bar_width - progress_width), 10, 
                         progress_width, bar_height))
        
        # 进度文本
        text1 = self.font.render(f"Team 1: {int(progress[1])}%", True, env_config.TEXT_COLOR)
        text2 = self.font.render(f"Team 2: {int(progress[2])}%", True, env_config.TEXT_COLOR)
        self.screen.blit(text1, (20, 15))
        self.screen.blit(text2, (screen_width - bar_width - 10 + 20, 15))
    
    def _draw_info(self, game_state, show_grid=False):
        """Draw game info and controls (English)"""
        controls = [
            "Controls:",
            "Left Mouse Button: Set target for Robot1",
            "R: Reset Game",
            "ESC: Quit",
            "G: Toggle movable grid display"
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
    
    def _draw_movable_grid(self, grid_map):
        """绘制可移动栅格（未被障碍物阻挡的格子）"""
        for row in range(grid_map.rows):
            for col in range(grid_map.cols):
                if not grid_map.is_blocked(col, row):
                    x = int(col * grid_map.cell_size * env_config.SCALE)
                    y = int(row * grid_map.cell_size * env_config.SCALE)
                    size = int(grid_map.cell_size * env_config.SCALE)
                    rect = pygame.Rect(x, y, size, size)
                    pygame.draw.rect(self.screen, (200, 255, 200), rect, 1)  # 绿色细线