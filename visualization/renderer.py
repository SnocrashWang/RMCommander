import pygame
import utils.global_config as global_config
from utils.constants import GameState
from utils.utils import meters_to_pixels, create_rect_from_center

class Renderer:
    def __init__(self, screen_width, screen_height):
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("机器人对抗仿真环境")
        self.font = pygame.font.SysFont(None, 24)
        self.large_font = pygame.font.SysFont(None, 36)
        
    def render(self, env_state, obstacles):
        """渲染整个环境"""
        # 绘制背景
        self.screen.fill(global_config.BACKGROUND)
        
        # 绘制中心增益区域
        self._draw_center_zone()
        
        # 绘制围墙
        self._draw_walls()
        
        # 绘制障碍物
        self._draw_obstacles(obstacles)
        
        # 绘制机器人
        self._draw_tank(
            env_state["robot1_pos"], 
            env_state["robot1_angle"], 
            global_config.robot1_COLOR, 
            1
        )
        self._draw_tank(
            env_state["robot2_pos"], 
            env_state["robot2_angle"], 
            global_config.robot2_COLOR, 
            2
        )
        
        # 绘制进度条
        self._draw_progress_bars(env_state["progress"])
        
        # 绘制状态信息
        self._draw_info(env_state["state"])
    
    def _draw_center_zone(self):
        """绘制中心区域"""
        center_x = global_config.FIELD_WIDTH / 2
        center_y = global_config.FIELD_HEIGHT / 2
        size = global_config.CENTER_ZONE_SIZE
        
        rect = create_rect_from_center(
            center_x, center_y, 
            size, size, 
            global_config.SCALE
        )
        pygame.draw.rect(self.screen, global_config.CENTER_ZONE_COLOR, rect)
    
    def _draw_walls(self):
        """绘制围墙"""
        # 上墙
        pygame.draw.rect(self.screen, global_config.WALL_COLOR, (0, 0, global_config.SCALE * global_config.FIELD_WIDTH, 10))
        # 下墙
        bottom_y = global_config.SCALE * global_config.FIELD_HEIGHT - 10
        pygame.draw.rect(self.screen, global_config.WALL_COLOR, (0, bottom_y, global_config.SCALE * global_config.FIELD_WIDTH, 10))
        # 左墙
        pygame.draw.rect(self.screen, global_config.WALL_COLOR, (0, 0, 10, global_config.SCALE * global_config.FIELD_HEIGHT))
        # 右墙
        right_x = global_config.SCALE * global_config.FIELD_WIDTH - 10
        pygame.draw.rect(self.screen, global_config.WALL_COLOR, (right_x, 0, 10, global_config.SCALE * global_config.FIELD_HEIGHT))
    
    def _draw_obstacles(self, obstacles):
        """绘制障碍物"""
        for obstacle in obstacles:
            start_x = meters_to_pixels(obstacle.shape.a.x, global_config.SCALE)
            start_y = meters_to_pixels(obstacle.shape.a.y, global_config.SCALE)
            end_x = meters_to_pixels(obstacle.shape.b.x, global_config.SCALE)
            end_y = meters_to_pixels(obstacle.shape.b.y, global_config.SCALE)
            thickness = meters_to_pixels(obstacle.shape.radius * 2, global_config.SCALE)
            
            pygame.draw.line(
                self.screen, 
                global_config.OBSTACLE_COLOR, 
                (start_x, start_y), 
                (end_x, end_y), 
                int(thickness)
            )
    
    def _draw_tank(self, position, angle, color, team):
        """绘制机器人"""
        x = meters_to_pixels(position[0], global_config.SCALE)
        y = meters_to_pixels(position[1], global_config.SCALE)
        radius = meters_to_pixels(global_config.TANK_RADIUS, global_config.SCALE)
        
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
        screen_width = global_config.SCALE * global_config.FIELD_WIDTH
        bar_width = screen_width // 2 - 20
        
        # 队伍1进度条
        pygame.draw.rect(self.screen, global_config.PROGRESS_BAR_BG, (10, 10, bar_width, bar_height))
        progress_width = int(bar_width * (progress[1] / 100))
        pygame.draw.rect(self.screen, global_config.PROGRESS_BAR1, (10, 10, progress_width, bar_height))
        
        # 队伍2进度条
        pygame.draw.rect(self.screen, global_config.PROGRESS_BAR_BG, (screen_width - bar_width - 10, 10, bar_width, bar_height))
        progress_width = int(bar_width * (progress[2] / 100))
        pygame.draw.rect(self.screen, global_config.PROGRESS_BAR2, 
                        (screen_width - bar_width - 10 + (bar_width - progress_width), 10, 
                         progress_width, bar_height))
        
        # 进度文本
        text1 = self.font.render(f"Team 1: {int(progress[1])}%", True, global_config.TEXT_COLOR)
        text2 = self.font.render(f"Team 2: {int(progress[2])}%", True, global_config.TEXT_COLOR)
        self.screen.blit(text1, (20, 15))
        self.screen.blit(text2, (screen_width - bar_width - 10 + 20, 15))
    
    def _draw_info(self, game_state):
        """绘制游戏信息"""
        # 显示控制说明
        controls = [
            "Controls:",
            "Tank 1: WASD to move, Q/E to rotate",
            "Tank 2: Arrow keys to move, </> to rotate",
            "R: Reset game",
            "ESC: Quit"
        ]
        
        for i, text in enumerate(controls):
            text_surface = self.font.render(text, True, global_config.TEXT_COLOR)
            self.screen.blit(text_surface, (10, global_config.SCALE * global_config.FIELD_HEIGHT - 150 + i * 30))
        
        # 显示游戏状态
        if game_state == GameState.TEAM1_WIN:
            win_text = self.large_font.render("Team 1 Wins!", True, global_config.robot1_COLOR)
            text_rect = win_text.get_rect(center=(global_config.SCALE * global_config.FIELD_WIDTH // 2, 
                                                 global_config.SCALE * global_config.FIELD_HEIGHT // 2))
            self.screen.blit(win_text, text_rect)
        elif game_state == GameState.TEAM2_WIN:
            win_text = self.large_font.render("Team 2 Wins!", True, global_config.robot2_COLOR)
            text_rect = win_text.get_rect(center=(global_config.SCALE * global_config.FIELD_WIDTH // 2, 
                                                 global_config.SCALE * global_config.FIELD_HEIGHT // 2))
            self.screen.blit(win_text, text_rect)