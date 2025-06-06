from utils.constants import GameState

class GameStateManager:
    def __init__(self):
        self.state = GameState.PLAYING
        self.center_zone_progress = {1: 0, 2: 0}
    
    def update(self, robot1_in_zone, robot2_in_zone, dt):
        """更新游戏状态"""
        if self.state != GameState.PLAYING:
            return
        
        # 更新中心区域进度
        if robot1_in_zone:
            self.center_zone_progress[1] += dt
        if robot2_in_zone:
            self.center_zone_progress[2] += dt
        
        # 检查胜利条件
        if self.center_zone_progress[1] >= 100:
            self.state = GameState.TEAM1_WIN
        elif self.center_zone_progress[2] >= 100:
            self.state = GameState.TEAM2_WIN
    
    def reset(self):
        """重置游戏状态"""
        self.state = GameState.PLAYING
        self.center_zone_progress = {1: 0, 2: 0}
    
    def is_game_over(self):
        """检查游戏是否结束"""
        return self.state != GameState.PLAYING