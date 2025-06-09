from utils.game_config import GameState, GAME_TIME_LIMIT, OCCUPATION_TARGET

class GameStateManager:
    def __init__(self):
        self.state = GameState.PLAYING
        self.center_zone_progress = {1: 0, 2: 0}
        self.total_time = GAME_TIME_LIMIT  # 游戏总时长
        self.remaining_time = GAME_TIME_LIMIT  # 剩余时间

    def update(self, robot1_in_zone, robot2_in_zone, dt):
        """更新游戏状态"""
        if self.state != GameState.PLAYING:
            return

        # 倒计时减少
        self.remaining_time = max(0, self.remaining_time - dt)

        # 更新中心区域进度
        if robot1_in_zone:
            self.center_zone_progress[1] += dt
        if robot2_in_zone:
            self.center_zone_progress[2] += dt

        # 检查胜利条件
        p1 = self.center_zone_progress[1]
        p2 = self.center_zone_progress[2]
        target = OCCUPATION_TARGET

        # 1. 有一方率先积满
        if p1 >= target > p2:
            self.state = GameState.RED_TEAM_WIN
        elif p2 >= target > p1:
            self.state = GameState.BLUE_TEAM_WIN
        # 2. 同时积满
        elif min(p1, p2) >= target:
            self.state = GameState.DRAW
        # 3. 时间到
        elif self.remaining_time <= 0:
            if p1 > p2:
                self.state = GameState.RED_TEAM_WIN
            elif p2 > p1:
                self.state = GameState.BLUE_TEAM_WIN
            else:
                self.state = GameState.DRAW

    def reset(self):
        """重置游戏状态"""
        self.state = GameState.PLAYING
        self.center_zone_progress = {1: 0, 2: 0}
        self.remaining_time = self.total_time

    def is_game_over(self):
        """检查游戏是否结束"""
        return self.state != GameState.PLAYING

    def get_remaining_time(self):
        return int(self.remaining_time)
