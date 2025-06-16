from base.game import GameStateManager
from utils.config.game_config import GameState, GameTeam
from RMUL.config.env_config import GAME_TIME_LIMIT, OCCUPATION_TARGET

class GameStateManagerRMUL(GameStateManager):
    def __init__(self):
        self.state = GameState.PLAYING
        self.center_zone_progress = {GameTeam.RED: 0, GameTeam.BLUE: 0}  # 红蓝队的进度
        self.total_time = GAME_TIME_LIMIT  # 游戏总时长
        self.remaining_time = GAME_TIME_LIMIT  # 剩余时间

    def update(self, robots_in_zone, dt):
        """更新游戏状态
        Args:
            robots_in_zone: 字典，key为队伍（GameTeam），value为该队伍是否有机器人在中心区域
            dt: 时间增量
        """
        if self.state != GameState.PLAYING:
            return

        # 倒计时减少
        self.remaining_time = max(0, self.remaining_time - dt)

        # 更新中心区域进度
        for team, has_robot in robots_in_zone.items():
            if has_robot:  # 如果该队伍有机器人在中心区域
                self.center_zone_progress[team] += dt  # 只要有一个机器人在区域中就增加进度

        # 检查胜利条件
        red_progress = self.center_zone_progress[GameTeam.RED]
        blue_progress = self.center_zone_progress[GameTeam.BLUE]
        target = OCCUPATION_TARGET

        # 1. 有一方率先积满
        if red_progress >= target > blue_progress:
            self.state = GameState.RED_TEAM_WIN
        elif blue_progress >= target > red_progress:
            self.state = GameState.BLUE_TEAM_WIN
        # 2. 同时积满
        elif min(red_progress, blue_progress) >= target:
            self.state = GameState.DRAW
        # 3. 时间到
        elif self.remaining_time <= 0:
            if red_progress > blue_progress:
                self.state = GameState.RED_TEAM_WIN
            elif blue_progress > red_progress:
                self.state = GameState.BLUE_TEAM_WIN
            else:
                self.state = GameState.DRAW

    def reset(self):
        """重置游戏状态"""
        self.state = GameState.PLAYING
        self.center_zone_progress = {GameTeam.RED: 0, GameTeam.BLUE: 0}
        self.remaining_time = self.total_time

    def get_remaining_time(self):
        return int(self.remaining_time)
