from typing import Dict
from utils.config.game_config import GameState, GameTeam
from utils.config.robot_config import ROBOT_ID, RobotType
from base_game.config.env_config import GAME_TIME_LIMIT

class GameStateManager:
    def __init__(self):
        self.state = GameState.PLAYING
        self.center_zone_progress = {GameTeam.RED: 0, GameTeam.BLUE: 0}  # 红蓝队的进度
        self.total_time = GAME_TIME_LIMIT  # 游戏总时长
        self.remaining_time = GAME_TIME_LIMIT  # 剩余时间

    def update(self, robot_hp: Dict[str, int], dt: float):
        """更新游戏状态
        Args:
            robot_hp: 机器人血量
            dt: 时间增量
        """
        if self.state != GameState.PLAYING:
            return

        # 倒计时减少
        self.remaining_time = max(0, self.remaining_time - dt)

        # 检查胜利条件
        red_hp = robot_hp[ROBOT_ID[GameTeam.RED][RobotType.STANDARD_3]]
        blue_hp = robot_hp[ROBOT_ID[GameTeam.BLUE][RobotType.STANDARD_3]]
        # 1. 有一方率先阵亡
        if red_hp <= 0:
            self.state = GameState.BLUE_TEAM_WIN
        elif blue_hp <= 0:
            self.state = GameState.RED_TEAM_WIN
        # 2. 时间到
        elif self.remaining_time <= 0:
            if red_hp > blue_hp:
                self.state = GameState.RED_TEAM_WIN
            elif blue_hp > red_hp:
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
