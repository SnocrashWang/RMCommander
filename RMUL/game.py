from base.game import GameStateManager
from utils.config.game_config import GameState, GameTeam
from RMUL.config.env_config import GAME_TIME_LIMIT, OCCUPATION_TARGET

class GameStateManagerRMUL(GameStateManager):
    def __init__(self):
        self.state = GameState.PLAYING
        self.total_time = GAME_TIME_LIMIT       # 总时长
        self._remaining_time = GAME_TIME_LIMIT  # 剩余时间

        self._economics = {GameTeam.RED: 0, GameTeam.BLUE: 0}  # 经济
        self._center_zone_progress = {GameTeam.RED: 0, GameTeam.BLUE: 0}  # 中心增益点占领进度

    def update(self, robots_in_zone, dt):
        """更新游戏状态
        Args:
            robots_in_zone: 字典，key为队伍（GameTeam），value为该队伍是否有机器人在中心区域
            dt: 时间增量
        """
        if self.state != GameState.PLAYING:
            return

        # 倒计时减少
        self._remaining_time = max(0, self._remaining_time - dt)

        # 更新经济
        if self.reach_remaining_time(300, dt):
            self.gain_economics(GameTeam.RED, 200)
            self.gain_economics(GameTeam.BLUE, 200)
        elif self.reach_remaining_time(240, dt):
            self.gain_economics(GameTeam.RED, 200)
            self.gain_economics(GameTeam.BLUE, 200)
        elif self.reach_remaining_time(180, dt):
            self.gain_economics(GameTeam.RED, 200)
            self.gain_economics(GameTeam.BLUE, 200)
        elif self.reach_remaining_time(120, dt):
            self.gain_economics(GameTeam.RED, 300)
            self.gain_economics(GameTeam.BLUE, 300)
        elif self.reach_remaining_time(60, dt):
            self.gain_economics(GameTeam.RED, 300)
            self.gain_economics(GameTeam.BLUE, 300)

        # 更新中心区域进度
        for team, has_robot in robots_in_zone.items():
            if has_robot:  # 如果该队伍有机器人在中心区域
                self._center_zone_progress[team] += dt  # 只要有一个机器人在区域中就增加进度

        # 检查胜利条件
        red_progress = self._center_zone_progress[GameTeam.RED]
        blue_progress = self._center_zone_progress[GameTeam.BLUE]
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
        elif self._remaining_time <= 0:
            if red_progress > blue_progress:
                self.state = GameState.RED_TEAM_WIN
            elif blue_progress > red_progress:
                self.state = GameState.BLUE_TEAM_WIN
            else:
                self.state = GameState.DRAW

    def gain_economics(self, team: GameTeam, amount: int):
        self._economics[team] += amount

    def cost_economics(self, team: GameTeam, amount: int):
        self._economics[team] -= amount

    def get_economics(self, team: GameTeam):
        return self._economics[team]

    def get_center_zone_progress(self, team: GameTeam):
        return self._center_zone_progress[team]
