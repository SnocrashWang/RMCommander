import math

class Obstacle:
    def __init__(self, obstacle_config):
        # obstacle_config 需要包含: p1, p2, thickness
        self.p1 = obstacle_config["p1"]
        self.p2 = obstacle_config["p2"]
        self.thickness = obstacle_config["thickness"]

        # 计算中心、长度、角度
        self.center = ((self.p1[0] + self.p2[0]) / 2, (self.p1[1] + self.p2[1]) / 2)
        self.length = math.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1])
        self.angle = math.atan2(self.p2[1] - self.p1[1], self.p2[0] - self.p1[0])  # 弧度

        # 计算矩形的四个顶点（相对于中心点）
        half_length = self.length / 2
        half_thickness = self.thickness / 2
        self.vertices = [
            (-half_length, -half_thickness),  # 左下
            (half_length, -half_thickness),   # 右下
            (half_length, half_thickness),    # 右上
            (-half_length, half_thickness)    # 左上
        ]
