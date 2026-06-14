import pymunk
import math

class Obstacle:
    def __init__(self, obstacle_config, physics_engine: pymunk.Space = None):
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

        if physics_engine:
            # 创建物理体
            self._body = pymunk.Body(body_type=pymunk.Body.STATIC)
            self._body.position = self.center
            # 设置旋转角度
            self._body.angle = self.angle
            
            # 创建多边形形状
            self._shape = pymunk.Poly(self._body, self.vertices)
            self._shape.elasticity = 0.8
            self._shape.friction = 0.5
            self._shape.filter = pymunk.ShapeFilter(categories=0b1, mask=0b1)

            physics_engine.add(self._body, self._shape)
