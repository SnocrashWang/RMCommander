import pymunk
import math

class Obstacle:
    def __init__(self, physics_engine, obstacle_config):
        # obstacle_config 需要包含: x1, y1, x2, y2, thickness
        x1 = obstacle_config["x1"]
        y1 = obstacle_config["y1"]
        x2 = obstacle_config["x2"]
        y2 = obstacle_config["y2"]
        self.p1 = (x1, y1)
        self.p2 = (x2, y2)
        self.thickness = obstacle_config["thickness"]

        # 计算中心、长度、角度
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)
        self.length = math.hypot(x2 - x1, y2 - y1)
        self.angle = math.atan2(y2 - y1, x2 - x1)  # 弧度

        # 创建物理体
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = self.center

        # 计算矩形的四个顶点（相对于中心点）
        half_length = self.length / 2
        half_thickness = self.thickness / 2
        
        # 计算矩形的四个顶点（相对于中心点）
        self.vertices = [
            (-half_length, -half_thickness),  # 左下
            (half_length, -half_thickness),   # 右下
            (half_length, half_thickness),    # 右上
            (-half_length, half_thickness)    # 左上
        ]
        
        # 创建多边形形状
        shape = pymunk.Poly(body, self.vertices)
        shape.elasticity = 0.8
        shape.friction = 0.5

        # 设置旋转角度
        body.angle = self.angle

        physics_engine.add(body, shape)
        self.shape = shape
