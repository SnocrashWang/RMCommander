import pymunk
import utils.env_config as env_config

class PhysicsEngine:
    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # 无重力
        self.setup_boundaries()
    
    def setup_boundaries(self):
        """设置四周围墙"""
        thickness = 0.1  # 墙厚度
        
        # 创建围墙的物理形状
        walls = [
            # 上墙
            [(0, 0), (env_config.FIELD_WIDTH, 0), thickness],
            # 下墙
            [(0, env_config.FIELD_HEIGHT), (env_config.FIELD_WIDTH, env_config.FIELD_HEIGHT), thickness],
            # 左墙
            [(0, 0), (0, env_config.FIELD_HEIGHT), thickness],
            # 右墙
            [(env_config.FIELD_WIDTH, 0), (env_config.FIELD_WIDTH, env_config.FIELD_HEIGHT), thickness]
        ]
        
        for wall in walls:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            shape = pymunk.Segment(body, wall[0], wall[1], wall[2])
            shape.elasticity = 0.8
            shape.friction = 0.5
            self.space.add(body, shape)
    
    def add_object(self, body, shape):
        """添加物理对象到空间"""
        self.space.add(body, shape)
    
    def remove_object(self, body, shape):
        """从空间移除物理对象"""
        self.space.remove(body, shape)
    
    def step(self, dt):
        """推进物理仿真"""
        self.space.step(dt)
    
    def get_collision_handler(self):
        """获取碰撞处理器"""
        return self.space.add_default_collision_handler()