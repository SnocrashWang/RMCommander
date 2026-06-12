import numpy as np
from abc import ABC
from typing import Dict
from gymnasium import spaces


class Action(ABC):
    """动作抽象基类"""

    # 子类需要定义的类属性
    _schema: Dict = {}

    def __init__(self, **kwargs):
        """初始化动作"""
        self._validate_schema()
        self._init_actions(**kwargs)

    def __setattr__(self, name, value):
        """设置动作值时的验证"""
        if name.startswith('_') or name in ['space', '_schema']:
            super().__setattr__(name, value)
        elif name in self._schema:
            # 验证新值
            value = self._coerce_value(name, value)
            if self._contains_value(name, value):
                super().__setattr__(name, value)
            else:
                raise ValueError(f"Invalid value for {name}: {value}, which should belong to {self._schema[name]}")
        else:
            super().__setattr__(name, value)

    def __repr__(self):
        action_str = ", ".join([f"{k}={getattr(self, k)}" for k in self._schema])
        return f"{self.__class__.__name__}({action_str})"

    @classmethod
    def _validate_schema(cls):
        """验证 schema 的有效性"""
        if not hasattr(cls, '_schema') or not cls._schema:
            raise ValueError(f"{cls.__name__} must define '_schema' class attribute")

    @classmethod
    def get_space(cls) -> spaces.Dict:
        """获取动作空间"""
        return spaces.Dict(cls._schema)
    
    def _init_actions(self, **kwargs):
        """初始化动作值"""
        for name, space in self._schema.items():
            if name in kwargs:
                value = self._coerce_value(name, kwargs[name])
                # 验证输入值
                if not self._contains_value(name, value):
                    raise ValueError(f"Invalid value for {name}: {value}")
                setattr(self, name, value)
            else:
                # 设置默认值
                if isinstance(space, spaces.Box):
                    setattr(self, name, (space.low + space.high) / 2.0)
                elif isinstance(space, spaces.Discrete):
                    setattr(self, name, 0)
                else:
                    setattr(self, name, None)

    def _contains_value(self, name, value) -> bool:
        """Validate one action field against its own space."""
        return self._schema[name].contains(value)

    def _coerce_value(self, name, value):
        space = self._schema[name]
        if isinstance(space, spaces.Box):
            return np.asarray(value, dtype=space.dtype)
        if isinstance(space, spaces.Discrete):
            return int(value)
        return value

    # def get_action_dict(self) -> Dict:
    #     """获取动作字典"""
    #     return {name: getattr(self, name) for name in self._schema}

    def to_array(self) -> np.ndarray:
        """将所有的属性值转换为一个NumPy数组，实际上基本用不到"""
        array = []
        for name, space in self._schema.items():
            if isinstance(space, spaces.Box):
                array.extend([*getattr(self, name)])
            elif isinstance(space, spaces.Discrete):
                array.append(getattr(self, name))
        return np.array(array)


if __name__ == "__main__":
    class RobotAction(Action):
        _schema = {
            "navigation_target_norm": spaces.Box(
                low=np.array([-1.0, 0.5], dtype=np.float32), 
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            ),
            "navigation_set": spaces.Discrete(2),
            "attack_target": spaces.Discrete(5),
            "spin": spaces.Discrete(2),
        }

    robot_action = RobotAction(
        navigation_target_norm=np.array([0.2, 0.8], dtype=np.float32),
        navigation_set=0,
        attack_target=2,
        spin=1
    )
    # print(robot_action.get_action_dict())
    print(RobotAction.__dict__)
    print(robot_action.__dict__)
    robot_action.spin = 0
    print(robot_action)
    print(robot_action.to_array())
    print(RobotAction())
