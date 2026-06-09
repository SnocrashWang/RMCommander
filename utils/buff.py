from enum import Enum
from typing import Dict, List, Tuple
from dataclasses import dataclass

class BuffType(Enum):
    ATTACK          = "attack"
    DEFENCE         = "defence"
    VULNERABILITY   = "vulnerability"
    COOLDOWN_RATE   = "cooldown_rate"
    COOLDOWN_CONST  = "cooldown_const"
    POWER           = "power"
    HEALING         = "healing"

@dataclass
class Buff:
    def __init__(self, name, **kwargs):
        self.name = name
        for buff, buff_value in kwargs.items():
            assert buff in [t.value for t in BuffType], f"Invalid buff name: {buff}"
            setattr(self, buff, buff_value)
    
    def __repr__(self):
        attrs = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"Buff({', '.join(attrs)})"

    def __eq__(self, other):
        """定义相等比较：比较 name 和所有 buff 值"""
        if not isinstance(other, Buff):
            return False
        if self.__dict__ != other.__dict__:
            return False
        for k in self.__dict__:
            if getattr(self, k) != getattr(other, k):
                return False
        return True

@dataclass
class ForbiddenZone(Buff):
    def __init__(self, name):
        self.name = name

class BuffManager():
    """用于记录机器人上的所有增益"""
    def __init__(self):
        self._buff_list = []
        for buff_type in BuffType:
            setattr(self, f"_{buff_type.value}_buff_list", [])

    def __repr__(self):
        """显示所有非空的增益列表"""
        active_buffs = []
        for buff_type in BuffType:
            buff_list = getattr(self, f"_{buff_type.value}_buff_list")
            active_buffs.append(f"_{buff_type.value}_buff_list={buff_list}")
        return f"BuffList({', '.join(active_buffs)})"

    def add_buff(self, buff: Buff):
        if buff not in self._buff_list:
            self._buff_list.append(buff)
        self.update_buff_list()

    def remove_buff(self, buff: Buff, strict: bool = True):
        try:
            self._buff_list.remove(buff)
        except Exception as e:
            if strict:
                raise e
        finally:
            self.update_buff_list()

    def clear_buff(self):
        self._buff_list.clear()

    def update_buff_list(self):
        for buff_type in BuffType:
            setattr(self, f"_{buff_type.value}_buff_list", [])
        for buff_type in BuffType:
            for buff in self._buff_list:
                buff_list = getattr(self, f"_{buff_type.value}_buff_list")
                buff_value = getattr(buff, buff_type.value, 0.0)
                if buff_value != 0.0:
                    buff_list.append(buff_value)

    def get_buff(self, buff_type: BuffType):
        """获取最大增益"""
        buff_list = getattr(self, f"_{buff_type.value}_buff_list")
        return max(buff_list) if buff_list else 0.0

if __name__ == "__main__":
    buff1 = Buff(name="buff1", attack=0.5, defence=0.25)
    buff2 = Buff(name="buff2", attack=1.0)
    print(buff1)
    robot_buff = BuffManager()
    # print(robot_buff)
    # print(robot_buff.get_buff(BuffType.ATTACK))
    # robot_buff.add_buff(buff1)
    # print(robot_buff)
    # print(robot_buff.get_buff(BuffType.ATTACK))
    # robot_buff.add_buff(buff2)
    # print(robot_buff)
    # print(robot_buff.get_buff(BuffType.ATTACK))
    # robot_buff.remove_buff(buff2)
    # print(robot_buff)
    # print(robot_buff.get_buff(BuffType.ATTACK))

    robot_buff.add_buff(Buff(name="buff", attack=0.25))
    robot_buff.remove_buff(Buff(name="buff", attack=0.25))
