import numpy as np
from abc import ABC
from typing import Dict
from gymnasium import spaces


def linear_norm(x, x_min, x_max):
    """norm [min, max] to [-1, 1]"""
    x_clip = min(max(x, x_min), x_max)
    return (x_clip - x_min) / (x_max - x_min) * 2 - 1

def sqrt_norm(x, index, x_min, x_max):
    """norm sqrt(x) to [-1, 1]"""
    def sqrt(x):
        return x ** index
    x_clip = min(max(x, x_min), x_max)
    return (sqrt(x_clip) - sqrt(x_min)) / (sqrt(x_max) - sqrt(x_min)) * 2 - 1

def reverse_sqrt_norm(x, index, x_min, x_max):
    """reverse grads from sqrt_norm()"""
    def reverse_sqrt(x):
        return -(x_max - x) ** index
    x_clip = min(max(x, x_min), x_max)
    return (reverse_sqrt(x_clip) - reverse_sqrt(x_min)) / (reverse_sqrt(x_max) - reverse_sqrt(x_min)) * 2 - 1


class Observation(ABC):
    """Observation abstract base class."""

    # Subclasses should define their observation fields in serialization order.
    _schema: Dict = {}

    def __init__(self, **kwargs):
        """Initialize observation fields."""
        self._validate_schema()
        self._init_observations(**kwargs)

    def __setattr__(self, name, value):
        """Validate observation values when fields are assigned."""
        if name.startswith("_") or name in ["space", "_schema"]:
            super().__setattr__(name, value)
        elif name in self._schema:
            value = self._coerce_value(name, value)
            if self._contains_value(name, value):
                super().__setattr__(name, value)
            else:
                raise ValueError(f"Invalid value for {name}: {value}, which should belong to {self._schema[name]}")
        else:
            super().__setattr__(name, value)

    def __repr__(self):
        observation_str = ", ".join([f"{k}={getattr(self, k)}" for k in self._schema])
        return f"{self.__class__.__name__}({observation_str})"

    @classmethod
    def _validate_schema(cls):
        """Validate that subclass defines a usable schema."""
        if not hasattr(cls, "_schema") or not cls._schema:
            raise ValueError(f"{cls.__name__} must define '_schema' class attribute")

    @classmethod
    def get_space(cls) -> spaces.Dict:
        """Get observation space."""
        cls._validate_schema()
        return spaces.Dict(cls._schema)

    @classmethod
    def from_array(cls, array: np.ndarray):
        """Create an observation from a flat NumPy array."""
        cls._validate_schema()
        array = np.asarray(array)
        expected_shape = (cls.get_array_size(),)
        assert array.shape == expected_shape

        kwargs = {}
        cursor = 0
        for name, space in cls._schema.items():
            size = cls._space_size(space)
            value = array[cursor:cursor + size]
            cursor += size

            if isinstance(space, spaces.Box):
                kwargs[name] = value.reshape(space.shape).astype(space.dtype)
            elif isinstance(space, spaces.Discrete):
                kwargs[name] = int(np.ceil(value[0]))
            else:
                kwargs[name] = value[0] if size == 1 else value

        return cls(**kwargs)

    @classmethod
    def get_array_size(cls) -> int:
        """Get flattened observation array size."""
        cls._validate_schema()
        return sum(cls._space_size(space) for space in cls._schema.values())

    @staticmethod
    def _space_size(space) -> int:
        if isinstance(space, spaces.Box):
            return int(np.prod(space.shape))
        if isinstance(space, spaces.Discrete):
            return 1
        raise TypeError(f"Unsupported observation space: {space}")

    def _init_observations(self, **kwargs):
        """Initialize observation values."""
        for name, space in self._schema.items():
            if name in kwargs:
                value = self._coerce_value(name, kwargs[name])
                if not self._contains_value(name, value):
                    raise ValueError(f"Invalid value for {name}: {value}")
                setattr(self, name, value)
            else:
                if isinstance(space, spaces.Box):
                    setattr(self, name, self._coerce_value(name, (space.low + space.high) / 2.0))
                elif isinstance(space, spaces.Discrete):
                    setattr(self, name, 0)
                else:
                    setattr(self, name, None)

    def _contains_value(self, name, value) -> bool:
        """Validate one observation field against its own space."""
        return self._schema[name].contains(value)

    def _coerce_value(self, name, value):
        space = self._schema[name]
        if isinstance(space, spaces.Box):
            return np.asarray(value, dtype=space.dtype).reshape(space.shape)
        if isinstance(space, spaces.Discrete):
            return int(value)
        return value

    def to_array(self) -> np.ndarray:
        """Convert observation fields to a flat NumPy array."""
        array = []
        for name, space in self._schema.items():
            value = getattr(self, name)
            if isinstance(space, spaces.Box):
                array.extend(np.asarray(value, dtype=space.dtype).reshape(-1))
            elif isinstance(space, spaces.Discrete):
                array.append(value)
            else:
                raise TypeError(f"Unsupported observation space: {space}")
        return np.array(array)


if __name__ == "__main__":
    class ObsEnv(Observation):
        _schema = {
            "remaining_time_norm": spaces.Box(
                low=0.0, 
                high=1.0,
                dtype=np.float32
            ),
        }
    env_obs = ObsEnv.from_array([0.8])
    print(env_obs)
    print(env_obs.__dict__)
    print(env_obs.to_array())
    env_obs = ObsEnv(
        remaining_time_norm=1.0
    )

    class ObsRobot(Observation):
        _schema = {
            "position_norm": spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32), 
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            ),
            "hp_norm": spaces.Box(
                low=0.0, 
                high=1.0,
                dtype=np.float32
            ),
            "team": spaces.Discrete(2),
        }
    robot_obs1 = ObsRobot.from_array([0.8, -0.2, 1.0, 1])
    print(robot_obs1)
    print(robot_obs1.__dict__)
    print(robot_obs1.to_array())

    print(linear_norm(150, 0, 200))
    print(sqrt_norm(160, 0.5, 0, 600))
