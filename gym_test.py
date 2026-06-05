import gymnasium as gym
from rules.base.game import RoboMasterGym

# env = gym.make("LunarLander-v3", render_mode="human")
# env = gym.make("Pendulum-v1", render_mode="human", g=9.81)  # default g=10.0
env = RoboMasterGym(render_mode="human")

observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()