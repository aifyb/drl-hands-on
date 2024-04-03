import random
from typing import TypeVar

gymnasium = False
try:
    import gym
except ImportError:
    import gymnasium as gym
    gymnasium = True

Action = TypeVar('Action')


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action: Action) -> Action:
        if random.random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        return action


if __name__ == "__main__":
    env = RandomActionWrapper(gym.make("CartPole-v1"))

    obs = env.reset()
    total_reward = 0.0

    while True:
        if gymnasium:
            obs, reward, done, truncated, info = env.step(0)
        else:
            obs, reward, done, _ = env.step(0)
        total_reward += reward
        if done:
            break

    print(f"Reward got: {total_reward: .4f}")
