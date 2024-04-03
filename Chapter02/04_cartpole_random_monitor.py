gymnasium = False

try:
    import gym
except ImportError:
    import gymnasium as gym
    gymnasium = True

if __name__ == "__main__":

    if gymnasium:
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        # env = gym.experimental.wrappers.RecordVideoV0(env, "recording")
        env = gym.wrappers.RecordVideo(env, "gymnasium_recording")

    else:
        env = gym.make("CartPole-v1")
        env = gym.wrappers.Monitor(env, "gym_recording")

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        if gymnasium:
            obs, reward, done, truncated, info = env.step(action)
        else:
            obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break

    print(f"Episode done in {total_steps} steps, total reward {total_reward: .4f}")
    env.close()
    env.env.close()
