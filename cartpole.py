import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, device="cpu", verbose=1)

start = time.time()
model.learn(total_timesteps=50_000)
print(f"Training took {time.time() - start:.0f} seconds")

after, _ = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Score after training: {after:.0f} (500 is the maximum)")
model.save("cartpole_ppo")