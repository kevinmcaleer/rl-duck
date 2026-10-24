"""B-roll pre-run: train the pendulum again, saving a checkpoint every 10,000 steps,
a reward log (logs/monitor.csv) and TensorBoard data (runs/), so you can film the
'flop, then last longer, then hold' progression and draw the reward curve."""
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

os.makedirs("logs", exist_ok=True)
env = Monitor(gym.make("InvertedPendulum-v5"), "logs/")
model = PPO("MlpPolicy", env, device="cpu", verbose=1, seed=1, tensorboard_log="runs")
model.learn(total_timesteps=100_000,
            callback=CheckpointCallback(save_freq=10_000, save_path="checkpoints", name_prefix="pend"))
model.save("ladder_final")