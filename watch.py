"""Watch the pendulum. Trained by default; --untrained for the flopping cold open.

python watch.py                        # trained model (pendulum_ppo.zip) in a window
python watch.py --untrained            # brand-new, untrained policy: it flops over
python watch.py --record               # no window: saves one MP4 per attempt to videos/
python watch.py --model checkpoints/pend_20000_steps   # any saved model
"""

import argparse
import gymnasium as gym
from stable_baselines3 import PPO

p = argparse.ArgumentParser()
p.add_argument("--untrained", action="store_true", help="use a fresh, untrained policy")
p.add_argument("--model", default="pendulum_ppo", help="saved model to load")
p.add_argument("--episodes", type=int, default=10, help="how many attempts to show")
p.add_argument("--record", action="store_true", help="save MP4s instead of opening a window")
p.add_argument("--size", type=int, default=1080, help="frame width and height in pixels")
args = p.parse_args()

label = "untrained" if args.untrained else args.model.replace("/", "_").replace(".zip", "")
mode = "rgb_array" if args.record else "human"
env = gym.make("InvertedPendulum-v5", render_mode=mode, width=args.size, height=args.size)
if args.record:
    env = gym.wrappers.RecordVideo(env, video_folder=f"videos/{label}", name_prefix=label,
                                    episode_trigger=lambda ep: True)

if args.untrained:
    model = PPO("MlpPolicy", env, device="cpu")
else:
    model = PPO.load(args.model, device="cpu")

for ep in range(args.episodes):
    obs, _ = env.reset()
    score, done = 0, False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        score += reward
        done = terminated or truncated
    print(f"Attempt {ep + 1}: score {score:.0f}")
env.close()