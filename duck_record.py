"""Headless recording of Open Duck Mini in MuJoCo (no viewer window, no mjpython needed).
Run from inside the Open_Duck_Playground folder.
EP1 (no walking):  ../duck/bin/python ../duck_record.py --onnx ../Open_Duck_Mini/BEST_WALK_ONNX_2.onnx --pose --out ../footage/duck_turntable.mp4
Without --pose it runs the pretrained walking policy: that is Ep2 material, not Ep1.
"""
import argparse, os, sys
import numpy as np
import mujoco
import imageio.v2 as imageio

sys.path.insert(0, os.getcwd())          # the repo's own 'playground' package
from playground.open_duck_mini_v2.mujoco_infer import MjInfer

p = argparse.ArgumentParser()
p.add_argument("--onnx", required=True)
p.add_argument("--seconds", type=float, default=15)
p.add_argument("--forward", type=float, default=0.15, help="forward speed command (max 0.15)")
p.add_argument("--out", default="duck_walk.mp4")
p.add_argument("--width", type=int, default=1920)
p.add_argument("--height", type=int, default=1088)
p.add_argument("--pose", action="store_true",
                help="no policy: hold the default standing pose and slowly orbit the camera (no walking)")
a = p.parse_args()

m = MjInfer("playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
            "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl", a.onnx, False)
m.model.vis.global_.offwidth = max(m.model.vis.global_.offwidth, a.width)
m.model.vis.global_.offheight = max(m.model.vis.global_.offheight, a.height)
renderer = mujoco.Renderer(m.model, height=a.height, width=a.width)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = 1
cam.distance, cam.azimuth, cam.elevation = 1.0, 135, -15
m.commands[0] = a.forward

fps = 50
writer = imageio.get_writer(a.out, fps=fps, quality=8)
steps_per_frame = int(round(1 / (fps * m.sim_dt)))
counter = 0
for _ in range(int(a.seconds * fps)):
    for _ in range(steps_per_frame):
        mujoco.mj_step(m.model, m.data)
        counter += 1
        if a.pose:
            m.data.ctrl = m.default_actuator
        elif counter % m.decimation == 0:
            m.imitation_i = (m.imitation_i + m.phase_frequency_factor) % m.PRM.nb_steps_in_period
            ph = m.imitation_i / m.PRM.nb_steps_in_period * 2 * np.pi
            m.imitation_phase = np.array([np.cos(ph), np.sin(ph)])
            obs = m.get_obs(m.data, m.commands)
            action = m.policy.infer(obs)
            m.last_last_last_action = m.last_last_action.copy()
            m.last_last_action = m.last_action.copy()
            m.last_action = action.copy()
            m.motor_targets = m.default_actuator + action * m.action_scale
            lim = m.max_motor_velocity * (m.sim_dt * m.decimation)
            m.motor_targets = np.clip(m.motor_targets, m.prev_motor_targets - lim, m.prev_motor_targets + lim)
            m.prev_motor_targets = m.motor_targets.copy()
            m.data.ctrl = m.motor_targets.copy()
    if a.pose:
        cam.azimuth += 0.6          # slow turntable, about 30 degrees per second
    renderer.update_scene(m.data, camera=cam)
    writer.append_data(renderer.render())
writer.close()
print(f"Saved {a.out}; base height at end: {m.data.qpos[2]:.3f} m, x moved: {m.data.qpos[0]:.2f} m")