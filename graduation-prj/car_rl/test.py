import mujoco.viewer
import time
from stable_baselines3 import PPO
from car_env import CarEnv

env = CarEnv()
model = PPO.load("car_ppo")
viewer = mujoco.viewer.launch_passive(env.model,env.data)
obs, _ = env.reset()

while viewer.is_running():

    action, _ = model.predict(
        obs,
        deterministic=True
    )
    obs, reward, terminated, truncated, info = env.step(action)
    viewer.sync()
    time.sleep(0.01)

    if terminated:
        print("Target reached!")
        break

viewer.close()