from stable_baselines3 import PPO
from car_env import CarEnv

env = CarEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(total_timesteps=100_000)
model.save("car_ppo")