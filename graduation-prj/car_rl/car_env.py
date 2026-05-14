import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CarEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path("car.xml")
        self.data = mujoco.MjData(self.model)

        # action: motor 1개
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # observation: x 위치 + x 속도
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32
        )

        self.target_x = 2.0

    def get_obs(self):
        x_pos = self.data.qpos[0]
        x_vel = self.data.qvel[0]
        return np.array(
            [x_pos, x_vel],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(
            self.model,
            self.data
        )
        obs = self.get_obs()
        return obs, {}

    def step(self, action):
        self.data.ctrl[0] = action[0]
        mujoco.mj_step(
            self.model,
            self.data
        )
        
        obs = self.get_obs()
        distance = abs(self.target_x - obs[0])
        reward = np.exp(-distance)

        terminated = distance < 0.1
        truncated = False
        info = {}
        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )