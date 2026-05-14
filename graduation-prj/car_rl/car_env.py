from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


def _wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _yaw_to_quat(yaw):
    half_yaw = 0.5 * yaw
    return np.array(
        [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
        dtype=np.float64,
    )


class CarEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 20,
    }

    def __init__(
        self,
        goal_mode="random",
        fixed_goal=None,
        arena_limit=3.0,
        max_episode_steps=240,
    ):
        super().__init__()

        xml_path = Path(__file__).resolve().with_name("car.xml")
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        if goal_mode not in {"random", "fixed"}:
            raise ValueError("goal_mode must be 'random' or 'fixed'.")

        self.goal_mode = goal_mode
        self.fixed_goal = None
        if fixed_goal is not None:
            self.fixed_goal = self._sanitize_goal(fixed_goal)
        elif self.goal_mode == "fixed":
            self.fixed_goal = np.array([2.0, 0.0, 0.0], dtype=np.float64)

        self.dt = float(self.model.opt.timestep)
        self.arena_limit = float(arena_limit)
        self.goal_sampling_limit = self.arena_limit - 0.7
        self.max_episode_steps = int(max_episode_steps)

        self.max_speed = 2.2
        self.max_reverse_speed = 1.0
        self.acceleration = 3.2
        self.drag = 0.9
        self.max_steer = 0.65
        self.max_steer_rate = 2.4
        self.wheelbase = 0.42
        self.wheel_radius = 0.04

        self.goal_radius = 0.40
        self.goal_yaw_tolerance = 0.45
        self.max_distance = np.sqrt(2.0) * self.arena_limit

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32,
        )

        self._joint_names = (
            "car_x",
            "car_y",
            "car_yaw",
            "front_left_steer",
            "front_right_steer",
            "front_left_spin",
            "front_right_spin",
            "rear_left_spin",
            "rear_right_spin",
        )
        self._qpos_adr = {}
        self._qvel_adr = {}
        for joint_name in self._joint_names:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                raise ValueError(f"Joint '{joint_name}' was not found in car.xml.")
            self._qpos_adr[joint_name] = int(self.model.jnt_qposadr[joint_id])
            self._qvel_adr[joint_name] = int(self.model.jnt_dofadr[joint_id])

        goal_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "goal",
        )
        self._goal_mocap_id = int(self.model.body_mocapid[goal_body_id])
        if self._goal_mocap_id < 0:
            raise ValueError("Goal body must be configured as a mocap body.")

        self.goal = np.zeros(3, dtype=np.float64)
        self.speed = 0.0
        self.steering_angle = 0.0
        self.wheel_spin = 0.0
        self.step_count = 0
        self.prev_distance = 0.0
        self.prev_yaw_error = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)

    def _sanitize_goal(self, goal):
        goal = np.asarray(goal, dtype=np.float64).reshape(3)
        goal[0] = np.clip(goal[0], -self.goal_sampling_limit, self.goal_sampling_limit)
        goal[1] = np.clip(goal[1], -self.goal_sampling_limit, self.goal_sampling_limit)
        goal[2] = _wrap_angle(goal[2])
        return goal

    def _set_joint_state(self, joint_name, qpos_value, qvel_value=0.0):
        self.data.qpos[self._qpos_adr[joint_name]] = qpos_value
        self.data.qvel[self._qvel_adr[joint_name]] = qvel_value

    def _get_pose(self):
        return np.array(
            [
                self.data.qpos[self._qpos_adr["car_x"]],
                self.data.qpos[self._qpos_adr["car_y"]],
                self.data.qpos[self._qpos_adr["car_yaw"]],
            ],
            dtype=np.float64,
        )

    def _goal_override(self, options):
        if not options:
            return None

        if "goal" in options and options["goal"] is not None:
            return self._sanitize_goal(options["goal"])

        goal_keys = {"goal_x", "goal_y", "goal_yaw"}
        if goal_keys.intersection(options):
            if self.fixed_goal is not None:
                base_goal = self.fixed_goal.copy()
            else:
                base_goal = np.array([2.0, 0.0, 0.0], dtype=np.float64)
            base_goal[0] = options.get("goal_x", base_goal[0])
            base_goal[1] = options.get("goal_y", base_goal[1])
            base_goal[2] = options.get("goal_yaw", base_goal[2])
            return self._sanitize_goal(base_goal)

        return None

    def _sample_goal(self):
        if self.goal_mode == "fixed" and self.fixed_goal is not None:
            return self.fixed_goal.copy()

        while True:
            goal_x = self.np_random.uniform(-self.goal_sampling_limit, self.goal_sampling_limit)
            goal_y = self.np_random.uniform(-self.goal_sampling_limit, self.goal_sampling_limit)
            goal_yaw = np.arctan2(goal_y, goal_x) + self.np_random.uniform(-0.35, 0.35)
            candidate = np.array(
                [
                    goal_x,
                    goal_y,
                    goal_yaw,
                ],
                dtype=np.float64,
            )
            if np.linalg.norm(candidate[:2]) > 1.2:
                return candidate

    def _set_goal(self, goal):
        self.goal[:] = self._sanitize_goal(goal)
        self.data.mocap_pos[self._goal_mocap_id] = np.array(
            [self.goal[0], self.goal[1], 0.02],
            dtype=np.float64,
        )
        self.data.mocap_quat[self._goal_mocap_id] = _yaw_to_quat(self.goal[2])

    def _goal_local(self):
        x_pos, y_pos, yaw = self._get_pose()
        dx = self.goal[0] - x_pos
        dy = self.goal[1] - y_pos
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        return np.array(
            [
                cos_yaw * dx + sin_yaw * dy,
                -sin_yaw * dx + cos_yaw * dy,
            ],
            dtype=np.float64,
        )

    def _distance_to_goal(self):
        return float(np.linalg.norm(self._goal_local()))

    def _yaw_error(self):
        return float(_wrap_angle(self.goal[2] - self._get_pose()[2]))

    def _sync_pose(self, pose, yaw_rate, steering_rate):
        x_pos, y_pos, yaw = pose
        x_vel = self.speed * np.cos(yaw)
        y_vel = self.speed * np.sin(yaw)
        wheel_velocity = self.speed / self.wheel_radius
        self.wheel_spin = _wrap_angle(self.wheel_spin + wheel_velocity * self.dt)

        self._set_joint_state("car_x", x_pos, x_vel)
        self._set_joint_state("car_y", y_pos, y_vel)
        self._set_joint_state("car_yaw", yaw, yaw_rate)
        self._set_joint_state("front_left_steer", self.steering_angle, steering_rate)
        self._set_joint_state("front_right_steer", self.steering_angle, steering_rate)

        for joint_name in (
            "front_left_spin",
            "front_right_spin",
            "rear_left_spin",
            "rear_right_spin",
        ):
            self._set_joint_state(joint_name, self.wheel_spin, wheel_velocity)

        mujoco.mj_forward(self.model, self.data)

    def get_obs(self):
        goal_local = self._goal_local()
        distance = np.linalg.norm(goal_local)
        yaw_error = self._yaw_error()
        obs = np.array(
            [
                goal_local[0] / self.max_distance,
                goal_local[1] / self.max_distance,
                distance / self.max_distance,
                np.cos(yaw_error),
                np.sin(yaw_error),
                self.speed / self.max_speed,
                self.steering_angle / self.max_steer,
                self.last_action[0],
                self.last_action[1],
                1.0 - (self.step_count / self.max_episode_steps),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def _info(self, success=False, out_of_bounds=False):
        distance_to_goal = self._distance_to_goal()
        yaw_error = self._yaw_error()
        return {
            "goal": self.goal.copy(),
            "car_pose": self._get_pose(),
            "distance_to_goal": distance_to_goal,
            "yaw_error": yaw_error,
            "speed": float(self.speed),
            "steering_angle": float(self.steering_angle),
            "episode_step": self.step_count,
            "success": bool(success),
            "position_success": bool(distance_to_goal < self.goal_radius),
            "pose_success": bool(
                distance_to_goal < self.goal_radius
                and abs(yaw_error) < self.goal_yaw_tolerance
            ),
            "out_of_bounds": bool(out_of_bounds),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.speed = 0.0
        self.steering_angle = 0.0
        self.wheel_spin = 0.0
        self.step_count = 0
        self.last_action = np.zeros(2, dtype=np.float32)

        start_pose = np.array(
            [
                0.0,
                0.0,
                self.np_random.uniform(-0.2, 0.2),
            ],
            dtype=np.float64,
        )
        self._sync_pose(start_pose, yaw_rate=0.0, steering_rate=0.0)

        goal = self._goal_override(options)
        if goal is None:
            goal = self._sample_goal()
        self._set_goal(goal)
        mujoco.mj_forward(self.model, self.data)

        self.prev_distance = self._distance_to_goal()
        self.prev_yaw_error = abs(self._yaw_error())
        return self.get_obs(), self._info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        prev_distance = self.prev_distance
        prev_yaw_error = self.prev_yaw_error

        throttle = float(action[0])
        steer_command = float(action[1])

        target_steering = steer_command * self.max_steer
        max_steer_delta = self.max_steer_rate * self.dt
        steering_delta = np.clip(
            target_steering - self.steering_angle,
            -max_steer_delta,
            max_steer_delta,
        )
        self.steering_angle = float(
            np.clip(
                self.steering_angle + steering_delta,
                -self.max_steer,
                self.max_steer,
            )
        )

        self.speed += throttle * self.acceleration * self.dt
        self.speed -= self.drag * self.speed * self.dt
        self.speed = float(np.clip(self.speed, -self.max_reverse_speed, self.max_speed))

        x_pos, y_pos, yaw = self._get_pose()
        x_pos += self.speed * np.cos(yaw) * self.dt
        y_pos += self.speed * np.sin(yaw) * self.dt
        yaw_rate = self.speed / self.wheelbase * np.tan(self.steering_angle)
        yaw = _wrap_angle(yaw + yaw_rate * self.dt)

        self.step_count += 1
        self.last_action = action.copy()
        self._sync_pose(
            pose=np.array([x_pos, y_pos, yaw], dtype=np.float64),
            yaw_rate=yaw_rate,
            steering_rate=steering_delta / self.dt,
        )

        distance = self._distance_to_goal()
        yaw_error = abs(self._yaw_error())
        success = (
            distance < self.goal_radius
            and yaw_error < self.goal_yaw_tolerance
        )
        out_of_bounds = abs(x_pos) > self.arena_limit or abs(y_pos) > self.arena_limit
        terminated = success or out_of_bounds
        truncated = self.step_count >= self.max_episode_steps and not terminated

        progress_reward = prev_distance - distance
        yaw_progress = prev_yaw_error - yaw_error
        reward = (
            2.5 * progress_reward
            + 0.4 * yaw_progress
            - 0.02 * distance
            - 0.01 * float(np.square(action).sum())
        )
        if distance < 0.9:
            reward += 0.6 * np.cos(self._yaw_error()) - 0.03 * abs(self.speed)
        if success:
            reward += 25.0
        if out_of_bounds:
            reward -= 10.0

        self.prev_distance = distance
        self.prev_yaw_error = yaw_error
        info = self._info(success=success, out_of_bounds=out_of_bounds)
        return self.get_obs(), reward, terminated, truncated, info

    def close(self):
        pass