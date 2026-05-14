import argparse
import time

import mujoco.viewer
import numpy as np

from car_env import CarEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Run a scripted demo in the upgraded car environment.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--goal-x", type=float, default=1.8)
    parser.add_argument("--goal-y", type=float, default=-1.2)
    parser.add_argument("--goal-yaw", type=float, default=1.57)
    return parser.parse_args()


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def scripted_controller(env, info):
    goal = info["goal"]
    x_pos, y_pos, yaw = info["car_pose"]
    dx = goal[0] - x_pos
    dy = goal[1] - y_pos
    distance = info["distance_to_goal"]
    heading_to_goal = np.arctan2(dy, dx)
    heading_error = wrap_angle(heading_to_goal - yaw)

    if distance > 0.55:
        heading_target = heading_error
        desired_speed = 1.4 if distance > 1.4 else 0.7
    else:
        heading_target = info["yaw_error"]
        desired_speed = 0.0

    steering = np.clip(heading_target / env.max_steer, -1.0, 1.0)
    throttle = np.clip((desired_speed - info["speed"]) / env.acceleration, -1.0, 1.0)
    if distance < 0.35:
        throttle = np.clip(-2.0 * info["speed"], -1.0, 1.0)
    return np.array([throttle, steering], dtype=np.float32)


def main():
    args = parse_args()
    goal = np.array([args.goal_x, args.goal_y, args.goal_yaw], dtype=np.float64)
    env = CarEnv(goal_mode="fixed", fixed_goal=goal)
    viewer = mujoco.viewer.launch_passive(env.model, env.data)

    obs, info = env.reset(seed=args.seed, options={"goal": goal})
    while viewer.is_running():
        action = scripted_controller(env, info)
        obs, reward, terminated, truncated, info = env.step(action)
        viewer.sync()
        time.sleep(env.dt)

        if terminated or truncated:
            print(
                f"success={info['success']} distance={info['distance_to_goal']:.3f} "
                f"yaw_error={info['yaw_error']:.3f} reward={reward:.3f}"
            )
            break

    viewer.close()
    env.close()


if __name__ == "__main__":
    main()