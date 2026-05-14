import argparse
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from car_env import CarEnv


def parse_args():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy on the upgraded car task.")
    parser.add_argument("--model-path", type=Path, default=base_dir / "car_ppo_nav.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--goal-x", type=float)
    parser.add_argument("--goal-y", type=float)
    parser.add_argument("--goal-yaw", type=float)
    return parser.parse_args()


def resolve_goal(args):
    goal_values = [args.goal_x, args.goal_y, args.goal_yaw]
    if all(value is None for value in goal_values):
        return None
    if any(value is None for value in goal_values):
        raise ValueError("Provide --goal-x, --goal-y and --goal-yaw together.")
    return np.array(goal_values, dtype=np.float64)


def rollout_episode(env, model, seed, viewer=None, goal=None):
    options = {"goal": goal} if goal is not None else None
    obs, info = env.reset(seed=seed, options=options)
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if viewer is not None:
            if not viewer.is_running():
                return {
                    "interrupted": True,
                    "reward": total_reward,
                    "info": info,
                    "terminated": terminated,
                    "truncated": truncated,
                }
            viewer.sync()
            time.sleep(env.dt)

    return {
        "interrupted": False,
        "reward": total_reward,
        "info": info,
        "terminated": terminated,
        "truncated": truncated,
    }


def main():
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    goal = resolve_goal(args)
    env = CarEnv(goal_mode="fixed" if goal is not None else "random", fixed_goal=goal)
    model = PPO.load(str(args.model_path))

    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    episode_results = []
    for episode_index in range(args.episodes):
        result = rollout_episode(
            env,
            model,
            seed=args.seed + episode_index,
            viewer=viewer,
            goal=goal,
        )
        info = result["info"]
        episode_results.append(result)
        print(
            "episode=",
            episode_index + 1,
            "success=",
            info["success"],
            "distance=",
            round(info["distance_to_goal"], 3),
            "yaw_error=",
            round(info["yaw_error"], 3),
            "reward=",
            round(float(result["reward"]), 3),
            "steps=",
            info["episode_step"],
        )
        if result["interrupted"]:
            break

    if viewer is not None:
        viewer.close()

    completed = [result for result in episode_results if not result["interrupted"]]
    if completed:
        success_count = sum(result["info"]["success"] for result in completed)
        mean_reward = np.mean([result["reward"] for result in completed])
        mean_distance = np.mean([result["info"]["distance_to_goal"] for result in completed])
        print(
            f"summary: success_rate={success_count / len(completed):.2f}, "
            f"mean_reward={mean_reward:.3f}, mean_final_distance={mean_distance:.3f}"
        )

    env.close()


if __name__ == "__main__":
    main()