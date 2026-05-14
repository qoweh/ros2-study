import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from car_env import CarEnv


def build_env(goal_mode="random", fixed_goal=None):
    return Monitor(CarEnv(goal_mode=goal_mode, fixed_goal=fixed_goal))


def parse_args():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train PPO on the upgraded toy car navigation task.")
    parser.add_argument("--timesteps", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--model-path", type=Path, default=base_dir / "car_ppo_nav")
    parser.add_argument("--log-dir", type=Path, default=base_dir / "runs")
    return parser.parse_args()


def main():
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / "best_model").mkdir(parents=True, exist_ok=True)
    (args.log_dir / "eval").mkdir(parents=True, exist_ok=True)

    env = build_env(goal_mode="random")
    eval_env = build_env(goal_mode="random")
    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.log_dir / "best_model"),
        log_path=str(args.log_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        tensorboard_log=str(args.log_dir / "tb"),
    )

    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(str(args.model_path))

    env.close()
    eval_env.close()
    print(f"Saved final model to {args.model_path}.zip")


if __name__ == "__main__":
    main()