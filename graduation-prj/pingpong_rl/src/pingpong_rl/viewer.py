from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time

import mujoco.viewer

from pingpong_rl.controllers import JointPositionController
from pingpong_rl.envs import PingPongSim
from pingpong_rl.utils.paths import SCENE_XML_PATH


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the MuJoCo ping-pong scene viewer.")
    parser.add_argument(
        "--mode",
        choices=("interactive", "passive"),
        default="interactive",
        help="Use MuJoCo's default interactive viewer or the previous passive scripted loop.",
    )
    parser.add_argument("--ball-height", type=float, default=0.22, help="Ball spawn height above the racket in meters.")
    parser.add_argument("--reset-height", type=float, default=0.25, help="Passive mode only: reset the ball when it falls below this z height.")
    parser.add_argument("--demo-joint", type=int, choices=range(1, 8), help="Passive mode only: 1-based Franka joint index to move with a sine wave.")
    parser.add_argument("--demo-amplitude", type=float, default=0.2, help="Passive mode only: sine-wave amplitude in radians.")
    parser.add_argument("--demo-frequency", type=float, default=0.25, help="Passive mode only: sine-wave frequency in Hz.")
    parser.add_argument("--control-dt", type=float, default=0.02, help="Controller period in seconds.")
    return parser.parse_args(argv)


def _make_sim(control_dt: float, ball_height: float) -> PingPongSim:
    sim = PingPongSim(control_dt=control_dt)
    sim.reset(ball_height=ball_height)
    return sim


def _run_interactive(args: argparse.Namespace) -> None:
    _ = args
    command = [sys.executable, "-m", "mujoco.viewer", f"--mjcf={SCENE_XML_PATH}"]
    raise SystemExit(subprocess.run(command, check=False).returncode)


def _run_passive(args: argparse.Namespace) -> None:
    sim = _make_sim(args.control_dt, args.ball_height)
    controller = JointPositionController(sim.home_joint_targets)

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            controller.reset()
            if args.demo_joint is not None:
                joint_index = args.demo_joint - 1
                offset = args.demo_amplitude * math.sin(2.0 * math.pi * args.demo_frequency * sim.data.time)
                controller.add_joint_offset(joint_index, offset)

            sim.step(joint_targets=controller.targets)

            if sim.ball_position[2] < args.reset_height:
                sim.reset(ball_height=args.ball_height)

            viewer.sync()
            time.sleep(sim.model.opt.timestep * sim.n_substeps)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "interactive":
        _run_interactive(args)
        return
    _run_passive(args)


if __name__ == "__main__":
    main()