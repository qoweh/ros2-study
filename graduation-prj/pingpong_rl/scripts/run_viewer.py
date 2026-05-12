from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import mujoco.viewer

from pingpong_rl.controllers import JointPositionController
from pingpong_rl.envs import PingPongSim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the MuJoCo ping-pong scene viewer.")
    parser.add_argument("--ball-height", type=float, default=0.25, help="Ball spawn height above the racket in meters.")
    parser.add_argument("--reset-height", type=float, default=0.55, help="Reset the ball when it falls below this z height.")
    parser.add_argument("--demo-joint", type=int, choices=range(1, 8), help="1-based Franka joint index to move with a sine wave.")
    parser.add_argument("--demo-amplitude", type=float, default=0.2, help="Sine-wave amplitude in radians.")
    parser.add_argument("--demo-frequency", type=float, default=0.25, help="Sine-wave frequency in Hz.")
    parser.add_argument("--control-dt", type=float, default=0.02, help="Controller period in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim = PingPongSim(control_dt=args.control_dt)
    controller = JointPositionController(sim.home_joint_targets)
    sim.reset()
    sim.reset_ball_above_racket(height=args.ball_height)

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            controller.reset()
            if args.demo_joint is not None:
                joint_index = args.demo_joint - 1
                offset = args.demo_amplitude * math.sin(2.0 * math.pi * args.demo_frequency * sim.data.time)
                controller.add_joint_offset(joint_index, offset)

            sim.step(joint_targets=controller.targets)

            if sim.ball_position[2] < args.reset_height:
                sim.reset()
                sim.reset_ball_above_racket(height=args.ball_height)

            viewer.sync()
            time.sleep(sim.model.opt.timestep * sim.n_substeps)


if __name__ == "__main__":
    main()