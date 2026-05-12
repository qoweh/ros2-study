from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pingpong_rl.envs import PingPongSim


class PingPongSimTest(unittest.TestCase):
    def test_scene_loads_and_ball_resets_above_racket(self) -> None:
        sim = PingPongSim()
        sim.reset()
        ball_position = sim.reset_ball_above_racket(height=0.22)

        self.assertEqual(sim.model.nbody, 14)
        self.assertGreater(ball_position[2], sim.racket_position[2])
        self.assertAlmostEqual(sim.ball_position[0], ball_position[0], places=6)
        self.assertAlmostEqual(sim.ball_position[1], ball_position[1], places=6)
        self.assertAlmostEqual(sim.data.ctrl[7], 255.0, places=6)
        self.assertEqual(sim.data.ncon, 0)

    def test_racket_grip_sits_between_fingers(self) -> None:
        sim = PingPongSim()
        sim.reset()

        left_finger_position = sim.data.xpos[sim.model.body("left_finger").id]
        right_finger_position = sim.data.xpos[sim.model.body("right_finger").id]
        racket_grip_position = sim.racket_grip_position
        racket_center_position = sim.racket_position

        self.assertLess(right_finger_position[0], racket_grip_position[0])
        self.assertLess(racket_grip_position[0], left_finger_position[0])
        self.assertAlmostEqual(racket_grip_position[2], left_finger_position[2], places=3)
        self.assertNotAlmostEqual(racket_center_position[1], racket_grip_position[1], places=3)
        self.assertAlmostEqual(racket_center_position[2], racket_grip_position[2], places=3)
        self.assertGreater(sim.ball_position[2], racket_center_position[2])

    def test_joint_target_update_changes_ctrl_buffer(self) -> None:
        sim = PingPongSim()
        targets = sim.home_joint_targets
        targets[0] += 0.1
        sim.step(joint_targets=targets, n_substeps=1)

        self.assertAlmostEqual(sim.data.ctrl[0], targets[0], places=6)


if __name__ == "__main__":
    unittest.main()