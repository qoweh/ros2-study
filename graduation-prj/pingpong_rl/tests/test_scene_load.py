from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pingpong_rl.envs import PingPongEEDeltaEnv, PingPongSim
from pingpong_rl.controllers import RacketCartesianController
from pingpong_rl.viewer import _ee_demo_target_position, _passive_viewer_is_running, parse_args


class PingPongSimTest(unittest.TestCase):
    def test_ball_bounces_off_floor_after_tuning(self) -> None:
        sim = PingPongSim()
        sim.reset(ball_position=(0.2, -0.25, 1.0), ball_velocity=(0.0, 0.0, 0.0))

        impact_seen = False
        post_impact_peak = None
        previous_vz = float(sim.ball_velocity[2])
        for _ in range(5000):
            sim.step(n_substeps=1)
            ball_height = float(sim.ball_position[2])
            vertical_velocity = float(sim.ball_velocity[2])
            if not impact_seen and sim.has_contact("ball_geom", "floor"):
                impact_seen = True
            elif impact_seen:
                if vertical_velocity > 0.0:
                    post_impact_peak = ball_height if post_impact_peak is None else max(post_impact_peak, ball_height)
                if post_impact_peak is not None and previous_vz > 0.0 and vertical_velocity <= 0.0:
                    break
            previous_vz = vertical_velocity

        self.assertTrue(impact_seen)
        self.assertIsNotNone(post_impact_peak)
        self.assertGreater(post_impact_peak, 0.2)

    def test_scene_loads_and_ball_resets_above_racket(self) -> None:
        sim = PingPongSim()
        sim.reset()
        ball_position = sim.reset_ball_above_racket(height=0.22)

        self.assertEqual(sim.model.nbody, 14)
        self.assertGreater(ball_position[2], sim.racket_position[2])
        self.assertAlmostEqual(sim.ball_position[0], ball_position[0], places=6)
        self.assertAlmostEqual(sim.ball_position[1], ball_position[1], places=6)
        self.assertAlmostEqual(sim.data.ctrl[7], sim.home_gripper_target, places=6)
        self.assertAlmostEqual(float(sim.data.joint("finger_joint1").qpos[0]), 0.012, places=6)
        self.assertAlmostEqual(float(sim.data.joint("finger_joint2").qpos[0]), 0.012, places=6)
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
        self.assertNotAlmostEqual(racket_center_position[1], racket_grip_position[1], places=3)
        self.assertGreater(sim.ball_position[2], racket_center_position[2])
        self.assertEqual(sim.data.ncon, 0)

    def test_joint_target_update_changes_ctrl_buffer(self) -> None:
        sim = PingPongSim()
        targets = sim.home_joint_targets
        targets[0] += 0.1
        sim.step(joint_targets=targets, n_substeps=1)

        self.assertAlmostEqual(sim.data.ctrl[0], targets[0], places=6)

    def test_ball_hits_racket_before_floor(self) -> None:
        sim = PingPongSim()
        sim.reset()

        first_target_contact: tuple[str, str] | None = None
        for _ in range(800):
            sim.step(n_substeps=1)
            if sim.has_contact("ball_geom", "racket_head"):
                first_target_contact = ("ball_geom", "racket_head")
                break
            if sim.has_contact("ball_geom", "floor"):
                first_target_contact = ("ball_geom", "floor")
                break

        self.assertEqual(first_target_contact, ("ball_geom", "racket_head"))

    def test_failure_reason_reports_out_of_bounds(self) -> None:
        sim = PingPongSim()
        sim.reset()
        sim.spawn_ball((1.8, 0.0, 0.5))

        self.assertEqual(sim.failure_reason(), "ball_out_of_bounds")

    def test_reset_if_failed_respawns_after_floor_contact(self) -> None:
        sim = PingPongSim()
        sim.reset()

        failure_reason = None
        for _ in range(1200):
            sim.step(n_substeps=1)
            failure_reason = sim.failure_reason()
            if failure_reason is not None:
                break

        self.assertEqual(failure_reason, "floor_contact")

        reset_reason = sim.reset_if_failed()
        self.assertEqual(reset_reason, "floor_contact")
        self.assertEqual(sim.data.ncon, 0)
        self.assertGreater(sim.ball_position[2], sim.racket_position[2])
        self.assertIsNone(sim.failure_reason())

    def test_racket_cartesian_controller_reduces_position_error(self) -> None:
        sim = PingPongSim(control_dt=0.02)
        sim.reset()
        controller = RacketCartesianController(sim)

        target_position = sim.racket_position + np.array([0.02, 0.0, 0.01])
        controller.set_target_position(target_position)
        initial_error = np.linalg.norm(target_position - sim.racket_position)

        for _ in range(15):
            joint_targets = controller.compute_joint_targets()
            sim.step(joint_targets=joint_targets, n_substeps=sim.n_substeps)

        final_error = np.linalg.norm(target_position - sim.racket_position)
        self.assertLess(final_error, initial_error)

    def test_passive_viewer_pause_helper_uses_run_flag(self) -> None:
        class FakeSimState:
            def __init__(self, run: int) -> None:
                self.run = run

        class FakeViewer:
            def __init__(self, run: int) -> None:
                self._sim_state = FakeSimState(run)

            def _get_sim(self) -> FakeSimState:
                return self._sim_state

        self.assertTrue(_passive_viewer_is_running(FakeViewer(1)))
        self.assertFalse(_passive_viewer_is_running(FakeViewer(0)))

    def test_viewer_parse_args_accepts_ee_demo(self) -> None:
        args = parse_args(["--demo-controller", "ee", "--ee-axis", "x", "--demo-amplitude", "0.03"])

        self.assertEqual(args.mode, "passive")
        self.assertEqual(args.demo_controller, "ee")
        self.assertEqual(args.ee_axis, "x")
        self.assertAlmostEqual(args.demo_amplitude, 0.03, places=6)

    def test_ee_demo_target_position_only_moves_selected_axis(self) -> None:
        anchor = np.array([0.5, 0.1, 0.6])
        target = _ee_demo_target_position(anchor, "z", amplitude=0.04, frequency=0.5, time_seconds=0.5)

        self.assertAlmostEqual(target[0], anchor[0], places=6)
        self.assertAlmostEqual(target[1], anchor[1], places=6)
        self.assertGreater(target[2], anchor[2])

    def test_ee_delta_env_step_clips_action_and_returns_contract(self) -> None:
        env = PingPongEEDeltaEnv()
        observation = env.reset()

        self.assertEqual(set(observation), {"joint_positions", "joint_velocities", "racket_position", "ball_position", "ball_velocity"})
        initial_target = env.target_position.copy()
        next_observation, reward, terminated, info = env.step((0.0, 0.0, 0.1))

        self.assertEqual(next_observation["joint_positions"].shape, (7,))
        self.assertEqual(next_observation["joint_velocities"].shape, (7,))
        self.assertEqual(next_observation["racket_position"].shape, (3,))
        self.assertEqual(next_observation["ball_position"].shape, (3,))
        self.assertEqual(next_observation["ball_velocity"].shape, (3,))
        self.assertAlmostEqual(float(info["applied_action"][2]), env.action_limit, places=6)
        self.assertGreater(float(info["target_position"][2]), float(initial_target[2]))
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)


if __name__ == "__main__":
    unittest.main()