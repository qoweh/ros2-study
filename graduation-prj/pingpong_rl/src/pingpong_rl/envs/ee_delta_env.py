from __future__ import annotations

from typing import Sequence

import numpy as np

from pingpong_rl.controllers import RacketCartesianController
from pingpong_rl.envs.pingpong_env import PingPongSim


_OBSERVATION_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("joint_positions", 7),
    ("joint_velocities", 7),
    ("racket_position", 3),
    ("target_position", 3),
    ("ball_position", 3),
    ("ball_velocity", 3),
)

_OBSERVATION_SLICES: dict[str, slice] = {}
_observation_offset = 0
for component_name, component_size in _OBSERVATION_COMPONENTS:
    _OBSERVATION_SLICES[component_name] = slice(_observation_offset, _observation_offset + component_size)
    _observation_offset += component_size
_OBSERVATION_SIZE = _observation_offset


class PingPongEEDeltaEnv:
    def __init__(
        self,
        sim: PingPongSim | None = None,
        action_limit: float = 0.03,
        ball_height: float = 0.22,
        contact_bonus: float = 1.0,
        height_reward_weight: float = 0.1,
        floor_penalty: float = -1.0,
        failure_penalty: float = -0.25,
    ) -> None:
        self.sim = PingPongSim() if sim is None else sim
        self.action_limit = float(action_limit)
        self.ball_height = float(ball_height)
        self.contact_bonus = float(contact_bonus)
        self.height_reward_weight = float(height_reward_weight)
        self.floor_penalty = float(floor_penalty)
        self.failure_penalty = float(failure_penalty)
        self.controller = RacketCartesianController(self.sim, max_position_step=self.action_limit)

    @property
    def target_position(self) -> np.ndarray:
        return self.controller.target_position

    @property
    def observation_size(self) -> int:
        return _OBSERVATION_SIZE

    @property
    def observation_slices(self) -> dict[str, slice]:
        return _OBSERVATION_SLICES.copy()

    def observation_dict(self) -> dict[str, np.ndarray]:
        return {
            "joint_positions": self.sim.joint_positions,
            "joint_velocities": self.sim.joint_velocities,
            "racket_position": self.sim.racket_position,
            "target_position": self.controller.target_position,
            "ball_position": self.sim.ball_position,
            "ball_velocity": self.sim.ball_velocity,
        }

    def observation(self) -> np.ndarray:
        observation_dict = self.observation_dict()
        return np.concatenate([observation_dict[name] for name, _ in _OBSERVATION_COMPONENTS])

    @classmethod
    def unflatten_observation(cls, observation: Sequence[float]) -> dict[str, np.ndarray]:
        observation_array = np.asarray(observation, dtype=float)
        if observation_array.shape != (_OBSERVATION_SIZE,):
            raise ValueError(f"Flat observation must have shape ({_OBSERVATION_SIZE},), got {observation_array.shape}.")

        return {
            name: observation_array[component_slice].copy()
            for name, component_slice in _OBSERVATION_SLICES.items()
        }

    def reset(
        self,
        ball_height: float | None = None,
        ball_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> tuple[np.ndarray, dict[str, object]]:
        spawn_height = self.ball_height if ball_height is None else float(ball_height)
        self.sim.reset(ball_height=spawn_height, ball_velocity=ball_velocity)
        self.controller.reset()
        info: dict[str, object] = {
            "failure_reason": None,
            "target_position": self.controller.target_position,
        }
        return self.observation(), info

    def step(self, action: Sequence[float]) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (3,):
            raise ValueError(f"EE delta action must have shape (3,), got {action_array.shape}.")

        applied_action = np.clip(action_array, -self.action_limit, self.action_limit)
        self.controller.add_target_offset(applied_action)
        joint_targets = self.controller.compute_joint_targets()
        self.sim.step(joint_targets=joint_targets, n_substeps=self.sim.n_substeps)

        failure_reason = self.sim.failure_reason()
        reward_terms = self._reward_terms(failure_reason)
        reward = float(sum(reward_terms.values()))
        terminated = failure_reason is not None
        truncated = False
        info: dict[str, object] = {
            "applied_action": applied_action.copy(),
            "target_position": self.controller.target_position,
            "failure_reason": failure_reason,
            "reward_terms": reward_terms,
            "racket_contact": self.sim.has_contact("ball_geom", "racket_head"),
        }
        return self.observation(), reward, terminated, truncated, info

    def _reward_terms(self, failure_reason: str | None) -> dict[str, float]:
        reward_terms: dict[str, float] = {
            "contact_bonus": 0.0,
            "height_term": self.height_reward_weight
            * max(float(self.sim.ball_position[2] - self.sim.racket_position[2]), 0.0),
            "failure_penalty": 0.0,
        }
        if self.sim.has_contact("ball_geom", "racket_head"):
            reward_terms["contact_bonus"] = self.contact_bonus
        if failure_reason == "floor_contact":
            reward_terms["failure_penalty"] = self.floor_penalty
        elif failure_reason is not None:
            reward_terms["failure_penalty"] = self.failure_penalty
        return reward_terms