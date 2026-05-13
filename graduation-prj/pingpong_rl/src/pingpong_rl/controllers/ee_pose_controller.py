from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np

from pingpong_rl.envs.pingpong_env import PingPongSim


class RacketCartesianController:
    def __init__(
        self,
        sim: PingPongSim,
        damping: float = 1.0e-4,
        position_gain: float = 0.6,
        max_position_step: float = 0.03,
    ) -> None:
        self._sim = sim
        self._damping = float(damping)
        self._position_gain = float(position_gain)
        self._max_position_step = float(max_position_step)

        self._joint_ids = [sim.model.joint(f"joint{index}").id for index in range(1, 8)]
        self._joint_qpos_indices = np.array([sim.model.jnt_qposadr[joint_id] for joint_id in self._joint_ids], dtype=int)
        self._joint_dof_indices = np.array([sim.model.jnt_dofadr[joint_id] for joint_id in self._joint_ids], dtype=int)
        self._joint_limits = sim.model.jnt_range[self._joint_ids].copy()
        self._jacobian = np.zeros((3, sim.model.nv), dtype=float)
        self.targets = sim.home_joint_targets.copy()
        self._target_position = sim.racket_position.copy()

    @property
    def target_position(self) -> np.ndarray:
        return self._target_position.copy()

    def reset(self) -> np.ndarray:
        self.targets[:] = self._sim.home_joint_targets
        self._target_position = self._sim.racket_position.copy()
        return self.targets.copy()

    def set_target_position(self, position: Sequence[float]) -> np.ndarray:
        position_array = np.asarray(position, dtype=float)
        if position_array.shape != (3,):
            raise ValueError(f"Target position must have shape (3,), got {position_array.shape}.")
        self._target_position = position_array.copy()
        return self.target_position

    def add_target_offset(self, delta: Sequence[float]) -> np.ndarray:
        delta_array = np.asarray(delta, dtype=float)
        if delta_array.shape != (3,):
            raise ValueError(f"Target delta must have shape (3,), got {delta_array.shape}.")
        self._target_position = self._target_position + delta_array
        return self.target_position

    def compute_joint_targets(self) -> np.ndarray:
        current_position = self._sim.racket_position
        position_error = self._target_position - current_position
        error_norm = np.linalg.norm(position_error)
        if error_norm > self._max_position_step:
            position_error = position_error * (self._max_position_step / error_norm)

        mujoco.mj_jacSite(
            self._sim.model,
            self._sim.data,
            self._jacobian,
            None,
            self._sim.racket_site_id,
        )
        task_jacobian = self._jacobian[:, self._joint_dof_indices]
        task_metric = task_jacobian @ task_jacobian.T + self._damping * np.eye(3)
        delta_q = task_jacobian.T @ np.linalg.solve(task_metric, self._position_gain * position_error)

        current_joint_positions = self._sim.data.qpos[self._joint_qpos_indices]
        next_targets = current_joint_positions + delta_q
        clipped_targets = np.clip(next_targets, self._joint_limits[:, 0], self._joint_limits[:, 1])
        self.targets[:] = clipped_targets
        return self.targets.copy()