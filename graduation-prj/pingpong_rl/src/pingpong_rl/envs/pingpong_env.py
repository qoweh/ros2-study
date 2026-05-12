from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from pingpong_rl.utils.paths import SCENE_XML_PATH


class PingPongSim:
    def __init__(self, scene_path: Path | str | None = None, control_dt: float = 0.02) -> None:
        scene_file = Path(scene_path) if scene_path is not None else SCENE_XML_PATH
        self.scene_path = scene_file.resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.control_dt = float(control_dt)
        self.n_substeps = max(1, int(round(self.control_dt / self.model.opt.timestep)))

        self.ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.racket_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "racket")

        if self.ball_joint_id < 0 or self.ball_body_id < 0 or self.racket_body_id < 0:
            raise ValueError("Scene is missing one of the required bodies or joints: ball_joint, ball, racket.")

        self._ball_qpos_adr = self.model.jnt_qposadr[self.ball_joint_id]
        self._ball_dof_adr = self.model.jnt_dofadr[self.ball_joint_id]
        self._home_ctrl = self.model.key_ctrl[0].copy()
        self._home_joint_targets = self._home_ctrl[:7].copy()
        self.reset()

    @property
    def home_joint_targets(self) -> np.ndarray:
        return self._home_joint_targets.copy()

    @property
    def ball_position(self) -> np.ndarray:
        return self.data.xpos[self.ball_body_id].copy()

    @property
    def racket_position(self) -> np.ndarray:
        return self.data.xpos[self.racket_body_id].copy()

    def reset(self, ball_position: Sequence[float] = (0.48, 0.0, 0.95), ball_velocity: Sequence[float] = (0.0, 0.0, 0.0)) -> mujoco.MjData:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:] = self._home_ctrl
        self.data.ctrl[7] = 255.0
        self.spawn_ball(ball_position, ball_velocity)
        mujoco.mj_forward(self.model, self.data)
        return self.data

    def spawn_ball(self, position: Sequence[float], velocity: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        position_array = np.asarray(position, dtype=float)
        velocity_array = np.asarray(velocity, dtype=float)

        if position_array.shape != (3,):
            raise ValueError(f"Ball position must have shape (3,), got {position_array.shape}.")
        if velocity_array.shape != (3,):
            raise ValueError(f"Ball velocity must have shape (3,), got {velocity_array.shape}.")

        qpos = self.data.qpos
        qvel = self.data.qvel

        qpos[self._ball_qpos_adr:self._ball_qpos_adr + 3] = position_array
        qpos[self._ball_qpos_adr + 3:self._ball_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        qvel[self._ball_dof_adr:self._ball_dof_adr + 3] = velocity_array
        qvel[self._ball_dof_adr + 3:self._ball_dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.ball_position

    def reset_ball_above_racket(
        self,
        height: float = 0.25,
        xy_offset: Sequence[float] = (0.0, 0.0),
        velocity: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        xy_offset_array = np.asarray(xy_offset, dtype=float)
        if xy_offset_array.shape != (2,):
            raise ValueError(f"xy_offset must have shape (2,), got {xy_offset_array.shape}.")

        spawn_position = self.racket_position + np.array([xy_offset_array[0], xy_offset_array[1], height])
        return self.spawn_ball(spawn_position, velocity)

    def set_arm_joint_targets(self, joint_targets: Sequence[float], gripper_target: float = 255.0) -> np.ndarray:
        joint_targets_array = np.asarray(joint_targets, dtype=float)
        if joint_targets_array.shape != (7,):
            raise ValueError(f"Arm targets must have shape (7,), got {joint_targets_array.shape}.")

        self.data.ctrl[:7] = joint_targets_array
        self.data.ctrl[7] = gripper_target
        return self.data.ctrl[:8].copy()

    def step(
        self,
        joint_targets: Sequence[float] | None = None,
        gripper_target: float = 255.0,
        n_substeps: int | None = None,
    ) -> mujoco.MjData:
        if joint_targets is not None:
            self.set_arm_joint_targets(joint_targets, gripper_target)

        step_count = self.n_substeps if n_substeps is None else max(1, int(n_substeps))
        for _ in range(step_count):
            mujoco.mj_step(self.model, self.data)
        return self.data