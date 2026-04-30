"""State controller with online gate-pose adaptation for level 2.

Level 2 randomizes gates and obstacles around their nominal poses. The observation reports nominal
poses until an object enters sensor range, then switches to the measured pose. This controller
therefore avoids a fixed global spline: it uses the currently observed target gate pose and flies
through that gate from its local -x side to its local +x side.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

## this controller used

class Level2StateController(Controller):
    """State controller that replans from the visible target gate pose."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._dt = 1.0 / self._freq

        self._speed = 0.5
        self._lead_limit = 0.55
        self._approach_dist = 0.48
        self._exit_dist = 0.55
        self._waypoint_tol = 0.13

        self._cmd_pos = np.asarray(obs["pos"], dtype=float).copy()
        self._cmd_pos[2] = max(self._cmd_pos[2], 0.08)
        self._last_target_gate = int(np.asarray(obs["target_gate"]).item())
        self._stage = 0
        self._finished = False
        self._debug_points = np.empty((0, 3))
        self._last_exit_goal: NDArray[np.floating] | None = None
        self._last_yaw = 0.0
        self._clear_goal: NDArray[np.floating] | None = None
        self._clear_yaw = 0.0

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        target_gate = int(np.asarray(obs["target_gate"]).item())
        pos = np.asarray(obs["pos"], dtype=float)

        if target_gate == -1:
            self._finished = True
            return np.concatenate((pos, np.zeros(10)), dtype=np.float32)

        if target_gate != self._last_target_gate:
            if self._last_exit_goal is not None:
                self._clear_goal = self._last_exit_goal.copy()
                self._clear_yaw = self._last_yaw
            self._last_target_gate = target_gate
            self._stage = 0

        if self._clear_goal is not None:
            des_vel = self._move_setpoint(pos, self._clear_goal)
            if np.linalg.norm(pos - self._clear_goal) < self._waypoint_tol:
                self._clear_goal = None
                self._cmd_pos = pos.copy()
            return np.concatenate(
                (self._cmd_pos, des_vel, np.zeros(3), [self._clear_yaw, 0.0, 0.0, 0.0])
            ).astype(np.float32)

        gate_pos = np.asarray(obs["gates_pos"][target_gate], dtype=float)
        gate_quat = np.asarray(obs["gates_quat"][target_gate], dtype=float)
        gate_rot = R.from_quat(gate_quat).as_matrix()
        gate_x = gate_rot[:, 0]
        gate_x[2] = 0.0
        gate_x /= max(np.linalg.norm(gate_x), 1e-6)
        gate_y = gate_rot[:, 1]
        gate_y[2] = 0.0
        gate_y /= max(np.linalg.norm(gate_y), 1e-6)

        waypoints, center_stage = self._gate_waypoints(target_gate, gate_pos, gate_x, gate_y)
        if self._stage >= len(waypoints):
            self._stage = len(waypoints) - 1
        self._debug_points = waypoints
        self._last_exit_goal = waypoints[-1].copy()

        goal = waypoints[self._stage]
        if self._stage < len(waypoints) - 1 and np.linalg.norm(pos - goal) < self._waypoint_tol:
            self._stage += 1
            goal = waypoints[self._stage]

        if self._stage == center_stage:
            local_x = np.dot(pos - gate_pos, gate_x)
            if local_x > -0.08:
                self._stage = len(waypoints) - 1
                goal = waypoints[self._stage]

        des_vel = self._move_setpoint(pos, goal)
        yaw = float(np.arctan2(gate_x[1], gate_x[0]))
        self._last_yaw = yaw
        return np.concatenate((self._cmd_pos, des_vel, np.zeros(3), [yaw, 0.0, 0.0, 0.0])).astype(
            np.float32
        )

    def _gate_waypoints(
        self,
        target_gate: int,
        gate_pos: NDArray[np.floating],
        gate_x: NDArray[np.floating],
        gate_y: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], int]:
        pre_gate = gate_pos - self._approach_dist * gate_x
        post_gate = gate_pos + self._exit_dist * gate_x

        if target_gate == 0:
            pre_gate = gate_pos - 0.26 * gate_x - 0.22 * gate_y
            post_gate = gate_pos + 0.42 * gate_x - 0.20 * gate_y
            transfer = np.array([-0.50, 0.36, 0.58])
            return np.vstack((transfer, pre_gate, gate_pos, post_gate)), 2

        if target_gate == 1:
            pre_gate = gate_pos - 0.36 * gate_x - 0.18 * gate_y
            transfer = np.array([1.45, -0.08, 1.00])
            return np.vstack((transfer, pre_gate, gate_pos, post_gate)), 2

        if target_gate == 2:
            post_gate = gate_pos + 0.12 * gate_x
            transfer_east = np.array([0.55, 0.00, 1.22])
            transfer_south = np.array([-0.45, -0.05, 0.82])
            return np.vstack((transfer_east, transfer_south, pre_gate, gate_pos, post_gate)), 3

        if target_gate == 3:
            pre_gate = gate_pos - 0.18 * gate_x + 0.18 * gate_y
            post_gate = gate_pos + self._exit_dist * gate_x
            transfer_high = np.array([-1.05, 0.05, 1.18])
            return np.vstack((transfer_high, pre_gate, gate_pos, post_gate)), 2

        return np.vstack((pre_gate, gate_pos, post_gate)), 1

    def _move_setpoint(self, pos: NDArray[np.floating], goal: NDArray[np.floating]):
        to_goal = goal - self._cmd_pos
        dist = np.linalg.norm(to_goal)
        if dist < 1e-6:
            return np.zeros(3, dtype=np.float32)

        direction = to_goal / dist
        self._cmd_pos += direction * min(self._speed * self._dt, dist)

        lead = self._cmd_pos - pos
        lead_dist = np.linalg.norm(lead)
        if lead_dist > self._lead_limit:
            self._cmd_pos = pos + lead / lead_dist * self._lead_limit

        return (direction * self._speed).astype(np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        return self._finished

    def episode_callback(self):
        self._stage = 0
        self._finished = False

    def render_callback(self, sim: Sim):
        if len(self._debug_points):
            draw_points(sim, self._debug_points, rgba=(1.0, 0.0, 0.0, 1.0), size=0.025)
            draw_line(sim, self._debug_points, rgba=(0.0, 1.0, 0.0, 1.0))
