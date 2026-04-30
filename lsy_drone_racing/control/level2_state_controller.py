"""State controller with robust centerline gate crossing for level 2.

Level 2 randomizes gate and obstacle poses. Hard-coded world-frame transfer points therefore break
when the randomization is unfavorable. This controller derives every waypoint directly from the
currently observed gate pose and enforces a center corridor crossing before switching to the
post-gate target.
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


class Level2StateController(Controller):
    """State controller that tracks gate-relative pre/center/post waypoints."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller and set up internal state."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._dt = 1.0 / self._freq

        self._speed = 1.1
        self._lead_limit = 0.6
        self._approach_dist = 0.45
        self._exit_dist = 0.35
        self._waypoint_tol = 0.18
        self._post_gate_tol = 0.25
        self._center_pass_radius = 0.10
        self._center_pass_plane_x = 0.03
        self._takeoff_height = 0.4
        self._hold_after_finish = 0.4

        start_pos = np.asarray(obs["pos"], dtype=float).copy()
        self._cmd_pos = start_pos.copy()
        self._cmd_pos[2] = max(self._cmd_pos[2], 0.08)
        self._last_target_gate = int(np.asarray(obs["target_gate"]).item())
        self._stage = 0
        self._takeoff_done = start_pos[2] > 0.25
        self._finished = False
        self._hold_steps_remaining = 0
        self._debug_points = np.empty((0, 3))

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return the next desired full-state command for the drone."""
        target_gate = int(np.asarray(obs["target_gate"]).item())
        pos = np.asarray(obs["pos"], dtype=float)

        # After final gate, hold briefly to let env register completion.
        if target_gate == -1:
            if self._hold_steps_remaining <= 0:
                self._hold_steps_remaining = int(self._hold_after_finish * self._freq)
                self._cmd_pos = pos.copy()
            self._hold_steps_remaining -= 1
            if self._hold_steps_remaining <= 0:
                self._finished = True
            return np.concatenate(
                (self._cmd_pos, np.zeros(3), np.zeros(3), np.zeros(4))
            ).astype(np.float32)

        # Vertical takeoff first to avoid early skimming collisions.
        if not self._takeoff_done:
            takeoff_target = np.array([pos[0], pos[1], self._takeoff_height])
            des_vel = self._move_setpoint(pos, takeoff_target)
            if pos[2] >= self._takeoff_height - 0.05:
                self._takeoff_done = True
                self._stage = 0
            return np.concatenate(
                (self._cmd_pos, des_vel, np.zeros(3), np.zeros(4))
            ).astype(np.float32)

        # New target gate => reset per-gate stage.
        if target_gate != self._last_target_gate:
            self._last_target_gate = target_gate
            self._stage = 0

        gate_pos = np.asarray(obs["gates_pos"][target_gate], dtype=float)
        gate_quat = np.asarray(obs["gates_quat"][target_gate], dtype=float)
        gate_x, _ = self._gate_axes(gate_quat)
        waypoints = self._gate_waypoints(gate_pos, gate_x)
        self._debug_points = waypoints

        goal = waypoints[self._stage]
        if self._stage == 0 and np.linalg.norm(pos - goal) < self._waypoint_tol:
            self._stage = 1
            goal = waypoints[self._stage]

        if self._stage == 1:
            local_pos = R.from_quat(gate_quat).apply(pos - gate_pos, inverse=True)
            lateral_dist = float(np.linalg.norm(local_pos[1:]))
            passed_center = (
                local_pos[0] > self._center_pass_plane_x and lateral_dist < self._center_pass_radius
            )
            if passed_center:
                self._stage = 2
                goal = waypoints[self._stage]

        if self._stage == 2 and np.linalg.norm(pos - goal) < self._post_gate_tol:
            pass

        des_vel = self._move_setpoint(pos, goal)
        return np.concatenate((self._cmd_pos, des_vel, np.zeros(3), np.zeros(4))).astype(np.float32)

    def _gate_axes(
        self, gate_quat: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return unit vectors along gate x/y axes in world frame."""
        rot = R.from_quat(gate_quat).as_matrix()
        gate_x = rot[:, 0].copy()
        gate_x /= max(np.linalg.norm(gate_x), 1e-6)
        gate_y = rot[:, 1].copy()
        gate_y /= max(np.linalg.norm(gate_y), 1e-6)
        return gate_x, gate_y

    def _gate_waypoints(
        self,
        gate_pos: NDArray[np.floating],
        gate_x: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Construct pre/center/post waypoints in gate-local x direction."""
        pre_gate = gate_pos - self._approach_dist * gate_x
        center = gate_pos.copy()
        post_gate = gate_pos + self._exit_dist * gate_x
        return np.vstack((pre_gate, center, post_gate))

    def _move_setpoint(
        self, pos: NDArray[np.floating], goal: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Move the commanded setpoint smoothly toward the active waypoint."""
        to_goal = goal - self._cmd_pos
        dist = float(np.linalg.norm(to_goal))
        if dist < 1e-6:
            return np.zeros(3, dtype=np.float32)

        direction = to_goal / dist
        self._cmd_pos += direction * min(self._speed * self._dt, dist)

        lead = self._cmd_pos - pos
        lead_dist = float(np.linalg.norm(lead))
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
        """Return whether the run has finished."""
        return self._finished

    def episode_callback(self) -> None:
        """Reset internal state between episodes."""
        self._stage = 0
        self._finished = False
        self._takeoff_done = False
        self._hold_steps_remaining = 0

    def render_callback(self, sim: Sim) -> None:
        """Draw current waypoints for debugging."""
        if len(self._debug_points):
            draw_points(sim, self._debug_points, rgba=(1.0, 0.0, 0.0, 1.0), size=0.025)
            draw_line(sim, self._debug_points, rgba=(0.0, 1.0, 0.0, 1.0))
