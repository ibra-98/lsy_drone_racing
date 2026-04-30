"""State controller with online gate-pose adaptation for level 2.

Level 2 randomizes gates and obstacles around their nominal poses. The observation reports nominal
poses until an object enters sensor range, then switches to the measured pose. This controller
avoids any hard-coded world coordinates: every waypoint is derived from the currently observed
target gate pose, so the trajectory automatically follows the gates as their poses are revealed.

The controller emits full-state setpoints
``[x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]`` for the ``"state"`` control mode.
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
    """State controller that always plans relative to the observed target gate pose."""

    # Tunables. Defaults are conservative; raise SPEED once the run is reliable.
    SPEED = 0.8  # m/s, target setpoint speed along the path
    LEAD_LIMIT = 0.6  # m, max distance the leading setpoint may run ahead of the drone
    APPROACH_DIST = 0.55  # m, distance of the pre-gate waypoint along the gate axis
    EXIT_DIST = 0.45  # m, distance of the post-gate waypoint along the gate axis
    WAYPOINT_TOL = 0.18  # m, distance at which a waypoint is considered reached
    POST_GATE_TOL = 0.25  # m, looser tolerance for clearing the post-gate point
    CENTER_PASS_RADIUS = 0.10  # m, require crossing near the gate center (yz in gate frame)
    CENTER_PASS_PLANE_X = 0.03  # m, require being slightly past the center plane before exit
    TAKEOFF_HEIGHT = 0.4  # m, intermediate height when starting from the floor
    HOLD_AFTER_FINISH = 0.4  # s, hover briefly after the last gate so the env registers a finish

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller and set up internal state."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._dt = 1.0 / self._freq

        start_pos = np.asarray(obs["pos"], dtype=float).copy()
        # Lift the initial setpoint a little so the first command is not below the floor.
        self._cmd_pos = start_pos.copy()
        self._cmd_pos[2] = max(self._cmd_pos[2], 0.05)

        self._last_target_gate = int(np.asarray(obs["target_gate"]).item())
        self._stage = 0  # Index into the per-gate waypoint list
        self._takeoff_done = start_pos[2] > 0.25
        self._finished = False
        self._hold_steps_remaining = 0
        self._debug_points = np.empty((0, 3))

    # ------------------------------------------------------------------
    # Waypoint construction (all gate-relative)
    # ------------------------------------------------------------------

    def _gate_axes(
        self, gate_quat: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return unit vectors along the gate's passing axis and lateral axis."""
        rot = R.from_quat(gate_quat).as_matrix()
        gate_x = rot[:, 0].copy()
        gate_x /= max(np.linalg.norm(gate_x), 1e-6)
        gate_y = rot[:, 1].copy()
        gate_y /= max(np.linalg.norm(gate_y), 1e-6)
        return gate_x, gate_y

    def _gate_waypoints(
        self, gate_pos: NDArray[np.floating], gate_x: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Three waypoints per gate, all in the gate's local frame.

        We slightly bias the pre-gate setpoint upward so the drone does not skim obstacles
        approaching short gates from below.
        """
        pre_gate = gate_pos - self.APPROACH_DIST * gate_x
        center = gate_pos.copy()
        post_gate = gate_pos + self.EXIT_DIST * gate_x
        return np.vstack((pre_gate, center, post_gate))

    # ------------------------------------------------------------------
    # Setpoint smoothing
    # ------------------------------------------------------------------

    def _move_setpoint(
        self, drone_pos: NDArray[np.floating], goal: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Advance ``self._cmd_pos`` toward ``goal`` and return a velocity feedforward."""
        to_goal = goal - self._cmd_pos
        dist = float(np.linalg.norm(to_goal))
        if dist < 1e-6:
            return np.zeros(3, dtype=np.float32)

        direction = to_goal / dist
        step = min(self.SPEED * self._dt, dist)
        self._cmd_pos = self._cmd_pos + direction * step

        # Don't let the leading setpoint run too far ahead of the actual drone state.
        lead = self._cmd_pos - drone_pos
        lead_dist = float(np.linalg.norm(lead))
        if lead_dist > self.LEAD_LIMIT:
            self._cmd_pos = drone_pos + lead / lead_dist * self.LEAD_LIMIT

        return (direction * self.SPEED).astype(np.float32)

    # ------------------------------------------------------------------
    # Controller interface
    # ------------------------------------------------------------------

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return the next desired full-state command for the drone."""
        target_gate = int(np.asarray(obs["target_gate"]).item())
        pos = np.asarray(obs["pos"], dtype=float)

        # 1) After the final gate is passed, hold position briefly, then mark finished.
        if target_gate == -1:
            if self._hold_steps_remaining <= 0:
                self._hold_steps_remaining = int(self.HOLD_AFTER_FINISH * self._freq)
                self._cmd_pos = pos.copy()
            self._hold_steps_remaining -= 1
            if self._hold_steps_remaining <= 0:
                self._finished = True
            return np.concatenate(
                (self._cmd_pos, np.zeros(3), np.zeros(3), np.zeros(4))
            ).astype(np.float32)

        # 2) Optional take-off: if we are still on the ground, climb straight up first.
        if not self._takeoff_done:
            takeoff_target = np.array([pos[0], pos[1], self.TAKEOFF_HEIGHT])
            des_vel = self._move_setpoint(pos, takeoff_target)
            if pos[2] >= self.TAKEOFF_HEIGHT - 0.05:
                self._takeoff_done = True
                self._stage = 0
            return np.concatenate(
                (self._cmd_pos, des_vel, np.zeros(3), np.zeros(4))
            ).astype(np.float32)

        # 3) Detect target-gate change → reset stage so we re-approach the new gate.
        if target_gate != self._last_target_gate:
            self._last_target_gate = target_gate
            self._stage = 0

        # 4) Build waypoints from the *currently observed* gate pose. Once the drone enters
        # the 0.7 m sensor range these are the true (randomized) positions; before that they
        # are the nominal positions from the config.
        gate_pos = np.asarray(obs["gates_pos"][target_gate], dtype=float)
        gate_quat = np.asarray(obs["gates_quat"][target_gate], dtype=float)
        gate_x, _ = self._gate_axes(gate_quat)
        waypoints = self._gate_waypoints(gate_pos, gate_x)
        self._debug_points = waypoints

        # 5) Stage advancement.
        # Stage 0: pre-gate; advance once close enough.
        # Stage 1: gate center; only jump to post-gate after passing near gate center.
        # Stage 2: post-gate; once close, the env will normally have updated target_gate and
        #          we will reset stage at the top of the next call.
        goal = waypoints[self._stage]
        if self._stage == 0 and np.linalg.norm(pos - goal) < self.WAYPOINT_TOL:
            self._stage = 1
            goal = waypoints[self._stage]
        if self._stage == 1:
            local_pos = R.from_quat(gate_quat).apply(pos - gate_pos, inverse=True)
            lateral_dist = float(np.linalg.norm(local_pos[1:]))
            passed_center = (
                local_pos[0] > self.CENTER_PASS_PLANE_X and lateral_dist < self.CENTER_PASS_RADIUS
            )
            if passed_center:  # Exit only after crossing through the center corridor.
                self._stage = 2
                goal = waypoints[self._stage]
        if self._stage == 2 and np.linalg.norm(pos - goal) < self.POST_GATE_TOL:
            # Stay on the exit point; the next call will pick up the new target gate.
            pass

        des_vel = self._move_setpoint(pos, goal)

        return np.concatenate(
            (self._cmd_pos, des_vel, np.zeros(3), np.zeros(4))
        ).astype(np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Return whether the controller has finished the run."""
        return self._finished

    def episode_callback(self) -> None:
        """Reset internal state between episodes."""
        self._stage = 0
        self._finished = False
        self._takeoff_done = False
        self._hold_steps_remaining = 0

    def render_callback(self, sim: Sim) -> None:
        """Visualize the current waypoints and setpoint when rendering is enabled."""
        if len(self._debug_points):
            draw_points(sim, self._debug_points, rgba=(1.0, 0.0, 0.0, 1.0), size=0.025)
            draw_line(sim, self._debug_points, rgba=(0.0, 1.0, 0.0, 1.0))
