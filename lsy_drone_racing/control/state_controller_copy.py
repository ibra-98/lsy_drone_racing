"""Adaptive trajectory controller for Level 2.

Builds a smooth cubic spline through the start position, the four gates, and a finish point.
Each gate contributes three waypoints: an approach point in front of the gate, the gate center,
and an exit point behind the gate, computed from the gate's orientation. Whenever the observed
gate poses change (which happens once the drone enters the sensor range and the true positions are
revealed), the trajectory is rebuilt from the current drone state so that subsequent gates are
hit at their actual locations rather than at their nominal ones.

The controller outputs full-state commands ``[x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate,
yrate]`` consumed by the environment when ``env.control_mode = "state"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

    try:
        from crazyflow import Sim
    except Exception:  # pragma: no cover - only needed for the optional render hook
        Sim = object  # type: ignore[assignment]


GATE_AXIS = np.array([1.0, 0.0, 0.0])  # Gate passing direction in gate-local frame


class TrajectoryController(Controller):
    """Spline-based trajectory tracker that replans when gate observations are updated."""

    # Tunables. Conservative defaults that pass Level 2 reliably; lower the time / increase
    # the speed once the basic controller works.
    NOMINAL_SPEED = 1.4  # m/s, target average speed used to time the spline segments
    APPROACH_OFFSET = 0.3  # m, distance of pre/post gate waypoints from the gate center
    TAKEOFF_HEIGHT = 0.4  # m, height of an extra waypoint above the start position
    POSE_CHANGE_TOL = 0.02  # m, rebuild the plan if a gate moved more than this in xy/z
    QUAT_CHANGE_TOL = np.deg2rad(2.0)  # rad, or rotated more than this
    FINISH_HOLD_TIME = 0.5  # s, extra time spent past the last gate before stopping

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller and build the initial plan from the first observation.

        Args:
            obs: Initial environment observation.
            info: Initial info dict from ``env.reset``.
            config: The race configuration (``ml_collections.ConfigDict``).
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._dt = 1.0 / self._freq
        self._tick = 0
        self._finished = False

        # Cached gate poses used in the current plan; we replan when these go stale.
        self._planned_gates_pos = np.asarray(obs["gates_pos"], dtype=np.float64).copy()
        self._planned_gates_quat = np.asarray(obs["gates_quat"], dtype=np.float64).copy()

        # Build the initial plan from the start position with zero initial velocity.
        start_pos = np.asarray(obs["pos"], dtype=np.float64)
        start_vel = np.asarray(obs["vel"], dtype=np.float64)
        self._plan_from_state(
            start_pos=start_pos,
            start_vel=start_vel,
            target_gate=int(obs["target_gate"]),
            tick=self._tick,
        )

    # ------------------------------------------------------------------
    # Plan construction
    # ------------------------------------------------------------------

    def _gate_axis_world(self, gate_quat: NDArray[np.floating]) -> NDArray[np.floating]:
        """World-frame unit vector along the gate passing direction (gate-local +x)."""
        return R.from_quat(gate_quat).apply(GATE_AXIS)

    def _build_waypoints(
        self, start_pos: NDArray[np.floating], target_gate: int
    ) -> NDArray[np.floating]:
        """Stack waypoints from the current drone position through every remaining gate."""
        waypoints: list[NDArray[np.floating]] = [start_pos]

        # Add a takeoff waypoint when starting from very low altitude. This gives the spline a
        # chance to rise gently instead of swinging through the ground when the first gate is
        # high up.
        if start_pos[2] < 0.2:
            waypoints.append(
                np.array([start_pos[0], start_pos[1], max(self.TAKEOFF_HEIGHT, start_pos[2] + 0.2)])
            )

        n_gates = self._planned_gates_pos.shape[0]
        target_gate = max(0, min(target_gate, n_gates - 1))

        for i in range(target_gate, n_gates):
            gate_pos = self._planned_gates_pos[i]
            axis = self._gate_axis_world(self._planned_gates_quat[i])
            before = gate_pos - axis * self.APPROACH_OFFSET
            after = gate_pos + axis * self.APPROACH_OFFSET

            # Skip the "before" waypoint if we are already past it (e.g. just passed gate i-1
            # and the next gate sits right next to us).
            if np.linalg.norm(before - waypoints[-1]) > 0.05:
                waypoints.append(before)
            waypoints.append(gate_pos)
            waypoints.append(after)

        # Add a finish waypoint a bit further beyond the last gate.
        last_gate_pos = self._planned_gates_pos[n_gates - 1]
        last_axis = self._gate_axis_world(self._planned_gates_quat[n_gates - 1])
        waypoints.append(last_gate_pos + last_axis * (self.APPROACH_OFFSET + 0.3))

        return np.asarray(waypoints, dtype=np.float64)

    def _plan_from_state(
        self,
        start_pos: NDArray[np.floating],
        start_vel: NDArray[np.floating],
        target_gate: int,
        tick: int,
    ) -> None:
        """Build a cubic spline from ``start_pos`` through all remaining gates.

        The spline is parameterised by absolute simulation time (in seconds) so that the
        controller can index into it using ``self._tick / self._freq`` regardless of when
        the plan was last rebuilt.
        """
        if target_gate < 0:  # All gates already passed; hold the current pose.
            self._t_start = tick * self._dt
            self._t_end = self._t_start + self.FINISH_HOLD_TIME
            t = np.array([self._t_start, self._t_end])
            pts = np.vstack([start_pos, start_pos])
            self._spline = CubicSpline(t, pts)
            self._spline_vel = self._spline.derivative()
            return

        waypoints = self._build_waypoints(start_pos, target_gate)

        # Time per segment ~ chord length / nominal speed, with a small floor so very short
        # segments do not produce extreme velocities.
        seg_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        seg_times = np.maximum(seg_lengths / self.NOMINAL_SPEED, 0.15)
        cum = np.concatenate(([0.0], np.cumsum(seg_times)))
        t = tick * self._dt + cum

        # Clamped boundary at the start (match current velocity), natural at the end.
        bc = ((1, np.asarray(start_vel, dtype=np.float64)), (2, np.zeros(3)))
        self._spline = CubicSpline(t, waypoints, bc_type=bc)
        self._spline_vel = self._spline.derivative()
        self._t_start = float(t[0])
        self._t_end = float(t[-1]) + self.FINISH_HOLD_TIME

    # ------------------------------------------------------------------
    # Replan trigger
    # ------------------------------------------------------------------

    def _gates_changed(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Return True when at least one observed gate pose has moved beyond tolerance."""
        new_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
        new_quat = np.asarray(obs["gates_quat"], dtype=np.float64)
        if np.any(np.linalg.norm(new_pos - self._planned_gates_pos, axis=1) > self.POSE_CHANGE_TOL):
            return True
        # Compare quaternions via the dot product, then convert to angle.
        dots = np.abs(np.sum(new_quat * self._planned_gates_quat, axis=1)).clip(0.0, 1.0)
        ang = 2.0 * np.arccos(dots)
        return bool(np.any(ang > self.QUAT_CHANGE_TOL))

    # ------------------------------------------------------------------
    # Controller interface
    # ------------------------------------------------------------------

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return the next desired full-state command for the drone.

        Args:
            obs: Current environment observation.
            info: Optional info dict (unused here).

        Returns:
            Length-13 ``np.float32`` array ``[x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate,
            yrate]`` understood by the environment when running in ``"state"`` control mode.
        """
        if self._gates_changed(obs):
            self._planned_gates_pos = np.asarray(obs["gates_pos"], dtype=np.float64).copy()
            self._planned_gates_quat = np.asarray(obs["gates_quat"], dtype=np.float64).copy()
            self._plan_from_state(
                start_pos=np.asarray(obs["pos"], dtype=np.float64),
                start_vel=np.asarray(obs["vel"], dtype=np.float64),
                target_gate=int(obs["target_gate"]),
                tick=self._tick,
            )

        t = min(self._tick * self._dt, self._t_end)
        if t >= self._t_end:
            self._finished = True
            t = self._t_end

        des_pos = self._spline(t)
        des_vel = self._spline_vel(t) if t < self._spline.x[-1] else np.zeros(3)

        action = np.concatenate(
            [des_pos, des_vel, np.zeros(3), np.zeros(1), np.zeros(3)]
        ).astype(np.float32)
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time counter and report whether the trajectory is exhausted."""
        self._tick += 1
        return self._finished

    def episode_callback(self) -> None:
        """Reset internal counters between episodes."""
        self._tick = 0
        self._finished = False

    def render_callback(self, sim: Sim) -> None:
        """Draw the planned trajectory in the simulator window when rendering is enabled."""
        try:
            from crazyflow.sim.visualize import draw_line, draw_points
        except Exception:
            return
        ts = np.linspace(self._t_start, self._spline.x[-1], 200)
        traj = self._spline(ts)
        draw_line(sim, traj, rgba=(0.0, 1.0, 0.0, 1.0))
        setpoint = self._spline(min(self._tick * self._dt, self._spline.x[-1])).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)