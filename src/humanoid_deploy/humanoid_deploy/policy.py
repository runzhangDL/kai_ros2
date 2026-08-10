"""Numpy inference for the exported brax PPO policy, plus a sanity check on
whether the robot's observations actually look like what the policy was
trained on.

No jax on the robot: the bundle written by ``tools/export_policy.py`` is a
plain ``.npz``, the forward pass is four matrix multiplies, and the whole thing
takes well under a millisecond. The exporter verifies this implementation
against brax's own inference function before it will write a bundle, so the
two cannot silently drift.

Network, as confirmed against brax 0.14.2's source:
    normalize -> Dense/swish x3 -> Dense(2*nu) -> take loc -> tanh
The second half of the output is the policy's std, which ``mode()`` discards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


class PolicyError(Exception):
    """Raised when the bundle is missing, malformed, or mismatched."""


def swish(x: np.ndarray) -> np.ndarray:
    """SiLU, brax's default MLP activation."""
    return x / (1.0 + np.exp(-x))


#: Labels for the walking policy's five extra observation entries, in order.
#: ``mjx_walk_free_env`` appends the gait clock and the velocity command after
#: the 48 entries it shares with the standing policy.
GAIT_TAIL_LABELS = ("phase_sin", "phase_cos", "cmd_vx", "cmd_vy", "cmd_wz")


@dataclass(frozen=True)
class ObservationLayout:
    """Where each quantity lives in the observation, derived from nu.

    Mirrors ``mjx_stand_env._get_obs``:
        accel(3) gyro(3) gravity(3) qpos(nu) qvel(nu) last_action(nu)

    ``mjx_walk_free_env`` is a strict superset: the first ``9 + 3*nu`` entries
    are byte-identical, and ``extra`` more follow (the gait clock and the
    velocity command). Keeping one layout for both is what lets the same
    observation builder feed either policy, and it is why ``last_action`` must
    stay at index ``9 + 2*nu`` -- do not reorder.
    """

    nu: int
    extra: int = 0

    @property
    def accel(self) -> slice:
        return slice(0, 3)

    @property
    def gyro(self) -> slice:
        return slice(3, 6)

    @property
    def gravity(self) -> slice:
        return slice(6, 9)

    @property
    def qpos(self) -> slice:
        return slice(9, 9 + self.nu)

    @property
    def qvel(self) -> slice:
        return slice(9 + self.nu, 9 + 2 * self.nu)

    @property
    def action(self) -> slice:
        return slice(9 + 2 * self.nu, 9 + 3 * self.nu)

    @property
    def tail(self) -> slice:
        """The gait clock and velocity command; empty for the stand policy."""
        return slice(9 + 3 * self.nu, 9 + 3 * self.nu + self.extra)

    @property
    def stand_size(self) -> int:
        """Length of the prefix the two policies share."""
        return 9 + 3 * self.nu

    @property
    def size(self) -> int:
        return 9 + 3 * self.nu + self.extra

    def label(self, index: int) -> str:
        names = (["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
                  "grav_x", "grav_y", "grav_z"]
                 + [f"qpos_{i}" for i in range(self.nu)]
                 + [f"qvel_{i}" for i in range(self.nu)]
                 + [f"act_{i}" for i in range(self.nu)]
                 + list(GAIT_TAIL_LABELS[: self.extra])
                 + [f"tail_{i}" for i in range(len(GAIT_TAIL_LABELS), self.extra)])
        return names[index] if 0 <= index < len(names) else f"obs_{index}"


class Policy:
    """The trained standing policy, plus the statistics it was trained on."""

    def __init__(self, bundle_path: str) -> None:
        try:
            data = np.load(bundle_path, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001
            raise PolicyError(f"cannot read policy bundle {bundle_path}: {exc}") from exc

        try:
            self.mean = data["mean"].astype(np.float64)
            self.std = data["std"].astype(np.float64)
            n_layers = int(data["n_layers"])
            self.layers = [
                (data[f"w{i}"].astype(np.float64), data[f"b{i}"].astype(np.float64))
                for i in range(n_layers)
            ]
            self.meta = json.loads(str(data["meta"]))
        except KeyError as exc:
            raise PolicyError(
                f"{bundle_path} is missing {exc}; re-export it with "
                "tools/export_policy.py"
            ) from exc

        if np.any(self.std <= 0) or not np.all(np.isfinite(self.std)):
            raise PolicyError("normalizer std contains non-positive or non-finite values")

        self.obs_size = int(self.layers[0][0].shape[0])
        self.nu = int(self.layers[-1][0].shape[1] // 2)
        # The bundle states its own tail length rather than letting this code
        # infer one from the observation size: an inferred tail would silently
        # absorb a genuine mismatch (a 53-input network fed by a 48-entry
        # builder) into "oh, five extra entries" instead of failing.
        extra = int(self.meta.get("obs_extra", 0))
        self.layout = ObservationLayout(self.nu, extra)
        if self.layout.size != self.obs_size:
            raise PolicyError(
                f"bundle obs size {self.obs_size} does not match the "
                f"9 + 3*{self.nu} + {extra} = {self.layout.size} layout the "
                "bundle's metadata declares"
            )
        if self.mean.shape != (self.obs_size,) or self.std.shape != (self.obs_size,):
            raise PolicyError("normalizer shape does not match the network input")

        self.joint_names: list[str] = list(self.meta["joint_names"])
        self.action_scale = float(self.meta["action_scale"])
        self.control_dt = float(self.meta["control_dt"])
        self.xml_lower = np.asarray(self.meta["xml_lower_rad"], dtype=np.float64)
        self.xml_upper = np.asarray(self.meta["xml_upper_rad"], dtype=np.float64)
        self.default_pose = np.asarray(self.meta["default_pose_rad"], dtype=np.float64)
        #: Gait metadata, present only for the walking bundle. Everything the
        #: node needs to advance the clock and pick a legal command lives here,
        #: extracted from the training sidecar at export time so no number is
        #: ever re-typed into a YAML file.
        self.gait: dict = dict(self.meta.get("gait") or {})

        if len(self.joint_names) != self.nu:
            raise PolicyError("bundle joint_names length does not match action size")
        if self.default_pose.shape != (self.nu,):
            raise PolicyError(
                f"bundle default_pose_rad has {self.default_pose.size} entries "
                f"but the policy drives {self.nu} joints"
            )

    # -- inference ---------------------------------------------------------

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.mean) / self.std

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action in [-1, 1]. This is brax's ``mode()``."""
        if obs.shape[-1] != self.obs_size:
            raise PolicyError(f"expected obs of size {self.obs_size}, got {obs.shape[-1]}")
        h = self.normalize(obs)
        for index, (weight, bias) in enumerate(self.layers):
            h = h @ weight + bias
            if index < len(self.layers) - 1:
                h = swish(h)
        return np.tanh(h[..., : self.nu])

    def action_to_targets(self, action: np.ndarray) -> np.ndarray:
        """Action -> joint position targets in radians, exactly as the env does.

        ``ctrl = clip(default_pose + action * action_scale, xml_range)``. The
        caller must still clamp to the *calibrated* limits; this reproduces the
        simulator, not the safety envelope.
        """
        action = np.clip(action, -1.0, 1.0)
        return np.clip(
            self.default_pose + action * self.action_scale,
            self.xml_lower,
            self.xml_upper,
        )

    # -- distribution check ------------------------------------------------

    def z_scores(self, obs: np.ndarray) -> np.ndarray:
        """How many training standard deviations from the training mean."""
        return (obs - self.mean) / self.std

    def out_of_distribution(
        self, obs: np.ndarray, limit: float
    ) -> list[tuple[int, str, float, float]]:
        """Observation elements that are implausibly far from training.

        This is the cheapest available detector for the sim-to-real mistakes
        that matter most: a swapped IMU axis, an accelerometer still in g, a
        gyro still in deg/s, a joint sign flip. Those do not look like small
        errors in normalized space -- ``grav_z`` alone has a training std of
        0.02, so pointing it the wrong way lands ~50 sigma out. Anything the
        policy has genuinely seen sits within a few sigma.

        Returns ``(index, label, value, z)`` for each offender.
        """
        z = self.z_scores(obs)
        return [
            (i, self.layout.label(i), float(obs[i]), float(z[i]))
            for i in np.nonzero(np.abs(z) > limit)[0]
        ]

    def describe(self) -> str:
        lines = [
            f"policy: obs={self.obs_size} nu={self.nu} "
            f"action_scale={self.action_scale} "
            f"trained at {1.0 / self.control_dt:.1f} Hz",
            f"  joints (MuJoCo order): {', '.join(self.joint_names)}",
        ]
        if np.any(np.abs(self.default_pose) > 1e-9):
            lines.append(
                "  residual about a non-zero nominal pose (deg): "
                + " ".join(f"{np.degrees(v):+.1f}" for v in self.default_pose)
            )
        if self.gait:
            lines.append(
                f"  gait: period {self.gait.get('gait_period_s')}s "
                f"duty {self.gait.get('duty_factor')} "
                f"phase step {self.gait.get('phase_increment_per_cycle'):.6f} "
                f"cmd_vx {self.gait.get('cmd_vx_range')}"
            )
        return "\n".join(lines)
