"""Export a trained brax PPO policy into a self-contained deployment bundle.

Run this on the TRAINING machine (it needs jax + brax + mujoco). It produces a
single ``.npz`` that the robot loads with nothing but numpy -- no jax on the
Jetson.

    python3 export_policy.py --params stand_params.pkl --xml robot/robot.xml \
                             --out policy_bundle.npz

Why the bundle carries more than weights
----------------------------------------
The deployment has to agree with training on things that live in the MuJoCo
model, not in the checkpoint: which joint is element 3 of the observation,
what the action scale was, what the control period was. Re-typing any of those
into a YAML file is an invitation for a silent mismatch -- and a joint-order
mismatch means every command goes to the wrong motor and the robot thrashes.

So the bundle is extracted from the same two files that produced the policy,
in one step, and the robot refuses to run without it.

The exporter also *verifies* the numpy forward pass against brax's own
inference function on random observations before writing anything. If those
disagree, the bundle is not written.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def swish(x):
    return x / (1.0 + np.exp(-x))


def numpy_forward(obs, mean, std, layers):
    """The exact computation the robot will run. Mirrors brax's MLP + normalize."""
    h = (obs - mean) / std
    for index, (weight, bias) in enumerate(layers):
        h = h @ weight + bias
        if index < len(layers) - 1:  # brax MLP: activate_final=False
            h = swish(h)
    loc = h[..., : h.shape[-1] // 2]  # NormalTanhDistribution: (loc, scale)
    return np.tanh(loc)  # .mode() = tanh(loc); scale is unused when deterministic


def extract(params):
    normalizer, policy = params[0], params[1]
    mean = np.asarray(normalizer.mean, dtype=np.float64)
    std = np.asarray(normalizer.std, dtype=np.float64)

    tree = policy["params"]
    layers = []
    for index in range(len(tree)):
        key = f"hidden_{index}"
        if key not in tree:
            raise SystemExit(f"unexpected policy structure: no '{key}'")
        layers.append(
            (
                np.asarray(tree[key]["kernel"], dtype=np.float64),
                np.asarray(tree[key]["bias"], dtype=np.float64),
            )
        )
    return mean, std, layers


def resolve_frame_skip(timestep, control_rate_hz, override=None):
    """Reproduce ``MuJoCoBipedStandMJX``'s rate -> frame_skip conversion exactly.

    The env takes a control *rate* and rounds it to a whole number of physics
    steps; the achieved rate is then ``1/(frame_skip*timestep)``, which is not
    always the number that was asked for. Deriving it the same way here is the
    point: the bundle must record the rate the policy was actually trained at,
    not the one on the command line.
    """
    if override is not None:
        frame_skip = int(override)
    else:
        frame_skip = int(round(1.0 / (control_rate_hz * timestep)))
    if frame_skip < 1:
        raise SystemExit(
            f"control_rate_hz={control_rate_hz} is faster than the physics "
            f"timestep {timestep}s"
        )
    actual = 1.0 / (frame_skip * timestep)
    if override is None and abs(actual - control_rate_hz) > 0.01:
        print(f"note  : {control_rate_hz} Hz is not an integer multiple of the "
              f"{timestep}s timestep; the env used {actual:.2f} Hz "
              f"(frame_skip={frame_skip}) and so does this bundle")
    return frame_skip, actual


def model_metadata(xml_path, control_rate_hz, frame_skip_override, action_scale):
    """Pull joint order, limits and timing straight from the trained model."""
    import mujoco

    mj = mujoco.MjModel.from_xml_path(xml_path)
    frame_skip, actual_hz = resolve_frame_skip(
        mj.opt.timestep, control_rate_hz, frame_skip_override)

    # Actuator order IS the action order and (for this robot, one actuator per
    # hinge in declaration order) the observation's joint order. We verify that
    # assumption below rather than trusting it.
    act_names, joint_names, lo, hi = [], [], [], []
    for i in range(mj.nu):
        act_names.append(mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
        jid = int(mj.actuator_trnid[i, 0])
        joint_names.append(mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, jid))
        lo.append(float(mj.jnt_range[jid, 0]))
        hi.append(float(mj.jnt_range[jid, 1]))

    # The observation slices qpos[7:] / qvel[6:], i.e. joints in *joint* order.
    # If actuator order differs from joint order, obs and action would be
    # permuted relative to each other and everything silently breaks.
    hinge_order = []
    for jid in range(mj.njnt):
        if mj.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            hinge_order.append(mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, jid))
    if hinge_order != joint_names:
        raise SystemExit(
            "actuator order != joint order in the XML.\n"
            f"  joints in qpos order : {hinge_order}\n"
            f"  joints in actuator   : {joint_names}\n"
            "The observation uses qpos order and the action uses actuator order,\n"
            "so these must match. Reorder the <actuator> block in robot.xml."
        )

    return {
        "joint_names": joint_names,
        "actuator_names": act_names,
        "xml_lower_rad": lo,
        "xml_upper_rad": hi,
        "action_scale": float(action_scale),
        "frame_skip": int(frame_skip),
        "physics_timestep": float(mj.opt.timestep),
        "control_dt": float(mj.opt.timestep * frame_skip),
        "control_rate_hz": float(actual_hz),
        "nq": int(mj.nq),
        "nv": int(mj.nv),
        "nu": int(mj.nu),
        "default_pose_rad": [0.0] * mj.nu,  # the env's target is literally 0 rad
    }


def verify_against_brax(params, mean, std, layers, obs_size, act_size, samples=512):
    """Numpy vs brax on random observations. Must match to float32 precision.

    Matmul precision is forced to 'highest' and the reference is evaluated on
    the CPU. Without that, an Ampere-or-newer GPU silently computes float32
    matmuls in TF32 -- about 10 mantissa bits -- and the comparison drifts by
    ~5e-3, which says nothing about whether this port is correct. The float64
    numpy path is in fact the more accurate of the two; the point of this check
    is to confirm the *structure* (activation, layer order, normalisation,
    tanh) matches, so both sides need to be computed the same way.
    """
    import jax
    import jax.numpy as jnp
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics

    jax.config.update("jax_default_matmul_precision", "highest")

    # Infer the architecture from the checkpoint rather than restating the
    # training script's constants -- otherwise this "verification" would only
    # confirm that two copies of the same guess agree.
    hidden = tuple(int(weight.shape[1]) for weight, _ in layers[:-1])
    nets = ppo_networks.make_ppo_networks(
        observation_size=obs_size,
        action_size=act_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=hidden,
        value_hidden_layer_sizes=hidden,
    )
    make_policy = ppo_networks.make_inference_fn(nets)
    policy = make_policy((params[0], params[1]), deterministic=True)

    rng = np.random.default_rng(0)
    # Cover both the normalised operating region and large out-of-distribution
    # excursions, which is exactly where a wrong activation would show up.
    obs = np.concatenate([
        mean + std * rng.normal(size=(samples, obs_size)),
        mean + 8.0 * std * rng.normal(size=(samples, obs_size)),
    ]).astype(np.float32)

    try:
        cpu = jax.devices("cpu")[0]
    except RuntimeError:  # pragma: no cover - cpu backend is always present
        cpu = None

    key = jax.random.PRNGKey(0)
    batched = jax.jit(jax.vmap(lambda o: policy(o, key)[0]))
    if cpu is not None:
        with jax.default_device(cpu):
            ref = np.asarray(batched(jnp.asarray(obs)))
    else:
        ref = np.asarray(batched(jnp.asarray(obs)))
    ours = numpy_forward(obs.astype(np.float64), mean, std, layers)
    err = float(np.max(np.abs(ref - ours)))
    return err, ref, ours


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="stand_params.pkl")
    parser.add_argument("--xml", default="robot/robot.xml")
    parser.add_argument("--out", default="policy_bundle.npz")
    parser.add_argument("--action-scale", type=float, default=0.4,
                        help="must equal the env's action_scale (default 0.4)")
    parser.add_argument("--control-rate-hz", type=float, default=25.0,
                        help="must equal the env's control_rate_hz; 25 Hz is "
                             "the rate the STS bus benchmark supports")
    parser.add_argument("--frame-skip", type=int, default=None,
                        help="explicit override, only if you passed frame_skip "
                             "to the env directly instead of a rate")
    parser.add_argument("--tolerance", type=float, default=2e-5)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    from brax.io import model

    params = model.load_params(args.params)
    mean, std, layers = extract(params)

    obs_size = int(layers[0][0].shape[0])
    act_size = int(layers[-1][0].shape[1] // 2)
    print(f"policy: obs={obs_size} act={act_size} "
          f"hidden={[int(layer[0].shape[1]) for layer in layers[:-1]]}")

    if np.any(std <= 0) or not np.all(np.isfinite(std)):
        raise SystemExit("normalizer std has non-positive or non-finite entries")

    meta = model_metadata(args.xml, args.control_rate_hz, args.frame_skip,
                          args.action_scale)
    if meta["nu"] != act_size:
        raise SystemExit(f"XML has {meta['nu']} actuators but policy emits {act_size}")
    expected_obs = 9 + (meta["nq"] - 7) + (meta["nv"] - 6) + meta["nu"]
    if expected_obs != obs_size:
        raise SystemExit(
            f"XML implies obs size {expected_obs} but the policy takes {obs_size}"
        )
    print(f"model : {meta['nu']} joints, control dt={meta['control_dt']:.4f}s "
          f"({meta['control_rate_hz']:.1f} Hz, frame_skip={meta['frame_skip']}, "
          f"physics {meta['physics_timestep']}s), action_scale={meta['action_scale']}")
    for name, lo, hi in zip(meta["joint_names"], meta["xml_lower_rad"], meta["xml_upper_rad"]):
        print(f"        {name:<20} [{np.degrees(lo):+7.1f}, {np.degrees(hi):+7.1f}] deg")

    if not args.skip_verify:
        err, ref, ours = verify_against_brax(params, mean, std, layers, obs_size, act_size)
        print(f"verify: max |numpy - brax| = {err:.3e} over 1024 random observations")
        if err > args.tolerance:
            raise SystemExit(
                f"FAILED: numpy port disagrees with brax by {err:.3e} "
                f"(tolerance {args.tolerance:.1e}). Bundle NOT written."
            )
        print(f"        action range seen: [{ours.min():+.3f}, {ours.max():+.3f}]")

    payload = {"mean": mean.astype(np.float32), "std": std.astype(np.float32),
               "meta": np.array(json.dumps(meta))}
    for index, (weight, bias) in enumerate(layers):
        payload[f"w{index}"] = weight.astype(np.float32)
        payload[f"b{index}"] = bias.astype(np.float32)
    payload["n_layers"] = np.array(len(layers))

    np.savez(args.out, **payload)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
