"""Tests for the MuJoCo PPO control benchmark (src/tasks/mujoco_rl)."""
import numpy as np
import pytest
import tensorflow as tf

from src.tasks.mujoco_rl.policies import GaussianPolicy, MODEL_TYPES
from src.tasks.mujoco_rl.ppo import _compute_gae, PPOConfig, train
from src.tasks.mujoco_rl import record as rec

OBS_DIM, ACT_DIM = 17, 6


def test_gae_matches_hand_computed():
    # gamma=lam=1 -> advantage+value collapses to the bootstrapped reward-to-go.
    rewards = np.array([[1.0], [1.0]], np.float32)
    values = np.array([[0.5], [0.5]], np.float32)
    dones = np.array([[0.0], [0.0]], np.float32)
    adv, ret = _compute_gae(rewards, values, dones,
                            next_value=np.array([0.0], np.float32),
                            next_done=np.array([0.0], np.float32),
                            gamma=1.0, lam=1.0)
    np.testing.assert_allclose(ret[:, 0], [2.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(adv[:, 0], [1.5, 0.5], atol=1e-5)


def test_gae_terminal_stops_bootstrap():
    # A done entering step 1 must cut the bootstrap from step 1 into step 0.
    rewards = np.array([[1.0], [1.0]], np.float32)
    values = np.array([[0.5], [0.5]], np.float32)
    dones = np.array([[0.0], [1.0]], np.float32)  # step 1 begins a fresh episode
    adv, ret = _compute_gae(rewards, values, dones,
                            next_value=np.array([9.9], np.float32),
                            next_done=np.array([0.0], np.float32),
                            gamma=1.0, lam=1.0)
    # step 0: no bootstrap across the boundary -> delta = 1 + 0 - 0.5 = 0.5
    np.testing.assert_allclose(adv[0, 0], 0.5, atol=1e-5)


@pytest.mark.parametrize("model_type", ["mlp", "ltc", "lrc", "lstm"])
def test_policy_forward_shapes(model_type):
    p = GaussianPolicy(model_type, OBS_DIM, ACT_DIM, units=16, seed=0)
    obs = tf.zeros([4, OBS_DIM])
    astate, cstate = p.initial_state(4)
    mean, a2 = p.actor_forward(obs, astate)
    value, c2 = p.value_forward(obs, cstate)
    assert mean.shape == (4, ACT_DIM)
    assert value.shape == (4,)
    if model_type == "lstm":
        assert isinstance(astate, list) and len(astate) == 2  # [h, c]
    elif model_type != "mlp":
        assert isinstance(astate, list) and len(astate) == 1
    # log-prob + entropy are finite
    act = p.sample_action(mean)
    assert np.isfinite(p.logprob(mean, act).numpy()).all()
    assert np.isfinite(float(p.entropy()))


def test_state_reset_zeros_done_envs():
    p = GaussianPolicy("ltc", OBS_DIM, ACT_DIM, units=8, seed=0)
    state = [tf.ones([3, 8])]
    done = tf.constant([[1.0], [0.0], [1.0]])
    reset = p.reset_state(state, done)
    np.testing.assert_allclose(reset[0].numpy()[0], np.zeros(8))
    np.testing.assert_allclose(reset[0].numpy()[1], np.ones(8))
    np.testing.assert_allclose(reset[0].numpy()[2], np.zeros(8))


@pytest.mark.parametrize("model_type", ["mlp", "lrc", "lstm"])
def test_checkpoint_roundtrip(tmp_path, model_type):
    p = GaussianPolicy(model_type, OBS_DIM, ACT_DIM, units=12, seed=1)
    obs = tf.random.normal([2, OBS_DIM])
    a, _ = p.initial_state(2)
    mean_before, _ = p.actor_forward(obs, a)
    path = str(tmp_path / "ckpt.npz")
    rec.save_policy(path, p, np.zeros(OBS_DIM, np.float32), np.ones(OBS_DIM, np.float32))
    q, mean, var, meta = rec.load_policy(path)
    a2, _ = q.initial_state(2)
    mean_after, _ = q.actor_forward(obs, a2)
    np.testing.assert_allclose(mean_before.numpy(), mean_after.numpy(), atol=1e-5)
    assert meta["model_type"] == model_type


def test_all_model_types_known():
    assert set(MODEL_TYPES) == {"mlp", "lstm", "gru", "ctrnn", "ltc", "lrc", "cfc"}


@pytest.mark.slow
@pytest.mark.parametrize("model_type", ["mlp", "lrc"])
def test_train_smoke_runs(model_type):
    cfg = PPOConfig(num_envs=4, num_steps=16, total_steps=256,
                    update_epochs=1, num_minibatches=2, units=8)
    out = train(model_type, "HalfCheetah-v5", seed=0, cfg=cfg)
    assert out["global_step"] >= 256
    assert out["obs_mean"].shape == (OBS_DIM,)
    assert out["policy"].model_type == model_type
