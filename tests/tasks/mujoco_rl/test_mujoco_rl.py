"""Tests for the MuJoCo PPO control benchmark (src/tasks/mujoco_rl)."""
import numpy as np
import pytest
import tensorflow as tf

from experiments.run_mujoco_benchmark import PROFILES, build_specs
from src.benchmark.registry import cell_kwargs as registry_cell_kwargs
from src.tasks.mujoco_rl.evaluation import EVAL_EPISODES, EVAL_SEED0, evaluate_policy
from src.tasks.mujoco_rl.policies import (GaussianPolicy, MODEL_TYPES,
                                          size_for_param_budget)
from src.tasks.mujoco_rl.ppo import _compute_gae, PPOConfig, PPOTrainer, train
from src.tasks.mujoco_rl import record as rec
from src.wirings import effective_param_count

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
    assert set(MODEL_TYPES) == {"mlp", "lstm", "gru", "ctrnn", "ltc", "lrc", "cfc", "cfc_lrc"}


# --------------------------------------------------- registry kwargs (issue #61)
def test_cfc_lrc_policy_steps_with_registry_kwargs():
    p = GaussianPolicy("cfc_lrc", OBS_DIM, ACT_DIM, units=8, seed=0)
    assert p.cell_kwargs == registry_cell_kwargs("cfc_lrc") == {"elastance_type": "asymmetric"}
    assert p.actor_cell._elastance_type == p.critic_cell._elastance_type == "asymmetric"
    astate, cstate = p.initial_state(3)
    mean, a2 = p.actor_forward(tf.zeros([3, OBS_DIM]), astate)
    value, c2 = p.value_forward(tf.zeros([3, OBS_DIM]), cstate)
    assert mean.shape == (3, ACT_DIM) and value.shape == (3,)
    assert len(a2) == len(c2) == 1


@pytest.mark.parametrize("model_type", [m for m in MODEL_TYPES if m != "mlp"])
def test_policy_cells_take_registry_kwargs(model_type):
    # default (None) = registry; an explicit dict wins; {} means no kwargs
    default = GaussianPolicy(model_type, OBS_DIM, ACT_DIM, units=6, seed=0)
    assert default.cell_kwargs == registry_cell_kwargs(model_type)
    bare = GaussianPolicy(model_type, OBS_DIM, ACT_DIM, units=6, cell_kwargs={}, seed=0)
    assert bare.cell_kwargs == {}


def test_lrc_policy_default_is_asymmetric_and_ode_unfolds_pass_through():
    assert GaussianPolicy("lrc", OBS_DIM, ACT_DIM, units=6).actor_cell._elastance_type == "asymmetric"
    kw = {**registry_cell_kwargs("lrc"), "ode_unfolds": 3}
    p = GaussianPolicy("lrc", OBS_DIM, ACT_DIM, units=6, cell_kwargs=kw)
    assert p.actor_cell._ode_unfolds == p.critic_cell._ode_unfolds == 3


def test_checkpoint_keeps_cell_kwargs_and_legacy_checkpoints_load_bare(tmp_path):
    p = GaussianPolicy("lrc", OBS_DIM, ACT_DIM, units=6, seed=1)
    path = str(tmp_path / "ckpt.npz")
    rec.save_policy(path, p, np.zeros(OBS_DIM, np.float32), np.ones(OBS_DIM, np.float32))
    assert rec.load_policy(path)[3]["cell_kwargs"] == {"elastance_type": "asymmetric"}
    # a checkpoint written before the registry kwargs: built bare, no cell_kwargs entry
    bare = GaussianPolicy("lrc", OBS_DIM, ACT_DIM, units=6, cell_kwargs={}, seed=1)
    legacy = str(tmp_path / "legacy.npz")
    rec.save_policy(legacy, bare, np.zeros(OBS_DIM, np.float32), np.ones(OBS_DIM, np.float32))
    old = {k: v for k, v in np.load(legacy).items() if k != "cell_kwargs"}
    np.savez(legacy, **old)
    q, _, _, meta = rec.load_policy(legacy)
    assert meta["cell_kwargs"] == {} and q.actor_cell._elastance_type != "asymmetric"
    obs = tf.random.normal([2, OBS_DIM])
    np.testing.assert_allclose(q.actor_forward(obs, q.initial_state(2)[0])[0].numpy(),
                               bare.actor_forward(obs, bare.initial_state(2)[0])[0].numpy(),
                               atol=1e-5)


# ------------------------------------------------------ parameter budget sizing
@pytest.mark.parametrize("model_type", ["mlp", "gru", "cfc_lrc"])
def test_size_for_param_budget_counts_actor_plus_critic(model_type):
    budget = 4000
    units, n = size_for_param_budget(model_type, OBS_DIM, ACT_DIM, budget, seed=0)
    pol = GaussianPolicy(model_type, OBS_DIM, ACT_DIM, units=units, seed=0)
    assert effective_param_count(pol) == n
    # n counts both torsos, both heads and log_std: more than one torso alone
    one_torso = sum(int(np.prod(v.shape)) for layer in
                    (pol.actor_cell if pol.recurrent else pol.actor_torso,)
                    for v in layer.trainable_variables)
    assert n > 2 * one_torso
    # closest width: the neighbours are at least as far from the budget
    for other in (units - 1, units + 1):
        q = GaussianPolicy(model_type, OBS_DIM, ACT_DIM, units=other, seed=0)
        assert abs(effective_param_count(q) - budget) >= abs(n - budget)


def test_trainer_reports_params_effective_and_budget_deviation():
    cfg = PPOConfig(num_envs=2, num_steps=8, total_steps=16, num_minibatches=2,
                    param_budget=12000)
    t = PPOTrainer("gru", "HalfCheetah-v5", seed=0, cfg=cfg)
    t.envs.close()
    assert t.policy.units != cfg.units                    # sized, not the default 64
    assert t.params_effective == effective_param_count(t.policy)
    assert t.budget_deviation == pytest.approx(abs(t.params_effective - 12000) / 12000)
    assert t.budget_deviation < 0.05


# ----------------------------------------- evaluation bookkeeping (stub env)
class _StubEnv:
    """Episode length 3 + seed % 3, reward 1 + seed % 5 per step, no physics."""

    def __init__(self, log):
        self.log, self.closed = log, False

    def reset(self, seed=None):
        self.log.append(seed)
        self.seed_, self.t = seed, 0
        self.n = 3 + seed % 3
        return np.full(OBS_DIM, float(seed % 7), np.float32), {}

    def step(self, action):
        assert action.shape == (ACT_DIM,) and np.abs(action).max() <= 1.0
        self.t += 1
        done = self.t >= self.n
        return (np.zeros(OBS_DIM, np.float32), 1.0 + self.seed_ % 5,
                False, done, {})

    def close(self):
        self.closed = True


@pytest.mark.parametrize("model_type", ["mlp", "gru"])
def test_evaluate_policy_bookkeeping_with_stub_env(model_type):
    pol = GaussianPolicy(model_type, OBS_DIM, ACT_DIM, units=6, seed=0)
    log = []
    ev = evaluate_policy(pol, "unused", np.zeros(OBS_DIM, np.float32),
                         np.ones(OBS_DIM, np.float32), env_fn=lambda: _StubEnv(log))
    seeds = [EVAL_SEED0 + i for i in range(EVAL_EPISODES)]
    assert EVAL_EPISODES == 20 and EVAL_SEED0 == 12345
    assert sorted(log) == seeds and ev["seeds"] == seeds       # one episode per fixed seed
    lengths = [3 + s % 3 for s in seeds]                       # episodes end independently
    returns = [n * (1.0 + s % 5) for n, s in zip(lengths, seeds)]
    assert ev["lengths"] == lengths
    np.testing.assert_allclose(ev["returns"], returns)
    assert ev["mean"] == pytest.approx(np.mean(returns))
    assert ev["std"] == pytest.approx(np.std(returns, ddof=1))


def test_evaluate_policy_is_deterministic_and_independent_of_batch_mates():
    pol = GaussianPolicy("gru", OBS_DIM, ACT_DIM, units=6, seed=0)
    args = (pol, "unused", np.zeros(OBS_DIM, np.float32), np.ones(OBS_DIM, np.float32))
    full = evaluate_policy(*args, env_fn=lambda: _StubEnv([]))
    again = evaluate_policy(*args, env_fn=lambda: _StubEnv([]))
    assert full["returns"] == again["returns"]
    solo = evaluate_policy(*args, n_episodes=1, seed0=EVAL_SEED0 + 7, env_fn=lambda: _StubEnv([]))
    assert solo["returns"][0] == pytest.approx(full["returns"][7])
    assert solo["std"] == 0.0


# --------------------------------------------------------- runner profiles
def _per_model_seeds(profile):
    specs = build_specs(PROFILES[profile]["models"], PROFILES[profile]["seeds"])
    return {m: [s["seed"] for s in specs if s["model"] == m]
            for m in PROFILES[profile]["models"]}


def test_final_profile_models_seeds_and_budget():
    prof = PROFILES["final"]
    assert prof["models"] == ["mlp", "gru", "cfc", "cfc_lrc"]
    assert _per_model_seeds("final") == {
        "mlp": list(range(5)), "gru": list(range(10)),
        "cfc": list(range(5)), "cfc_lrc": list(range(10))}
    assert prof["total_steps"] == 1_000_000 and prof["param_budget"] == 12_000


def test_final_numeric_adds_lrc_and_ltc_with_one_unfold():
    prof = PROFILES["final_numeric"]
    assert prof["models"] == PROFILES["final"]["models"] + ["lrc", "ltc"]
    assert prof["ode_unfolds"] == 1 and prof["param_budget"] == 12_000
    seeds = _per_model_seeds("final_numeric")
    assert seeds["lrc"] == seeds["ltc"] == list(range(5))
    assert {m: seeds[m] for m in PROFILES["final"]["models"]} == _per_model_seeds("final")


def test_legacy_profiles_unchanged():
    assert len(build_specs(PROFILES["v1"]["models"], PROFILES["v1"]["seeds"])) == 21
    assert "param_budget" not in PROFILES["v1"]


@pytest.mark.slow
@pytest.mark.parametrize("model_type", ["mlp", "lrc", "cfc_lrc"])
def test_train_smoke_runs(model_type):
    cfg = PPOConfig(num_envs=4, num_steps=16, total_steps=256,
                    update_epochs=1, num_minibatches=2, units=8)
    out = train(model_type, "HalfCheetah-v5", seed=0, cfg=cfg)
    assert out["global_step"] >= 256
    assert out["obs_mean"].shape == (OBS_DIM,)
    assert out["policy"].model_type == model_type
    assert out["units"] == 8 and out["budget_deviation"] is None   # no budget set
    assert out["params_effective"] == effective_param_count(out["policy"])
