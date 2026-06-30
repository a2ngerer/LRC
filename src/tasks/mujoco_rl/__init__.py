"""MuJoCo continuous-control RL benchmark.

PPO with the thesis cells (LTC / LRC / CfC / CT-RNN) and classical baselines
(LSTM / GRU / MLP) as recurrent policies on a single MuJoCo locomotion task
(HalfCheetah-v5 by default). The point of comparison: can the continuous-time /
liquid neuron types serve as RL control policies, and how do they compare to
gated-memory and feedforward baselines at learning to drive a simulated body.

Layout:
    envs.py      vectorized + normalized training envs, obs-stat persistence
    policies.py  Gaussian actor-critic, feedforward or single-cell recurrent torso
    ppo.py       PPO trainer (GAE, clipped objective, truncated-BPTT for cells)
    record.py    deterministic rollout -> rendered mp4 with an info overlay
"""
