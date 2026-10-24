# PPO Hyperparameter Optimization Report

## Scope and audit

The original repository contained no PPO checkpoint or baseline metric artifact, so the stated historical baseline could not be reconstructed. The PPO adapter was also using `Discrete(17)` actions despite the continuous-action requirement. It was repaired to expose a one-dimensional continuous action in `[-1, 1]`, mapped to physical power `[-80, 80]`; DQN remained discrete. The focused test suite passed 11/11 after the repair.

## Protocol

All candidates used the same pure-profit reward, training days 0-299, seeds 0, 1, and 2, and 48,000 requested environment steps. SB3 completed partial final rollouts, producing 48,128 steps for 1,024 rollouts and 49,152 for 2,048/4,096 rollouts. Validation selection used days 260-299 only. The final frozen evaluation used days 300-364 only.

## Selection

The staged search selected `n_steps=4096`, learning rate `0.0001`, `n_epochs=4`, batch size `128`, clip range `0.3`, and max gradient norm `0.5`. Mean validation profit was 5.289 per day across seeds, with zero measured safety violations. The gradient-norm 1.0 rerun matched the 0.5 result and was not selected.

## Final held-out test

The selected configuration achieved mean daily profit 5.799 across the three seeds on days 300-364. Per-seed details are in `final_test_summary.csv`. Maximum measured safety violation was zero. Mean throughput and variability are reported there.

## Limitations

The historical baseline scores supplied in the task were not present in the workspace, and no equivalent fresh baseline final-test run was completed before freezing the selected configuration. Therefore this report does not claim a baseline improvement. Only validation data drove selection; test results were generated afterward and were not used to revise the configuration. The requested GAE, gamma, initial log standard deviation, entropy, hidden-layer, and value-function coefficient candidates are included in the registry.

## Artifacts

`experiment_registry.csv`, `validation_summary.csv`, `final_test_summary.csv`, `best_config.json`, `best_checkpoint`, and the three PNG plots in this directory contain the recorded outputs. Checkpoint paths and per-seed metadata are preserved under `results/hpo/clip_03/`.
