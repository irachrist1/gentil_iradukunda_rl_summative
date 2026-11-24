# District Food Dispatch

A reinforcement learning benchmark for simulated emergency food distribution in Rwanda. One dispatcher chooses which district receives a shipment and when to replenish the shared hub. All demand, alert and road data are synthetic.

## Run with uv

From the repository root:

```sh
uv sync --locked
uv run main.py
```

`main.py` opens the best learned policy in the simulator. The same replay can be run with the following commands.

```sh
uv run play.py --seed 20000 --delay 5
uv run play.py --algorithm dqn --seed 20000 --headless
uv run play.py --seed 30000 --stress --headless
uv run play.py --headless --seed 20000 --capture assets/simulator.png --trace logs/demo-trace.json
```

Replay reads a local checkpoint, runs the same environment and preprocessing used during training, and prints one line per simulated day. OpenGL presents the 2D simulation when supported. The SDL display is the fallback. `--headless` runs without a window.

## Reproduce the evidence

```sh
uv run python -m unittest discover -s tests
uv run main.py smoke --jobs 4
uv run main.py train --jobs 4
uv run python -m training.plots
```

The full run reuses completed artifacts with the same configuration and training budget. To recompute from scratch in a disposable fresh copy, remove `logs/runs` from that copy before running the full training command. The smoke run uses its own directory, leaves the reported artifacts in place and is excluded from Git.

- `environment/custom_env.py` implements the Gymnasium environment. `environment/rendering.py` visualizes its live state.
- `training/config.py` defines ten distinct settings per algorithm. Each run consumes 24,576 environment transitions. Each selected setting is run with two additional training seeds.
- `training/dqn_training.py` and `training/pg_training.py` train the four policies with TorchRL actors and the appropriate learning objectives. The Monte Carlo REINFORCE objective uses the TorchRL actor distribution without a critic.
- `logs/runs` contains all 48 final checkpoints, configurations and training traces. `logs/sweep.json` reports validation results. `logs/evaluation.json` contains episode-level held-out returns, service and shortage metrics.
- `models/dqn` and `models/pg` contain the four seed-41 policies used in the report. Each has its algorithm, configuration number and training seed serialized with its PyTorch state dictionary.

Seed-41 runs choose each algorithm's configuration by final mean return over the same 32 validation scenarios. Held-out evaluation uses 96 new nominal scenarios and 96 stress scenarios. A random-action policy and a stock-aware dispatch rule run on the same scenarios. All evaluation policies use a fixed decision rule, with argmax for learned policies.

PPO run 04 leads the seed-41 learned policies at −12.42 nominal mean return and 91.5% of demand served. The dispatch rule serves 92.9% nominally and leads all evaluated policies at −10.51. The demo shows what the learned policy does and where it falls short.
