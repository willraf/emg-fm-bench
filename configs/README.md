# Config Structure

- `default.yaml` — entry point, defines defaults list
- `experiment/` — run-specific overrides (start here)
- `dataset/` — dataset configs (paths, splits, transforms)
- `model/` — architecture configs
- `training/` — patience, epochs and gradient accumultion
- `task/` — everything specific to the task. optimiser, scheduler, AMP settings
- `transforms/` — reusable transform definitions

## Common Usage
# Run a specific experiment
python -m scripts.main +experiment=test

# Override a single value
python -m scripts.main +experiment=test training.epochs=100

# Multi-GPU
python -m torch.distributed.run --nproc_per_node=4 -m scripts.main +experiment=test

# Full config file (e.g. if continuing previous experiment, saved in experiments/<experiment_name>/config.yaml)
python -m scripts.main --config_name main.yaml 