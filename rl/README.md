# PointerCAD RL data

The source PointerCAD dataset and its `train_val_test.json` remain the source of
truth. Everything below `format/pointercad_rl` is derived and can be rebuilt.

```text
format/pointercad_rl/
├── episodes/
│   ├── config.yaml
│   ├── train.parquet
│   ├── validation.parquet
│   └── test.parquet
├── rollouts/
│   └── <run_id>/
│       ├── config.yaml
│       ├── trajectories/
│       │   └── part-*.parquet
│       ├── steps/
│       │   └── part-*.parquet
│       ├── scores/
│       │   └── part-*.parquet
│       └── data/
│           ├── cad/
│           ├── mesh/
│           └── states/
└── preferences/
    └── <view_id>/
        ├── config.yaml
        ├── train.parquet
        └── validation.parquet
```

## Responsibilities

- `episodes` groups the existing step-level split into full-model tasks. It does
  not contain generated responses, rewards or policy information.
- `rollouts` stores complete generated trajectories, their individual steps,
  raw geometric metrics, checkpoint scores and heavy generated files under
  `data/`.
- `preferences` stores only lightweight references from a preferred trajectory
  to a rejected trajectory. Rebuilding a reward or pairing strategy does not
  require regenerating rollouts.

## Builders

The episode builder reads `config/rl_dataset.yaml`:

```bash
python preprocessing/build_rl_episode_index.py -c config/rl_dataset.yaml
```

The preference builder reads `config/rl_preferences.yaml`:

```bash
python preprocessing/build_rl_preferences.py -c config/rl_preferences.yaml
```

Both commands require `pyarrow`. They are intentionally separate from rollout
generation: generating and executing trajectories will be implemented against
the same schemas in `rl/schemas.py`.

## Naming

Dataset-facing preference fields are consistently named `preferred_*` and
`rejected_*`. Heavy generated files live in `data/`; the name `artifacts` is not
used by this data format.
