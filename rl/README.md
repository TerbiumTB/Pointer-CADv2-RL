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
python -m preprocessing.build_rl_episode_index -c config/rl_dataset.yaml
```

The rollout generator reads `config/rl_rollouts.yaml`:

```bash
python -m preprocessing.generate_rl_rollouts -c config/rl_rollouts.yaml
```

It loads the SFT checkpoint, samples several full trajectories for every
episode, executes CAD operations progressively and writes one immutable
Parquet shard per `generation.write_shard_size` completed trajectories. Every
step stores the exact B-Rep graph visible before generation and behavior-policy
log-probabilities for all four PointerCAD channels.

Set `generation.max_episodes_per_split` or `generation.task_ids` for a smoke
subset. A stopped run can be continued with `generation.resume: true`, but the
stored and requested data-affecting configurations must match. Resume removes
only uncommitted files for the deterministic trajectory being regenerated.

The preference builder reads `config/rl_preferences.yaml`:

```bash
python -m preprocessing.build_rl_preferences -c config/rl_preferences.yaml
```

All builders require `pyarrow`. Rollout generation additionally requires the
full PointerCAD, DGL and OpenCascade environment.

The same commands have launch scripts that activate `pointercad-rl-local`,
resolve paths relative to the repository and save console logs:

```bash
./scripts/rl_build_episodes.sh
./scripts/rl_generate_rollouts.sh
./scripts/rl_build_preferences.sh
./scripts/dpo_train.sh
```

Settings can be overridden without editing a script:

```bash
CONFIG_PATH=/path/to/rl_rollouts.yaml GPU_ID=2 \
  ./scripts/rl_generate_rollouts.sh

CONFIG_PATH=/path/to/dpo_train.yaml GPU_IDS=0,1,2,3 NUM_GPUS=4 \
  ./scripts/dpo_train.sh
```

Common overrides are `CONDA_ROOT`, `CONDA_ENV_NAME`, `HF_HOME`,
`CONFIG_PATH` and `LOG_DIR`. Rollout generation currently uses one GPU;
multi-GPU configuration applies to DPO training through Accelerate.

## Full-episode DPO

`dpo_train.py` implements the baseline DPO loop without adapting trajectories
to TRL's text-only completion schema. PointerCAD probabilities are evaluated
over every stored environment state and summed over:

- plan/LM tokens;
- structured labels;
- conditional length/angle selections;
- conditional B-Rep pointers.

Every `StepRecord` used for DPO must contain `state_before_graph_path`, relative
to its rollout run. The graph must describe the exact B-Rep visible to the
policy before that stored action. This lets DPO replay a generated episode
without executing OpenCascade inside every optimization step.

Configure `config/dpo_train.yaml`, then launch:

```bash
accelerate launch dpo_train.py -c config/dpo_train.yaml
```

Reference modes:

- `model`: load a frozen reference checkpoint and score it online;
- `cached`: read `ScoreRecord.total_logp` for `reference.checkpoint_hash`.

The default uses `model` and therefore needs memory for both policy and frozen
reference. `cached` avoids the second model, but rollout preparation must have
materialized scores for every referenced trajectory.

## Naming

Dataset-facing preference fields are consistently named `preferred_*` and
`rejected_*`. Heavy generated files live in `data/`; the name `artifacts` is not
used by this data format.
