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
subset. Re-running the same run skips committed deterministic trajectory IDs
and generates only missing trajectories; the stored and requested
data-affecting configurations must match. Pass `--force` (`-f`) to delete the
whole `<output_root>/<run_id>` directory and generate it from scratch. A normal
repeat run removes only uncommitted files for a deterministic trajectory that
still needs to be generated.

The preference builder reads `config/rl_preferences.yaml`:

```bash
python -m preprocessing.build_rl_preferences -c config/rl_preferences.yaml
```

When rollouts were sampled from the frozen DPO reference checkpoint with all
four temperatures equal to `1.0`, materialize cached reference scores directly
from their stored behavior log-probabilities:

```bash
python -m preprocessing.build_rl_reference_scores \
    -c config/rl_reference_scores.yaml
```

The builder verifies the checkpoint SHA256 against the rollout run config,
sums all four channels over every stored step and appends idempotent
`ScoreRecord` shards. It does not run model inference or CAD execution.

All builders require `pyarrow`. Rollout generation additionally requires the
full PointerCAD, DGL and OpenCascade environment.

The same commands have launch scripts that activate `pointercad-rl-local`,
resolve paths relative to the repository and save console logs:

```bash
./scripts/rl_build_episodes.sh
./scripts/rl_generate_rollouts.sh
./scripts/rl_build_preferences.sh
./scripts/rl_build_reference_scores.sh
./scripts/dpo_train.sh
```

Settings can be overridden without editing a script:

```bash
./scripts/rl_build_episodes.sh -c /path/to/rl_dataset.yaml
./scripts/rl_generate_rollouts.sh --config /path/to/rl_rollouts.yaml
./scripts/rl_build_preferences.sh -c /path/to/rl_preferences.yaml
./scripts/rl_build_reference_scores.sh -c /path/to/rl_reference_scores.yaml
NUM_GPUS=4 ./scripts/dpo_train.sh --config /path/to/dpo_train.yaml --test
```

The `CONFIG_PATH` environment variable remains available as an alternative.
Common overrides are `CONDA_ENV_NAME`, `HF_HOME`, `CONFIG_PATH` and `LOG_DIR`.
The scripts preserve scheduler-provided `CUDA_VISIBLE_DEVICES` instead of
overwriting it. Without `--test` they load `~/dist_env.sh` or fall back to
SenseCore variables. Rollout generation currently uses one visible GPU;
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

DPO enables deterministic layer-wise gradient checkpointing by default through
`training.gradient_checkpointing`. Decoder layers are recomputed during
backward while dropout and BatchNorm remain in eval mode. The LM selected-token
loss is also checkpointed in small vocabulary chunks, avoiding retained
float32 `[tokens, vocabulary]` intermediates. These settings trade additional
compute for substantially lower full-episode activation memory.

`model.max_input_length` must match the rollout generator's input truncation.
`model.max_replay_length` is a separate, larger bound for that input plus the
stored generated positions; with the default 3072 input and 1024 generation
limits it is 4096. Exact replay never silently truncates a stored completion.

Standard CAD evaluation does not create these scores: it generates new
deterministic trajectories and geometry metrics instead of scoring the stored
rollout actions. For behavior-derived scores, set
`reference.checkpoint_hash` to the exact SHA256 printed by the reference-score
builder and switch `reference.mode` to `cached`.

## Naming

Dataset-facing preference fields are consistently named `preferred_*` and
`rejected_*`. Heavy generated files live in `data/`; the name `artifacts` is not
used by this data format.
